import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import httpx
import tiktoken
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import ChunkConfig
from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .schema import Chunk, Paper
from .sections import section_spans

ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def parse_feed(xml: str, selection: str) -> list[Paper]:
    papers = []
    for entry in ET.fromstring(xml).findall("a:entry", ATOM):
        identifier = entry.findtext("a:id", "", ATOM).rsplit("/", 1)[-1]
        if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", identifier):
            continue
        pdf = next(
            (
                link.attrib["href"]
                for link in entry.findall("a:link", ATOM)
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )
        if pdf is None:
            continue
        papers.append(
            Paper(
                paper_id=identifier,
                title=" ".join(entry.findtext("a:title", "", ATOM).split()),
                authors=[a.findtext("a:name", "", ATOM) for a in entry.findall("a:author", ATOM)],
                year=int(entry.findtext("a:published", "", ATOM)[:4]),
                source_url=f"https://arxiv.org/abs/{identifier}",
                pdf_url=pdf.replace("http://", "https://"),
                abstract=" ".join(entry.findtext("a:summary", "", ATOM).split()),
                selection=selection,
                license=entry.findtext("arxiv:license", None, ATOM),
            )
        )
    return papers


class ArxivSource:
    def __init__(self, interval: float = 3.1):
        self.interval = interval
        self.last_request = 0.0
        self.client = httpx.Client(
            timeout=60,
            follow_redirects=False,
            headers={"User-Agent": "Jev-Agentic-RAG/0.1 research ingestion"},
        )

    def close(self):
        self.client.close()

    def get(self, url: str, params=None) -> httpx.Response:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}:
            raise ValueError("only HTTPS arXiv URLs are permitted")
        for attempt in range(3):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            response = self.client.get(url, params=params)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("request attempts exhausted")

    def discover(self, seeds_path: Path, target: int) -> list[Paper]:
        seeds = json.loads(seeds_path.read_text(encoding="utf-8"))["arxiv_ids"]
        papers = self._query({"id_list": ",".join(seeds), "max_results": len(seeds)}, "seed")
        by_id = {re.sub(r"v\d+$", "", p.paper_id): p for p in papers}
        queries = [
            'ti:"retrieval augmented" OR ti:"retrieval-augmented"',
            'ti:"language agents" OR ti:"tool learning" OR ti:"agentic"',
            'ti:"efficient transformer" OR ti:"linear attention" OR ti:"long context"',
        ]
        # Balance coverage instead of letting the first broad RAG query fill the corpus.
        per_topic = max(1, (target - len(by_id) + len(queries) - 1) // len(queries))
        excluded_domains = (
            "image caption",
            "audio caption",
            "time series",
            "time-series",
            "depression",
            "medicine",
            "clinical",
            "accountancy",
            "autonomous driving",
            "forest change",
            "video corpus",
        )
        groups = []
        for query in queries:
            found = self._query(
                {
                    "search_query": f"({query}) AND (cat:cs.CL OR cat:cs.LG)",
                    "max_results": 100,
                    "sortBy": "relevance",
                },
                "topic_candidate",
            )
            found = [
                p for p in found if not any(term in p.title.lower() for term in excluded_domains)
            ]
            groups.append(found)
            added = 0
            for paper in found:
                key = re.sub(r"v\d+$", "", paper.paper_id)
                if key in by_id:
                    continue
                by_id[key] = paper
                added += 1
                if added >= per_topic or len(by_id) >= target:
                    break
        for paper in (p for group in groups for p in group):
            if len(by_id) < target:
                by_id.setdefault(re.sub(r"v\d+$", "", paper.paper_id), paper)
        return list(by_id.values())[:target]

    def _query(self, params, selection):
        return parse_feed(self.get("https://export.arxiv.org/api/query", params).text, selection)

    def download(self, paper: Paper, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{paper.paper_id}.pdf"
        if path.exists() and path.read_bytes().startswith(b"%PDF-"):
            return path
        response = self.get(paper.pdf_url)
        if not response.content.startswith(b"%PDF-"):
            raise ValueError("response is not a PDF")
        temporary = path.with_suffix(".pdf.part")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        return path


def chunk_pages(paper: Paper, pages: list[str], config: ChunkConfig) -> list[Chunk]:
    tokenizer = tiktoken.get_encoding(config.tokenizer)
    chunks = []
    current_section = "unknown"
    for number, text in enumerate(pages, 1):
        spans, current_section = section_spans(text, current_section)
        for span_id, (section, body) in enumerate(spans):
            tokens = tokenizer.encode(body, disallowed_special=())
            for start in range(0, len(tokens), config.size - config.overlap):
                window = tokens[start : start + config.size]
                if not window or not tokenizer.decode(window).strip():
                    continue
                chunks.append(
                    Chunk(
                        paper_id=paper.paper_id,
                        title=paper.title,
                        authors=paper.authors,
                        year=paper.year,
                        page=number,
                        section=section,
                        section_source="heading_heuristic_v1"
                        if section != "unknown"
                        else "unknown",
                        chunk_id=f"{paper.paper_id}:p{number}:s{span_id}:t{start}",
                        source_url=paper.source_url,
                        text=tokenizer.decode(window),
                        token_count=len(window),
                    )
                )
                if start + config.size >= len(tokens):
                    break
    return chunks


def ingest(manifest: Path, raw: Path, processed: Path, config: ChunkConfig, limit=None):
    records, all_chunks = [], []
    papers = [Paper.model_validate(p) for p in read_jsonl(manifest)]
    for paper in papers[:limit]:
        path = raw / f"{paper.paper_id}.pdf"
        if not path.exists():
            records.append({"paper_id": paper.paper_id, "status": "missing_pdf"})
            continue
        try:
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            chunks = chunk_pages(paper, pages, config)
            if not chunks:
                raise ValueError("no extractable text; OCR/manual review required")
            all_chunks.extend(chunks)
            records.append(
                {
                    "paper_id": paper.paper_id,
                    "status": "parsed",
                    "pdf_sha256": sha256_file(path),
                    "pages": len(pages),
                    "empty_pages": sum(not p.strip() for p in pages),
                    "chunks": len(chunks),
                }
            )
            print(f"Parsed {paper.paper_id}: {len(chunks)} chunks", file=sys.stderr, flush=True)
        except (PdfReadError, OSError, ValueError) as error:
            records.append(
                {
                    "paper_id": paper.paper_id,
                    "status": "parse_failed",
                    "error_type": type(error).__name__,
                }
            )
    write_jsonl(processed / "chunks.jsonl", (c.model_dump() for c in all_chunks))
    report = {
        "manifest_sha256": sha256_file(manifest),
        "chunking": config.model_dump(),
        "section_parser": "heading_heuristic_v1",
        "unknown_section_chunks": sum(c.section == "unknown" for c in all_chunks),
        "parsed_papers": sum(r["status"] == "parsed" for r in records),
        "chunks": len(all_chunks),
        "papers": records,
    }
    write_json(processed / "ingestion.json", report)
    return report
