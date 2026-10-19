import argparse
import json
from pathlib import Path

import httpx

from .config import load_config
from .ingestion import ArxivSource, ingest
from .io import read_jsonl, write_json, write_jsonl
from .schema import Paper


def main():
    parser = argparse.ArgumentParser(description="Local paper knowledge base (no Jev/LLM calls)")
    parser.add_argument("--config", default="configs/base.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--target", type=int, default=120)
    discover.add_argument("--seeds", default="data/seed_papers.json")
    discover.add_argument("--refresh", action="store_true")
    download = commands.add_parser("download")
    download.add_argument("--limit", type=int)
    parse = commands.add_parser("ingest")
    parse.add_argument("--limit", type=int)
    commands.add_parser("index")
    query = commands.add_parser("search")
    query.add_argument("question")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "discover":
        if config.paths.manifest.exists() and not args.refresh:
            parser.error("manifest already exists; use --refresh intentionally to replace it")
        source = ArxivSource()
        try:
            papers = source.discover(Path(args.seeds), args.target)
            write_jsonl(config.paths.manifest, (p.model_dump() for p in papers))
            print(json.dumps({"papers": len(papers), "review_status": "pending"}))
        finally:
            source.close()
    elif args.command == "download":
        papers = [Paper.model_validate(p) for p in read_jsonl(config.paths.manifest)]
        source, records = ArxivSource(), []
        try:
            for paper in papers[: args.limit]:
                try:
                    path = source.download(paper, config.paths.raw)
                    record = {
                        "paper_id": paper.paper_id,
                        "status": "downloaded",
                        "bytes": path.stat().st_size,
                    }
                except (httpx.HTTPError, OSError, ValueError) as error:
                    record = {
                        "paper_id": paper.paper_id,
                        "status": "failed",
                        "error_type": type(error).__name__,
                    }
                    if isinstance(error, httpx.HTTPStatusError):
                        record["http_status"] = error.response.status_code
                records.append(record)
                write_json(config.paths.raw / "downloads.json", records)
                print(json.dumps(record), flush=True)
        finally:
            source.close()
    elif args.command == "ingest":
        report = ingest(
            config.paths.manifest,
            config.paths.raw,
            config.paths.processed,
            config.chunking,
            args.limit,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "papers"}))
    elif args.command == "index":
        from .retrieval import build_index

        print(json.dumps(build_index(config)))
    elif args.command == "search":
        from .retrieval import HybridRetriever

        retriever = HybridRetriever(config)
        try:
            print(json.dumps(retriever.search(args.question), ensure_ascii=False, indent=2))
        finally:
            retriever.close()


if __name__ == "__main__":
    main()
