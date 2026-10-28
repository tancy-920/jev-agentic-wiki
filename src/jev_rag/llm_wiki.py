"""Whole Markdown knowledge units. No dense chunk search or cross encoder.

Historical WikiRetriever is intentionally unchanged. Article faithfulness does
not certify the underlying originals or human-reviewed ground truth.
"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from jev_rag.io import sha256_file
from jev_rag.schema import Chunk

MODES = ("llm_wiki", "llm_wiki_graph")


class ArticleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_id: str = Field(pattern=r"^[a-z0-9-]{1,80}$")
    title: str = Field(min_length=1)
    aliases: list[str]
    markdown_file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_chunk_ids: list[str] = Field(min_length=1)
    related: list[str]
    review_status: str = "AI_SOURCE_GROUNDED_DRAFT"


def tokens(text):
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return words


def article_content(text):
    """Provenance and link lists are metadata, never topical matching evidence."""
    return re.split(r"^## (?:来源追溯|关联知识)\s*$", text, maxsplit=1, flags=re.M)[0]


def facet_tokens(text):
    """Longer Chinese spans suppress accidental two-character graph matches."""
    result = set(re.findall(r"[a-z0-9_]{3,}", text.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(phrase[i:i + 3] for i in range(len(phrase) - 2))
    return result


class ArticleStore:
    def __init__(self, root, *, source_store=None):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text("utf8"))
        if manifest.get("type") != "LLM_WIKI_COMPLETE_ARTICLES_V1":
            raise ValueError("whole-article Wiki manifest required")
        rows = [ArticleRecord.model_validate(r) for r in manifest["articles"]]
        self.pages = {r.article_id: r for r in rows}
        if len(self.pages) != len(rows):
            raise ValueError("duplicate article")
        self.texts, self.chunks = {}, {}
        self.manifest = manifest
        if source_store is not None and manifest["source_binding"] != source_store.binding():
            raise ValueError("article source catalog changed")
        paths = set()
        for row in rows:
            path = (self.root / row.markdown_file).resolve()
            if not path.is_relative_to(self.root) or path.suffix != ".md":
                raise ValueError("article path escapes local Wiki")
            if path in paths:
                raise ValueError("duplicate article file")
            paths.add(path)
            if sha256_file(path) != row.sha256:
                raise ValueError("article changed: rebuild manifest")
            text = path.read_text("utf8")
            if not text.startswith("# ") or len(text) > 18000:
                raise ValueError("complete bounded Markdown article required")
            if not set(row.related) <= self.pages.keys() or row.article_id in row.related:
                raise ValueError("invalid article relation")
            if source_store is not None:
                if not set(row.source_chunk_ids) <= source_store.chunks.keys():
                    raise ValueError("article provenance missing")
            self.texts[row.article_id] = text
            identifier = "wiki-article:" + row.article_id
            self.chunks[identifier] = Chunk(
                source_id=identifier, source_kind="wiki_article", source_version=row.sha256,
                source_sha256=row.sha256, title=row.title + "（AI整理词条）",
                authors=[], year=2026, section="完整知识词条；非原论文逐字原文",
                chunk_id=identifier, source_url="", locator=row.markdown_file,
                text=text, token_count=len(text.encode("utf8")),
            )
        self.term_catalog = None
        self.content_tokens = {i: tokens(article_content(t)) for i, t in self.texts.items()}
        self.title_tokens = {r.article_id: tokens(" ".join([r.title, *r.aliases])) for r in rows}
        self.content_facets = {i: facet_tokens(article_content(t)) for i, t in self.texts.items()}
        frequencies = Counter(t for terms in self.content_tokens.values() for t in terms)
        self.idf = {t: math.log(1 + len(rows) / (1 + count)) for t, count in frequencies.items()}

    def topical_score(self, identifier, query_tokens):
        """Small in-memory article catalog score, not chunk recall or neural reranking."""
        title = query_tokens & self.title_tokens[identifier]
        body = query_tokens & self.content_tokens[identifier]
        return sum(3 * self.idf.get(t, 1) for t in title) + sum(self.idf.get(t, 1) for t in body)

    def binding(self):
        # Recheck bodies as well as the manifest for live source drift guards.
        return {"article_manifest": sha256_file(self.root / "manifest.json"),
                "articles": {r.article_id: sha256_file(self.root / r.markdown_file)
                             for r in self.pages.values()},
                "source_binding": self.manifest["source_binding"]}

    def search(self, query, limit=4):
        if (not isinstance(query, str) or not query.strip() or len(query) > 6000
            or type(limit) is not int or not 1 <= limit <= 8):
            raise ValueError("bounded article query required")
        query_tokens, ranked = tokens(query), []
        for row in self.pages.values():
            exact = max((len(a) for a in [row.title, *row.aliases]
                         if alias_match(a, query)), default=0)
            title_overlap = len(query_tokens & tokens(" ".join([row.title, *row.aliases])))
            body_overlap = self.topical_score(row.article_id, query_tokens)
            # Body-only isolated common words must not force an arbitrary answer.
            if not exact and not title_overlap:
                continue
            # Alias length is not importance: a broad long name must not outrank
            # all specific requested concepts. Score only article content/title,
            # excluding provenance and links; keep exact matches as eligibility.
            ranked.append(((bool(exact), body_overlap, title_overlap), row.article_id))
        # Keep partial title matches only when they cover a requested concept
        # absent from exact hits. A shared word like KV must not add unrelated
        # neighbors, while an exact hit for topic A cannot erase topic B.
        covered = set().union(*(query_tokens & self.title_tokens[i]
                                for score, i in ranked if score[0]))
        ranked = [(score, i) for score, i in ranked
                  if score[0] or (query_tokens & self.title_tokens[i]) - covered]
        ranked.sort(key=lambda item: (tuple(-n for n in item[0]), item[1]))
        return [self.pages[i] for _, i in ranked[:limit]]


def alias_match(alias, query):
    # Apply identifier boundaries to phrases/hyphenated aliases as well as words.
    left = r"(?<![a-z0-9_])" if re.match(r"[a-zA-Z0-9_]", alias) else ""
    right = r"(?![a-z0-9_])" if re.search(r"[a-zA-Z0-9_]$", alias) else ""
    return bool(alias) and re.search(left + re.escape(alias.lower()) + right, query.lower()) is not None


class ArticleRetriever:
    whole_articles = True

    def __init__(self, store, *, mode="llm_wiki_graph", traces=None, max_chars=24000):
        if mode not in MODES:
            raise ValueError("unknown article mode")
        if type(max_chars) is not int or not 1 <= max_chars <= 24000:
            raise ValueError("bounded whole-article context required")
        self.store, self.mode = store, mode
        self.chunks = list(store.chunks.values())
        self.traces = traces if traces is not None else []
        self.max_chars = max_chars

    def search(self, query):
        return self.search_plan([query], query, top_k=6)

    def merge_context(self, previous, fresh, *, top_k):
        """Keep original scope during repair without slicing or growing budgets.

        Alternate earlier context and new hits, so a broad supplementary query
        cannot replace all of the evidence for the original question.
        """
        selected, omitted, seen, size = [], [], set(), 0
        for index in range(max(len(previous), len(fresh))):
            for group in (previous, fresh):
                if index >= len(group):
                    continue
                chunk = group[index]
                identifier = chunk["chunk_id"]
                if identifier in seen:
                    continue
                seen.add(identifier)
                if len(selected) >= min(top_k, 6) or size + len(chunk["text"]) > self.max_chars:
                    omitted.append(identifier)
                    continue
                selected.append(chunk)
                size += len(chunk["text"])
        return selected, dict(policy="whole_article_scope_preserving_repair_v1",
                              selected_ids=[c["chunk_id"] for c in selected],
                              omitted_whole=omitted, article_characters=size,
                              max_characters=self.max_chars, new_provider_calls=0)

    def search_plan(self, queries, original_query, *, top_k=6):
        if (not isinstance(queries, (list, tuple)) or not 1 <= len(queries) <= 2
            or type(top_k) is not int or not 1 <= top_k <= 20
            or not isinstance(original_query, str) or not original_query.strip()
            or any(not isinstance(q, str) or not q.strip() for q in queries)):
            raise ValueError("bounded article plan required")
        started = perf_counter()
        # Original question is always a signal; queries never authorize tools.
        query = " ".join([original_query, *queries])
        seeds = self.store.search(query, limit=min(4, top_k))
        seed_ids = [r.article_id for r in seeds]
        neighbors = []
        candidates = []
        if self.mode == "llm_wiki_graph":
            # Rank ALL authorized one-hop candidates by the user's information
            # needs; graph file order no longer consumes the two-neighbor budget.
            parents = {}
            for seed in seeds:
                for neighbor in seed.related:
                    if neighbor not in seed_ids:
                        parents.setdefault(neighbor, []).append(seed.article_id)
            original_terms, query_terms = tokens(original_query), tokens(query)
            seed_names = set().union(*(facet_tokens(" ".join([s.title, *s.aliases])) for s in seeds))
            requested_facets = facet_tokens(query) - seed_names
            for neighbor, anchors in parents.items():
                score = (2 * self.store.topical_score(neighbor, original_terms)
                         + self.store.topical_score(neighbor, query_terms))
                matches = sorted(requested_facets & self.store.content_facets[neighbor])
                candidates.append(dict(article_id=neighbor, score=score, parents=anchors,
                                       matched_facets=matches,
                                       eligible=bool(matches) or not requested_facets))
            candidates.sort(key=lambda c: (-c["score"], c["article_id"]))
            # Lexically implicit topics can still be useful authorized links.
            # Such zero-score links are last, with an explicit traceable fallback.
            for candidate in candidates:
                candidate["reason"] = ("no_new_query_facet" if not candidate["eligible"] else
                                       "query_match" if candidate["score"] else "authorized_related_fallback")
            neighbors = [c["article_id"] for c in candidates if c["eligible"]][:2]
        selected, omitted, size = [], [], 0
        for identifier in [*seed_ids, *neighbors[:2]]:
            text = self.store.texts[identifier]
            if len(selected) >= min(top_k, 6) or size + len(text) > self.max_chars:
                omitted.append(identifier)
                continue
            size += len(text)
            selected.append(identifier)
        self.traces.append(dict(
            stage="whole_article_navigation", mode=self.mode, seeds=seed_ids,
            graph_neighbors=neighbors[:2], selected=selected, omitted_whole=omitted,
            graph_selection="query_weighted_whole_article_one_hop_v2",
            graph_candidates=candidates,
            complete_articles=True, article_characters=size, chunk_recall_calls=0,
            dense_search_calls=0, reranker_calls=0, new_provider_calls=0,
            latency_ms=(perf_counter() - started) * 1000,
            verification_scope="article_faithfulness_not_original_source_certification",
        ))
        return [{"chunk": self.store.chunks["wiki-article:" + i].model_dump(),
                 "reranker_score": None} for i in selected]


class WikiProductRouter:
    """Lazy legacy baseline; startup/ordinary Wiki never load embedding models."""
    def __init__(self, articles, legacy_factory, *, mode="llm_wiki_graph", traces=None):
        self.articles, self.legacy_factory, self.mode = articles, legacy_factory, mode
        self.traces = traces if traces is not None else []
        self.store = articles
        self.chunks = list(articles.chunks.values())
        self._legacy = None

    def with_mode(self, mode):
        if mode in MODES:
            return ArticleRetriever(self.articles, mode=mode, traces=self.traces)
        if mode not in {"source", "wiki", "wiki_graph"}:
            raise ValueError("unknown knowledge mode")
        if self._legacy is None:
            self._legacy = self.legacy_factory()
        return self._legacy.with_mode(mode)

    def search(self, query):
        return self.with_mode(self.mode).search(query)

    def close(self):
        if self._legacy is not None:
            self._legacy.close()


def article_digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()
