"""Source-linked Wiki and navigation graph. No hosted calls or executable model output.

Wiki entries route to immutable source spans. Editorial descriptions and graph
links are navigation, never factual evidence or human-reviewed ground truth.
"""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_rag.io import read_jsonl, sha256_file
from jev_rag.schema import Chunk

KNOWLEDGE_MODES = ("source", "wiki", "wiki_graph")


def nonpeak_allowed(now=None, *, override_date=None):
    local = (now or datetime.now(timezone.utc)) + timedelta(hours=8)
    return (override_date is not None and type(override_date) is str and override_date == local.date().isoformat()) or local.weekday() >= 5 or not (9 <= local.hour < 12 or 14 <= local.hour < 18)


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,100}$")
    title: str
    kind: str
    origin: str
    version: str
    snapshot: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized: str
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser: str
    retrieved_at_utc: str
    authors: list[str] = Field(default_factory=list)
    year: int = 2026
    paper_id: str | None = None
    measurement_scope: str | None = None


class WikiReadingSection(BaseModel):
    """Editorial reading instructions with exact source pointers, never evidence prose."""

    model_config = ConfigDict(extra="forbid")
    heading: str = Field(min_length=1)
    guide: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    navigation_kind: Literal["section", "premise_bundle"] = "section"


class WikiCoverageNote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept: str = Field(min_length=1)
    limitation: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class WikiPage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: str = Field(pattern=r"^[a-z0-9-]{1,80}$")
    title: str
    aliases: list[str]
    guide: str
    source_ids: list[str]
    evidence_ids: list[str]
    related: list[str]
    review_status: str = "AI_EDITORIAL_DRAFT_NOT_GROUND_TRUTH"
    provenance: str = "source-excerpt-catalog-v1"
    navigation_kind: Literal["topic", "workflow"] = "topic"
    reading_sections: list[WikiReadingSection] = Field(default_factory=list)
    coverage_notes: list[WikiCoverageNote] = Field(default_factory=list)


class WikiLearningPath(BaseModel):
    """Editorial reading order, never prerequisite/causal evidence."""

    model_config = ConfigDict(extra="forbid")
    path_id: str = Field(pattern=r"^[a-z0-9-]{1,80}$")
    title: str = Field(min_length=1)
    guide: str = Field(min_length=1)
    page_ids: list[str] = Field(min_length=1)
    role: Literal["editorial_reading_order_not_evidence"] = "editorial_reading_order_not_evidence"


class WikiCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "transformer-mygpt-wiki-v1"
    sources_sha256: str
    chunks_sha256: str
    pages: list[WikiPage]
    edges: list[dict]
    generation_calls: int = 0
    factual_certification: bool = False
    learning_paths: list[WikiLearningPath] = Field(default_factory=list)

    @model_validator(mode="after")
    def graph_integrity(self):
        ids = {p.page_id for p in self.pages}
        if len(ids) != len(self.pages):
            raise ValueError("duplicate Wiki pages")
        for page in self.pages:
            if not set(page.related) <= ids or page.page_id in page.related:
                raise ValueError("invalid navigation target")
            if len(page.related) != len(set(page.related)):
                raise ValueError("duplicate navigation target")
        edge_pairs = []
        for edge in self.edges:
            if (
                edge.get("from") not in ids
                or edge.get("to") not in ids
                or edge.get("relation") != "RELATED_TOPIC"
                or edge.get("role") != "navigation_not_semantic_evidence"
            ):
                raise ValueError("only closed-set topic navigation edges allowed")
            if edge["from"] == edge["to"]:
                raise ValueError("self navigation edge")
            edge_pairs.append((edge["from"], edge["to"]))
        if len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("duplicate navigation edge")
        declared_pairs = {(p.page_id, target) for p in self.pages for target in p.related}
        if set(edge_pairs) != declared_pairs:
            raise ValueError("graph edges differ from declared related topics")
        if len({p.path_id for p in self.learning_paths}) != len(self.learning_paths):
            raise ValueError("duplicate learning path")
        for path in self.learning_paths:
            if len(path.page_ids) != len(set(path.page_ids)) or not set(path.page_ids) <= ids:
                raise ValueError("invalid learning path topic")
        return self


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("source snapshot escapes corpus")
    return path


class KnowledgeStore:
    def __init__(self, root: Path, *, verify=True):
        self.root = root
        source_rows = [KnowledgeSource.model_validate(v) for v in read_jsonl(root / "sources.jsonl")]
        chunk_rows = [Chunk.model_validate(v) for v in read_jsonl(root / "chunks.jsonl")]
        self.sources = {s.source_id: s for s in source_rows}
        self.chunks = {c.chunk_id: c for c in chunk_rows}
        if len(self.sources) != len(source_rows) or len(self.chunks) != len(chunk_rows):
            raise ValueError("duplicate source or chunk identity")
        self.catalog = WikiCatalog.model_validate_json((root / "wiki.json").read_text("utf-8"))
        if self.catalog.sources_sha256 != sha256_file(
            root / "sources.jsonl"
        ) or self.catalog.chunks_sha256 != sha256_file(root / "chunks.jsonl"):
            raise ValueError("stale Wiki catalog: rebuild before serving")
        self.pages = {p.page_id: p for p in self.catalog.pages}
        for page in self.pages.values():
            if not set(page.source_ids) <= self.sources.keys():
                raise ValueError("unknown Wiki source")
            if not set(page.evidence_ids) <= self.chunks.keys():
                raise ValueError("dangling Wiki source span")
            if any(self.chunks[i].identity not in page.source_ids for i in page.evidence_ids):
                raise ValueError("Wiki evidence does not belong to declared sources")
            for section in [*page.reading_sections, *page.coverage_notes]:
                if not set(section.evidence_ids) <= set(page.evidence_ids):
                    raise ValueError("Wiki reading/coverage reference outside declared page evidence")
                if len(section.evidence_ids) != len(set(section.evidence_ids)):
                    raise ValueError("duplicate Wiki reading/coverage references")
        if verify:
            self.verify_sources()

    @cached_property
    def term_catalog(self):
        """Build attested spelling suggestions only for callers that need them.

        Read-only browsing still verifies every source, but does not prepare
        model-planning vocabulary on every page interaction.
        """
        from jev_rag.terms import TermCatalog

        return TermCatalog(self.pages.values(), self.chunks.values())

    def verify_sources(self):
        texts = {}
        for source in self.sources.values():
            if sha256_file(inside(self.root, source.snapshot)) != source.snapshot_sha256:
                raise ValueError("raw source snapshot changed")
            path = inside(self.root, source.normalized)
            if sha256_file(path) != source.normalized_sha256:
                raise ValueError("normalized source snapshot changed")
            texts[source.source_id] = path.read_text("utf-8")
        for chunk in self.chunks.values():
            source = self.sources.get(chunk.identity)
            if source is None or chunk.source_sha256 != source.normalized_sha256:
                raise ValueError("source identity/hash mismatch")
            if texts[chunk.identity][chunk.start_char : chunk.end_char] != chunk.text:
                raise ValueError("chunk differs from its exact source span")

    def search(self, query, limit=5, *, correct_aliases=True):
        if not 1 <= limit <= 8 or not query.strip() or len(query) > 6000:
            raise ValueError("bounded Wiki search required")
        if correct_aliases:
            query, _ = self.resolve_aliases(query)
        tokens = self._navigation_tokens(query)
        exact, partial = [], []
        lowered = query.lower()
        for page in self.pages.values():
            alias_matches = {a.lower() for a in page.aliases if self._alias_matches(a, lowered)}
            headings = self._navigation_tokens(" ".join([page.title, *page.aliases]))
            overlap = tokens & headings
            if not alias_matches and not overlap:
                continue
            # Navigation rank only; no calibrated probability or truth claim.
            # Duplicate/nested aliases are not independent votes. Exact topic
            # matches route directly; shared Chinese characters must not pull
            # unrelated pages into the direct-topic anchors.
            score = (max(map(len, alias_matches), default=0),
                     len(overlap) / max(1, len(headings)))
            (exact if alias_matches else partial).append((score, page.page_id))
        ranked = exact or partial
        ranked.sort(key=lambda pair: (-pair[0][0], -pair[0][1], pair[1]))
        return [self.pages[i] for _, i in ranked[:limit]]

    def resolve_aliases(self, query):
        """Suggest only unique adjacent-letter swaps of long known aliases.

        Used for local search, never a factual label or an altered user question.
        Short acronyms, numbers, unknown words and ambiguous matches stay intact.
        """
        aliases = {a.lower(): a for p in self.pages.values() for a in p.aliases
                   if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{4,}", a)}
        corrections = []

        def replace(match):
            word = match.group()
            lower = word.lower()
            if lower in aliases:
                return word
            matches = set()
            for index in range(len(lower) - 1):
                swapped = lower[:index] + lower[index + 1] + lower[index] + lower[index + 2:]
                if swapped != lower and swapped in aliases:
                    matches.add(swapped)
            if len(matches) != 1:
                return word
            alias = aliases[matches.pop()]
            item = {"original": word, "source_alias": alias, "role": "retrieval_suggestion_not_ground_truth"}
            if item not in corrections:
                corrections.append(item)
            return alias

        resolved = re.sub(r"(?<![a-zA-Z0-9_])[a-zA-Z][a-zA-Z0-9_]{4,}(?![a-zA-Z0-9_])", replace, query)
        return resolved, corrections

    @staticmethod
    def _navigation_tokens(text):
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
        return tokens

    @staticmethod
    def _alias_matches(alias, query):
        lowered = alias.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]", lowered):
            return lowered == query.strip()
        if re.fullmatch(r"[a-z0-9_]+", lowered):
            return re.search(rf"(?<![a-z0-9_]){re.escape(lowered)}(?![a-z0-9_])", query) is not None
        return lowered in query

    def neighbors(self, page_ids, *, max_hops=1, limit=5):
        if not 0 <= max_hops <= 2 or not 1 <= limit <= 20 or len(page_ids) > 5:
            raise ValueError("graph navigation budget exceeded")
        if not set(page_ids) <= self.pages.keys():
            raise ValueError("unknown Wiki seed")
        seen, frontier, result = set(page_ids), list(page_ids), []
        for _ in range(max_hops):
            following = []
            for identifier in frontier:
                for target in self.pages[identifier].related:
                    if target not in seen:
                        seen.add(target)
                        following.append(target)
                        result.append(self.pages[target])
                        if len(result) >= limit:
                            return result
            frontier = following
        return result

    def evidence(self, page_id, limit=8, *, query=None):
        if not 1 <= limit <= 20 or page_id not in self.pages:
            raise ValueError("bounded existing Wiki page required")
        return [self.chunks[i] for i in topic_entry_ids(self.pages[page_id], query, limit)]

    def binding(self):
        return {
            name: sha256_file(self.root / name)
            for name in ("sources.jsonl", "chunks.jsonl", "wiki.json", "index.json")
            if (self.root / name).exists()
        }


class WikiRetriever:
    """Fuse source hits and Wiki source pointers, then locally rerank original text.

    No editorial Wiki prose or graph edges are injected as answer evidence. The
    same candidate/text limits apply to source-only and Wiki modes.
    """

    def __init__(self, base, store, mode="wiki_graph"):
        if mode not in KNOWLEDGE_MODES:
            raise ValueError("unknown knowledge mode")
        self.base, self.store, self.mode = base, store, mode
        self.chunks, self.traces = base.chunks, base.traces
        self.term_resolution = None

    def with_mode(self, mode):
        return WikiRetriever(self.base, self.store, mode)

    def with_term_resolution(self, resolution):
        # Called with code-validated planner selections on a per-question wrapper.
        result = WikiRetriever(self.base, self.store, self.mode)
        result.term_resolution = resolution
        return result

    def close(self):
        self.base.close()

    def search(self, query):
        return self.search_plan([query], query, top_k=self.base.config.retrieval.top_k)

    def search_plan(self, queries, original_query, *, top_k=20):
        if not 1 <= len(queries) <= 2 or len(set(queries)) != len(queries):
            raise ValueError("one or two bounded queries required")
        if not 1 <= top_k <= min(20, self.base.config.retrieval.top_k):
            raise ValueError("evidence budget exceeded")
        from jev_rag.agent.intent import validate_search_query
        from jev_rag.retrieval import reciprocal_rank_fusion

        for query in [original_query, *queries]:
            validate_search_query(query)
        started = perf_counter()
        effective_queries = []
        effective_original, corrections = self.store.resolve_aliases(original_query)
        for query in queries:
            effective, changes = self.store.resolve_aliases(query)
            if effective not in effective_queries:
                effective_queries.append(effective)
            corrections.extend(c for c in changes if c not in corrections)
        if self.term_resolution is not None:
            resolution = self.term_resolution
            # No automatic fuzzy correction after a KEEP/ambiguous planner choice.
            effective_queries = list(dict.fromkeys([*queries, original_query, resolution["original_query"], resolution["expanded_query"]]))[:4]
            effective_original = original_query
            corrections = resolution["corrections"]
        rankings, by_id = [], {}
        for query in effective_queries:
            hits = self.base._search_candidates(query, top_k=60)
            by_id.update({h["chunk"]["chunk_id"]: h for h in hits})
            rankings.append([h["chunk"]["chunk_id"] for h in hits])
        seeds = (
            self.store.search(" ".join([original_query, *effective_queries]), limit=5,
                              correct_aliases=self.term_resolution is None)
            if self.mode != "source"
            else []
        )
        neighbors = (
            self.store.neighbors([p.page_id for p in seeds], limit=3)
            if self.mode == "wiki_graph"
            else []
        )
        pointers = []
        # Direct Wiki topics outrank graph neighbors; at most 3 spans/topic.
        entry_ids_by_page = {}
        seed_ids = {p.page_id for p in seeds}
        for page in [*seeds, *neighbors]:
            entries = self.store.evidence(page.page_id, limit=3,
                query=original_query if page.page_id in seed_ids else None)
            entry_ids_by_page[page.page_id] = [c.chunk_id for c in entries]
            for chunk in entries:
                if chunk.chunk_id not in pointers:
                    pointers.append(chunk.chunk_id)
                    by_id.setdefault(
                        chunk.chunk_id, {"chunk": chunk.model_dump(), "reranker_score": None}
                    )
        if pointers:
            rankings.append(pointers)
        fused = reciprocal_rank_fusion(rankings, self.base.config.retrieval.rrf_k)
        # Rerank the bounded union (up to60), not only the first20 before scoring.
        candidates = [{**by_id[key], "knowledge_rrf_score": score} for key, score in fused[:60]]
        ranked = self.base._rerank(effective_original, [dict(c) for c in candidates])
        original_ranked_ids = [h["chunk"]["chunk_id"] for h in ranked]
        context_rankings = []
        if re.search(r"[\u4e00-\u9fff]", original_query):
            supports = getattr(getattr(self.base, "reranker", None), "supports_query", None)
            for query in effective_queries:
                if query.isascii() and query != original_query and (supports is None or supports(query)):
                    context_hits = self.base._rerank(query, [dict(c) for c in candidates])
                    context_rankings.append([h["chunk"]["chunk_id"] for h in context_hits])
        if context_rankings:
            # English retrieval context and the authoritative original question
            # are independent rank signals; never add uncalibrated raw scores.
            # Two planned queries jointly contribute one context ranking.
            context_ids = [i for i, _ in reciprocal_rank_fusion(context_rankings, self.base.config.retrieval.rrf_k)]
            final_ranks = reciprocal_rank_fusion([original_ranked_ids, context_ids], self.base.config.retrieval.rrf_k)
            ranked_by_id = {h["chunk"]["chunk_id"]: h for h in ranked}
            ranked = [{**ranked_by_id[i], "question_context_rrf_score": score} for i, score in final_ranks]
        # A multi-topic question must not lose its third/fourth explicitly named
        # topic merely because two general overviews won the navigation rank.
        # Reserve bounded original entries per exact original-query topic (max5).
        # Planner-only topics and graph neighbors never force additional slots.
        # A directly matched workflow is a multi-stage reading route. Preserve
        # its bounded three entry spans rather than only its first stage. This
        # never admits editorial prose or graph neighbors as answer evidence.
        workflow = next((p for p in seeds[:2] if p.navigation_kind == "workflow"), None)
        explicit_topics = [p for p in seeds if any(
            self.store._alias_matches(alias, original_query.lower()) for alias in p.aliases)]
        # Entries supplement coverage; navigation metadata must not displace
        # the reranker's head with a lower-ranked conditional implementation.
        entries = (workflow.evidence_ids[:3] if workflow else
                   [entry_ids_by_page[p.page_id][index] for index in range(3 if explicit_topics else 1)
                    for p in (explicit_topics or seeds[:2]) if len(entry_ids_by_page[p.page_id]) > index])
        anchors = list(dict.fromkeys(entries))[:min(3 if workflow else 12 if explicit_topics else 2, top_k)]
        ranked_by_id = {h['chunk']['chunk_id']: h for h in ranked}
        anchor_hits = [ranked_by_id.get(i, by_id[i]) for i in anchors]
        reranked, navigation_selection = merge_navigation_anchors(ranked, anchor_hits, top_k)
        self.traces.append(
            {
                "stage": "wiki_source_navigation",
                "mode": self.mode,
                "rerank_original_query": original_query,
                "effective_search_queries": effective_queries,
                "effective_rerank_query": effective_original,
                "term_corrections": corrections,
                "query_terminology": self.term_resolution,
                "rerank_search_context_used": bool(context_rankings),
                "rerank_context_rankings": context_rankings,
                "authoritative_original_reranker_ids": original_ranked_ids,
                "wiki_pages": [p.page_id for p in seeds],
                "graph_neighbors": [p.page_id for p in neighbors],
                "source_pointer_ids": pointers,
                "final_ids": [h["chunk"]["chunk_id"] for h in reranked],
                "overview_anchor_ids": anchors,
                "overview_anchor_policy": "protected-reranker-head8;bounded-tail-topic-coverage-v2;no-forced-neighbors",
                "navigation_selection": navigation_selection,
                "topic_entry_ids": entry_ids_by_page,
                "original_reranker_ids": [h["chunk"]["chunk_id"] for h in ranked],
                "latency_ms": (perf_counter() - started) * 1000,
                "editorial_content_used_as_evidence": False,
                "new_provider_calls": 0,
            }
        )
        return reranked


def merge_navigation_anchors(ranked, anchors, top_k):
    """Protect local ranking before filling bounded coverage slots with originals.

    At the normal20 budget, eight reranked originals retain their exact order.
    A smaller budget protects at least one result. Wiki entries can fill only
    remaining slots; neither ordering nor inclusion certifies semantic support.
    """
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("bounded navigation top_k required")
    protected = ranked[:min(8, max(1, top_k // 2))]
    selected, seen = [], set()
    for hit in [*protected, *anchors, *ranked]:
        identifier = hit['chunk']['chunk_id']
        if identifier not in seen and len(selected) < top_k:
            selected.append(hit)
            seen.add(identifier)
    protected_ids = [h['chunk']['chunk_id'] for h in protected]
    anchor_ids = [h['chunk']['chunk_id'] for h in anchors]
    return selected, dict(
        protected_reranker_ids=protected_ids,
        admitted_anchor_ids=[i for i in anchor_ids if i in seen and i not in protected_ids],
        omitted_anchor_ids=[i for i in anchor_ids if i not in seen],
        displaced_reranker_ids=[h['chunk']['chunk_id'] for h in ranked[:top_k]
                               if h['chunk']['chunk_id'] not in seen],
        semantic_support_certified=False, additional_provider_calls=0)


def topic_entry_ids(page, query, limit):
    """Navigate to existing sections named by exact technical terms in a query.

    Headings are navigation metadata only. No editorial text enters evidence,
    no new source is admitted, and absence of a heading match keeps old order.
    English technical terms use ASCII boundaries even beside Chinese text.
    """
    if not query:
        return page.evidence_ids[:limit]
    terms = set(re.findall(r"[a-z](?:[a-z0-9_-]{2,}|[0-9])", query.lower()))
    matches = []
    for index, section in enumerate(page.reading_sections):
        score = sum(bool(re.search(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])",
                                  section.heading.lower())) for term in terms)
        if score:
            matches.append((-score, index, section))
    if not matches:
        return page.evidence_ids[:limit]
    sections = [section for _, _, section in sorted(matches, key=lambda row: row[:2])]
    # A declared bundle keeps a bounded operation/definition/premise group
    # together. It changes navigation only, never source support certification.
    if getattr(sections[0], "navigation_kind", "section") == "premise_bundle":
        ordered = [*sections[0].evidence_ids[:3], *page.evidence_ids]
        return list(dict.fromkeys(ordered))[:limit]
    # One entry per named section before its deeper spans; preserve an overview
    # when there is room. Every selected ID already belongs to this Wiki topic.
    ordered = [s.evidence_ids[0] for s in sections if s.evidence_ids]
    ordered += page.evidence_ids[:1]
    ordered += [i for s in sections for i in s.evidence_ids]
    ordered += page.evidence_ids
    return list(dict.fromkeys(ordered))[:limit]


def snapshot_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def product_store_path(workspace):
    """Use the configured versioned product corpus for UI and fingerprints."""
    from jev_rag.config import load_config

    path = workspace / "configs/transformer.yaml"
    relative = load_config(path).paths.processed if path.is_file() else Path("data/processed/transformer_wiki")
    return inside(workspace, relative)


def load_product_config(workspace):
    """Explicit isolated product config; historical studies keep base.yaml."""
    from jev_rag.config import load_config

    path = workspace / "configs/transformer.yaml"
    if path.exists():
        config = load_config(path)
        KnowledgeStore(workspace / config.paths.processed)
        return config
    return load_config(workspace / "configs/base.yaml")
