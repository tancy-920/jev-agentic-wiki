import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .agent.intent import validate_search_query
from .config import Config
from .io import read_jsonl, sha256_file, write_json
from .schema import Chunk


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, identifier in enumerate(ranking, 1):
            if identifier not in seen:
                scores[identifier] += 1 / (k + rank)
                seen.add(identifier)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def merge_search_hits(rankings, *, top_k=20, rrf_k=60):
    """Bounded multi-query fusion; model output cannot supply executable filters."""
    by_id, ids = {}, []
    for hits in rankings:
        ranking = []
        for hit in hits:
            chunk = Chunk.model_validate(hit["chunk"]).model_dump()
            key = chunk["chunk_id"]
            if key in by_id and by_id[key]["chunk"] != chunk:
                raise ValueError("conflicting payloads for the same chunk ID")
            by_id[key] = {**hit, "chunk": chunk}
            ranking.append(key)
        ids.append(ranking)
    return [
        {**by_id[key], "plan_rrf_score": score}
        for key, score in reciprocal_rank_fusion(ids, rrf_k)[:top_k]
    ]


class LocalEmbedder:
    """Token-window pooling avoids the model's silent truncation of 600-token chunks."""

    def __init__(self, config):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(
            config.model,
            device=config.device,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        self.dimension = self.model.get_embedding_dimension()
        self.revision = config.revision or getattr(
            self.model[0].auto_model.config, "_commit_hash", None
        )

    def encode(self, texts: list[str], query=False) -> list[list[float]]:
        if not texts:
            return []
        tokenizer = self.model.tokenizer
        size = min(
            self.config.window_tokens,
            self.model.max_seq_length - tokenizer.num_special_tokens_to_add(),
        )
        if size <= self.config.window_overlap:
            raise ValueError("embedding model token budget too small for configured overlap")
        windows, owners = [], []
        for owner, text in enumerate(texts):
            if query:
                text = self.config.query_prefix + text
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if not tokens:
                windows.append("")
                owners.append(owner)
            for start in range(0, len(tokens), size - self.config.window_overlap):
                windows.append(tokenizer.decode(tokens[start : start + size]))
                owners.append(owner)
                if start + size >= len(tokens):
                    break
        vectors = self.model.encode(
            windows,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        pooled = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for owner, vector in zip(owners, vectors, strict=True):
            pooled[owner] += vector
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).tolist()


def connect(config: Config):
    if config.qdrant.mode == "local":
        return QdrantClient(path=str(config.qdrant.path))
    if config.qdrant.mode == "server":
        return QdrantClient(url=config.qdrant.url)
    raise ValueError("qdrant mode must be local or server")


def build_index(config: Config, embedder=None):
    path = config.paths.processed / "chunks.jsonl"
    chunks = [Chunk.model_validate(c) for c in read_jsonl(path)]
    if not chunks:
        raise ValueError("cannot index an empty corpus")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk IDs")
    embedder = embedder or LocalEmbedder(config.embedding)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "corpus": sha256_file(path),
                "embedding": config.embedding.model_dump(),
                "resolved_revision": getattr(embedder, "revision", None),
                "kind": getattr(embedder, "kind", "REAL"),
                "dimension": embedder.dimension,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    collection = f"papers_{fingerprint[:20]}"
    client = connect(config)
    try:
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=embedder.dimension, distance=models.Distance.COSINE
                ),
            )
        for start in range(0, len(chunks), config.embedding.batch_size):
            batch = chunks[start : start + config.embedding.batch_size]
            vectors = embedder.encode([c.text for c in batch])
            client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, c.chunk_id)),
                        vector=vector,
                        payload=c.model_dump(),
                    )
                    for c, vector in zip(batch, vectors, strict=True)
                ],
                wait=True,
            )
            if start % (config.embedding.batch_size * 25) == 0:
                print(
                    f"Indexed {min(start + len(batch), len(chunks))}/{len(chunks)} chunks",
                    file=sys.stderr,
                    flush=True,
                )
        count = client.count(collection, exact=True).count
        if count != len(chunks):
            raise RuntimeError("index count does not match corpus")
    finally:
        client.close()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(mode="json"),
        "versions": {name: version(name) for name in ("qdrant-client", "rank-bm25", "numpy")},
        "collection": collection,
        "fingerprint": fingerprint,
        "chunks_sha256": sha256_file(path),
        "chunks": count,
        "indexed_papers": len({c.paper_id for c in chunks if c.paper_id is not None}),
        "indexed_sources": len({c.identity for c in chunks}),
        "embedding": config.embedding.model_dump(),
        "resolved_revision": getattr(embedder, "revision", None),
        "kind": getattr(embedder, "kind", "REAL"),
        "dimension": embedder.dimension,
    }
    write_json(config.paths.processed / "index.json", metadata)
    return metadata


class HybridRetriever:
    def __init__(self, config: Config, embedder=None):
        self.config = config
        self.traces = []
        path = config.paths.processed / "chunks.jsonl"
        metadata = json.loads((config.paths.processed / "index.json").read_text(encoding="utf-8"))
        if metadata["chunks_sha256"] != sha256_file(path):
            raise ValueError("corpus changed: rebuild the index before querying")
        if metadata["embedding"] != config.embedding.model_dump():
            raise ValueError("embedding configuration changed: rebuild the index")
        self.embedder = embedder or LocalEmbedder(config.embedding)
        if metadata["kind"] != getattr(self.embedder, "kind", "REAL"):
            raise ValueError("MOCK and REAL embeddings cannot be mixed")
        if metadata["dimension"] != self.embedder.dimension:
            raise ValueError("embedding dimensions do not match index")
        if metadata["resolved_revision"] != getattr(self.embedder, "revision", None):
            raise ValueError("embedding revision changed: rebuild the index")
        self.chunks = [Chunk.model_validate(c) for c in read_jsonl(path)]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Okapi([lexical_tokens(c.text) or ["__empty__"] for c in self.chunks])
        self.collection = metadata["collection"]
        self.reranker = None
        if config.retrieval.reranker_enabled:
            from .reranking import LocalReranker

            self.reranker = LocalReranker(config.retrieval, config.embedding.device)
        self.client = connect(config)

    def close(self):
        self.client.close()

    def search(self, query: str) -> list[dict]:
        if not query.strip():
            return []
        return self._rerank(query, self._search_candidates(query))

    def search_plan(self, queries: list[str], original_query: str, *, top_k=20) -> list[dict]:
        if not 1 <= len(queries) <= 2 or len(set(queries)) != len(queries):
            raise ValueError("plan requires one or two distinct queries")
        if not 1 <= top_k <= min(20, self.config.retrieval.top_k):
            raise ValueError("invalid global passage budget")
        validate_search_query(original_query)
        for query in queries:
            validate_search_query(query)
        results = merge_search_hits(
            [self._search_candidates(query) for query in queries],
            top_k=top_k,
            rrf_k=self.config.retrieval.rrf_k,
        )
        # Score the final bounded union against the authoritative question, once.
        return self._rerank(original_query, results)

    def _search_candidates(self, query: str, *, top_k=None) -> list[dict]:
        started = perf_counter()
        settings = self.config.retrieval
        dense = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.encode([query], query=True)[0],
            limit=settings.dense_k,
            with_payload=True,
        ).points
        dense_finished = perf_counter()
        dense_ids = [point.payload["chunk_id"] for point in dense]
        tokens = lexical_tokens(query)
        sparse_scores = self.bm25.get_scores(tokens)
        # Require actual lexical overlap; avoid arbitrary BM25 ties on OOK queries.
        sparse = [
            i
            for i in range(len(self.chunks))
            if set(tokens).intersection(lexical_tokens(self.chunks[i].text))
        ]
        sparse.sort(key=lambda i: (-float(sparse_scores[i]), self.chunks[i].chunk_id))
        sparse_ids = [self.chunks[i].chunk_id for i in sparse[: settings.sparse_k]]
        sparse_finished = perf_counter()
        limit = settings.top_k if top_k is None else top_k
        if not 1 <= limit <= 60:
            raise ValueError("candidate union must be bounded to60")
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], settings.rrf_k)[:limit]
        results = [
            {
                "chunk": self.by_id[key].model_dump(),
                "rrf_score": score,
                "dense_rank": dense_ids.index(key) + 1 if key in dense_ids else None,
                "sparse_rank": sparse_ids.index(key) + 1 if key in sparse_ids else None,
                "reranker_score": None,
            }
            for key, score in fused
        ]
        self.traces.append(
            {
                "stage": "hybrid_candidates",
                "query": query,
                "dense_ids": dense_ids,
                "sparse_ids": sparse_ids,
                "fused": [
                    {k: v for k, v in item.items() if k != "chunk"}
                    | {"chunk_id": item["chunk"]["chunk_id"]}
                    for item in results
                ],
                "dense_embedding_search_ms": (dense_finished - started) * 1000,
                "bm25_ms": (sparse_finished - dense_finished) * 1000,
                "fusion_ms": (perf_counter() - sparse_finished) * 1000,
            }
        )
        return results

    def _rerank(self, query, results):
        started = perf_counter()
        before = [item["chunk"]["chunk_id"] for item in results]
        if self.reranker and results:
            scores = self.reranker.score(query, [item["chunk"]["text"] for item in results])
            for item, score in zip(results, scores, strict=True):
                item["reranker_score"] = float(score)
            results.sort(key=lambda item: -item["reranker_score"])
        self.traces.append(
            {
                "stage": "local_rerank",
                "query": query,
                "input_ids": before,
                "ranked": [
                    {
                        "chunk_id": item["chunk"]["chunk_id"],
                        "reranker_score": item.get("reranker_score"),
                    }
                    for item in results
                ],
                "latency_ms": (perf_counter() - started) * 1000,
            }
        )
        return results
