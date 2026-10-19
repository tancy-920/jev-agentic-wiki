"""Rebuild local search from published input text; never contact an LLM provider."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

from reproduce import ROOT, read, verify, write


def validate_corpus():
    corpus = ROOT / "data/corpus"
    sources = [json.loads(s) for s in (corpus / "sources.jsonl").read_text(encoding="utf8").splitlines()]
    texts = {}
    for source in sources:
        path = (corpus / source["normalized"].replace("\\", "/")).resolve()
        if not path.is_relative_to(corpus) or hashlib.sha256(path.read_bytes()).hexdigest() != source["normalized_sha256"]:
            raise ValueError("Normalized source text changed")
        texts[source["source_id"]] = path.read_text(encoding="utf8")
    chunks = [json.loads(s) for s in (corpus / "chunks.jsonl").read_text(encoding="utf8").splitlines()]
    for chunk in chunks:
        identity = chunk.get("source_id") or chunk.get("paper_id")
        if texts[identity][chunk["start_char"]:chunk["end_char"]] != chunk["text"]:
            raise ValueError("Chunk not equal to exact normalized source span")
    return dict(sources=len(sources), chunks=len(chunks), original_pdf_binaries_included=False,
                normalized_text_and_exact_chunk_spans_verified=True)


def run(output, device, warmups, repetitions):
    verify()
    proof = validate_corpus()
    if output.exists():
        raise ValueError("Fresh output directory required; do not overwrite prior timing runs")
    sys.path.insert(0, str(ROOT / "vendor/src"))
    from jev_rag.config import Config
    from jev_rag.knowledge import KnowledgeStore, WikiRetriever
    from jev_rag.llm_wiki import ArticleStore, WikiProductRouter
    from jev_rag.retrieval import HybridRetriever, build_index

    output.mkdir(parents=True)
    working = output / "corpus"
    shutil.copytree(ROOT / "data/corpus", working)
    config_data = read("data/wiki_setup.json")["config"]
    config_data["paths"] = dict(raw=str(working / "snapshots"), processed=str(working),
                                manifest=str(working / "sources.jsonl"))
    config_data["qdrant"]["path"] = str(output / "qdrant")
    config_data["embedding"]["device"] = device
    config_data["embedding"]["local_files_only"] = True
    config_data["retrieval"]["reranker_local_files_only"] = True
    config = Config.model_validate(config_data)
    metadata = build_index(config)  # locally generated index stays under ignored outputs/
    # PDF bytes are deliberately omitted; exact normalized spans were checked above.
    source = KnowledgeStore(working, verify=False)
    articles = ArticleStore(ROOT / "data/wiki")
    router = WikiProductRouter(articles, lambda: WikiRetriever(HybridRetriever(config), source))
    rows = []
    try:
        started = perf_counter()
        router.with_mode("source")
        cold_seconds = perf_counter() - started
        for case, plan in zip(read("data/wiki_cases.json"), read("data/shared_plans.json"), strict=True):
            cid = case["id"]
            queries = plan["resolution"]["plan"]["queries"]
            original = plan["resolution"]["retrieval_translation"]["english_query"]
            modes = ["source", "llm_wiki"] + (["llm_wiki_graph"] if cid in {"N05", "N06"} else [])
            samples, contexts = [], {}
            for repeat in range(warmups + repetitions):
                for offset in range(len(modes)):
                    mode = modes[(repeat + offset) % len(modes)]
                    selected = router.with_mode(mode)
                    if hasattr(selected, "with_term_resolution"):
                        selected = selected.with_term_resolution(plan["terminology"])
                    started = perf_counter()
                    hits = selected.search_plan(queries, original, top_k=20)
                    elapsed = perf_counter() - started
                    evidence, characters = [], 0
                    for hit in hits:
                        chunk = hit["chunk"]
                        if characters + len(chunk["text"]) <= 24000:
                            evidence.append(chunk)
                            characters += len(chunk["text"])
                    ids = [c["chunk_id"] for c in evidence]
                    if mode in contexts and contexts[mode]["ids"] != ids:
                        raise ValueError("Nondeterministic repeated retrieval")
                    saved = read(f"data/retrieval/{cid}.json")["contexts"][mode]["evidence"]
                    contexts[mode] = dict(ids=ids, characters=characters,
                        historical_evidence_ids_match=ids == [c["chunk_id"] for c in saved])
                    if repeat >= warmups:
                        samples.append(dict(mode=mode, repeat=repeat - warmups, seconds=elapsed))
            row = dict(id=cid, contexts=contexts, timing_rows=samples,
                median_seconds={m: median(s["seconds"] for s in samples if s["mode"] == m) for m in modes})
            rows.append(row)
            write(output / f"{cid}.json", row)
            print(cid, {m: contexts[m]["historical_evidence_ids_match"] for m in modes}, flush=True)
        write(output / "summary.json", dict(type="NEW_LOCAL_RETRIEVAL_RUN_NOT_ARCHIVED_TIMINGS", provider_calls=0,
            device=device, warmups=warmups, repetitions=repetitions, config=config_data,
            source_validation=proof, index_metadata=metadata, cold_loading_seconds=cold_seconds,
            hardware_dependent=True, archived_scores_not_assigned_to_new_evidence=True, rows=rows))
    finally:
        router.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.check_inputs:
        verify()
        print(json.dumps(validate_corpus(), indent=2))
        return
    if not args.output or not 0 <= args.warmups <= 10 or not 1 <= args.repetitions <= 100:
        parser.error("fresh --output, warmups0..10, repetitions1..100 required")
    run(args.output.resolve(), args.device, args.warmups, args.repetitions)


if __name__ == "__main__":
    main()
