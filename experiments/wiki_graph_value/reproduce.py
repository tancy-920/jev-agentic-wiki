"""Public experiment reproduction. Default execution is offline and standard-library only."""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def verify():
    manifest = read("MANIFEST.json")
    for rel, expected in manifest["files"].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact changed: {rel}")
    return len(manifest["files"])


def close(actual, expected):
    if not math.isclose(actual, expected, abs_tol=1e-12):
        raise ValueError(f"Reproduction mismatch: {actual} != {expected}")


def wiki_metrics():
    """Aggregate archived measurements/judgments; never invent fresh judgments."""
    review = read("data/wiki_review.json")
    records = read("data/wiki_records.json")
    by_key = {(r["id"], r["mode"]): r for r in records}
    physical = {c["request_sha256"]: c for r in records for c in r["calls"] if c["mode"] == "LIVE"}
    rows = []
    for metric in review["metrics"]:
        cid = metric["id"]
        row = dict(id=cid, input_tokens=metric["input_tokens"],
                   output_tokens=metric["output_tokens"], coverage={}, supported_items={},
                   total_items=None, retrieval_median_ms={})
        timing = read(f"data/retrieval/{cid}.json")
        for mode in ("source", "llm_wiki"):
            judged = next(r for r in review["rows"] if r["id"] == cid and r["mode"] == mode)
            record = by_key[cid, mode]
            answer = record["answer"]["answer"]
            observed = physical[record["calls"][0]["request_sha256"]]
            if observed["input_tokens"] != row["input_tokens"][mode] or observed["output_tokens"] != row["output_tokens"][mode]:
                raise ValueError("Summary token usage differs from physical request metadata")
            for item in judged["items"]:
                if item["answer_quote"] not in answer:
                    raise ValueError(f"Reviewer quotation absent: {cid}/{mode}")
            score = sum(item["fully_supported"] for item in judged["items"])
            if score != judged["supported_items"]:
                raise ValueError("Saved score and item labels disagree")
            row["supported_items"][mode] = score
            row["total_items"] = len(judged["items"])
            row["coverage"][mode] = score / row["total_items"]
            close(row["coverage"][mode], judged["coverage"])
            samples = [t["seconds"] for t in timing["timing_rows"] if t["mode"] == mode]
            if len(samples) != 30:
                raise ValueError("Expected 30 recorded warm retrieval samples")
            seconds = median(samples)
            close(seconds, metric["retrieval_median_seconds"][mode])
            row["retrieval_median_ms"][mode] = seconds * 1000
        row["paired_input_reduction"] = 1 - row["input_tokens"]["llm_wiki"] / row["input_tokens"]["source"]
        close(row["paired_input_reduction"], metric["paired_input_reduction"])
        rows.append(row)
    integration = [r for r in rows if r["id"] in {"N02", "N03", "N05", "N06"}]
    macro = {m: mean(r["coverage"][m] for r in rows) for m in ("source", "llm_wiki")}
    integrative = {m: mean(r["coverage"][m] for r in integration) for m in macro}
    for mode in macro:
        close(macro[mode], review["primary_macro_coverage"][mode])
        close(integrative[mode], review["integration_macro_coverage"][mode])
    reduction = mean(r["paired_input_reduction"] for r in rows)
    close(reduction, review["all6_mean_paired_input_reduction"])
    return dict(setup=dict(kind="ARCHIVED_LIVE_MEASUREMENTS_AND_AI_LABELS", cases=6,
                           model="deepseek-flash", reasoning="none", temperature=0,
                           max_output_tokens=2048, human_attested=False, fresh_provider_calls=0,
                           scope="Aggregate saved judgments and measurements, not fresh QA evaluation"),
                rows=rows, summary=dict(macro_coverage=macro,
                    integration_macro_coverage=integrative, mean_paired_input_reduction=reduction,
                    coverage_delta_pp=100 * (macro["llm_wiki"] - macro["source"]),
                    integration_delta_pp=100 * (integrative["llm_wiki"] - integrative["source"]),
                    complete_efficiency_acceptance=review["efficiency_target_complete_acceptance"],
                    unsupported_claims=review["unsupported_claims"],
                    accounting=review["accounting"]))


def navigation_metrics():
    # Generated from exact original functions at package build time, not provider output.
    import random
    from collections import Counter

    from navigation import ArticleStore, aggregate, recommend, shortest_paths

    store = ArticleStore(ROOT / "data/wiki")
    saved = read("data/navigation_reference.json")
    tasks = read("data/navigation_tasks.json")
    edges = {i: set(p.related) for i, p in store.pages.items()}
    names = sorted(edges)
    shuffled = names.copy()
    random.Random(19).shuffle(shuffled)
    mapping = dict(zip(names, shuffled, strict=True))
    permuted = {mapping[i]: {mapping[j] for j in js} for i, js in edges.items()}
    distances = Counter()
    for name in names:
        distances.update(len(p) - 1 for i, p in shortest_paths(edges, name).items() if i != name)
    total = len(names) * (len(names) - 1)
    structure = dict(nodes=len(names), directed_edges=sum(map(len, edges.values())),
        possible_ordered_pairs=total, reachable_pairs=sum(distances.values()),
        unreachable_pairs=total - sum(distances.values()),
        distance_histogram={str(k): v for k, v in sorted(distances.items())},
        reachable_within={str(k): sum(n for distance, n in distances.items() if distance <= k) / total
                          for k in (1, 2, 3)}, no_outgoing=sorted(i for i, js in edges.items() if not js))
    if structure != saved["structure"]:
        raise ValueError("Recomputed graph structure differs")
    conditions = {}
    for condition in ("entry_only", "full_intent"):
        rows = []
        for case in tasks:
            start = case["start"]
            page = store.pages[start]
            query = " ".join([page.title, *page.aliases]) if condition == "entry_only" else case["question"]
            direct, graph, paths = recommend(store, edges, start, query)
            _, placebo, _ = recommend(store, permuted, start, query)
            empty_direct, empty_graph, _ = recommend(store, {i: set() for i in names}, start, query)
            if direct != empty_graph or direct != empty_direct:
                raise ValueError("Empty graph does not recover baseline")
            targets = sorted(set(case["targets"]) - {start})
            rows.append(dict(id=case["id"], group=case["group"], start=start, query=query,
                targets=targets, direct=direct, graph=graph, permuted=placebo,
                graph_added=sorted(set(graph) - set(direct)), graph_removed=sorted(set(direct) - set(graph)),
                target_paths={i: paths.get(i) for i in targets},
                actual_graph_recommendation_paths={i: paths.get(i) for i in graph},
                hit_delta=len(set(graph) & set(targets)) - len(set(direct) & set(targets))))
        if rows != saved["conditions"][condition]["rows"]:
            raise ValueError("Fresh navigation rankings or paths differ from archived reference")
        stats = {a: aggregate(rows, a) for a in ("direct", "graph", "permuted")}
        stats["graph_vs_direct"] = dict(wins=sum(r["hit_delta"] > 0 for r in rows),
            ties=sum(r["hit_delta"] == 0 for r in rows), losses=sum(r["hit_delta"] < 0 for r in rows),
            macro_recall_delta_pp=100 * (stats["graph"]["macro_recall"] - stats["direct"]["macro_recall"]))
        if stats != saved["conditions"][condition]["statistics"]:
            raise ValueError("Navigation statistics differ")
        conditions[condition] = dict(rows=rows, statistics=stats)
    return dict(setup=dict(kind="FRESH_OFFLINE_NAVIGATION", slots=6, preserved_direct=3,
        max_hops=3, placebo_seed=19, provider_calls=0, human_attested=False,
        independent_user_trials=False), structure=structure, conditions=conditions)


def offline(output):
    count = verify()
    wiki = wiki_metrics()
    graph = navigation_metrics()
    write(output / "bench_wiki_vs_rag.json", wiki)
    write(output / "bench_graph_navigation.json", graph)
    expected_wiki = read("results/bench_wiki_vs_rag.json")
    expected_graph = read("results/bench_graph_navigation.json")
    if wiki != expected_wiki or graph != expected_graph:
        raise ValueError("Published result JSON and recomputation differ")
    write(output / "verification.json", dict(verified_files=count, fresh_provider_calls=0,
        results_match=True, navigation_rankings_and_paths_match=True,
        judgments="Archived nonblind AI decisions, not independently regenerated labels"))
    print(json.dumps(dict(verified_files=count, provider_calls=0,
        wiki_coverage=wiki["summary"]["macro_coverage"],
        input_reduction=wiki["summary"]["mean_paired_input_reduction"],
        graph_entry=graph["conditions"]["entry_only"]["statistics"],
        output=str(output)), ensure_ascii=False, indent=2))


def live(max_calls, output):
    """New independent, capped fixed-context generation; never reuse historical grants."""
    verify()
    if not 1 <= max_calls <= 14:
        raise ValueError("Explicit max-calls in 1..14 required; all attempts count")
    if output.exists():
        raise ValueError("Use a fresh output directory; failed/old runs are never overwritten")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("Set your own DEEPSEEK_API_KEY in the process environment")
    sys.path.insert(0, str(ROOT / "vendor/src"))
    from pydantic import BaseModel, ConfigDict, Field

    from jev_rag.config import LLMConfig
    from jev_rag.knowledge import nonpeak_allowed
    from jev_rag.llm.adapter import CompatibleAdapter, LiveGate
    from jev_rag.llm.structured import StructuredOutputPolicy, parse_json

    class Answer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answer: str = Field(min_length=1, max_length=18000)
        citations: list[str] = Field(max_length=20)

    if not nonpeak_allowed():
        raise ValueError("No requests during weekday 09:00–12:00 or 14:00–18:00 Asia/Shanghai")
    setup = read("data/wiki_setup.json")
    ledger = dict(type="NEW_FIXED_CONTEXT_GENERATION_NOT_HISTORICAL_RUN", cap=max_calls,
                  used=0, status="active", rows=[], cost_usd=None, model="deepseek-flash",
                  reused_planning=True, fresh_retrieval=False, human_attested=False)
    output.mkdir(parents=True)

    def consume(remaining):
        ledger["used"] = max_calls - remaining
        write(output / "ledger.json", ledger)  # persist before every network attempt

    gate = LiveGate(entrypoint="benchmark-live", max_calls=max_calls, on_consume=consume)
    config = LLMConfig.model_validate(setup["config"]["llm"])
    config.max_output_tokens = 2048
    cache = {}
    write(output / "ledger.json", ledger)
    try:
        for record in read("data/wiki_records.json"):
            request_id = record["request_contract_sha256"]
            row = dict(id=record["id"], mode=record["mode"], timestamp_utc=datetime.now(timezone.utc).isoformat())
            if request_id in cache:
                row.update(status="EXACT_REQUEST_REUSE", reused_from=cache[request_id])
                ledger["rows"].append(row)
                continue
            if gate.remaining == 0:
                ledger["status"] = "budget_exhausted"
                break
            if not nonpeak_allowed():
                ledger["status"] = "stopped_nonpeak_boundary"
                break
            adapter = CompatibleAdapter(config, api_key=key, gate=gate,
                structured_policy=StructuredOutputPolicy(responses_format="strict_function"),
                sampling_temperature=0.0)
            raw = None
            try:
                raw = adapter.complete(**record["request_kwargs"])
                parsed = Answer.model_validate(parse_json(raw))
                if not set(parsed.citations) <= set(record["citation_mapping"]):
                    raise ValueError("Unknown citation id")
                row.update(status="parsed", answer=parsed.model_dump(), raw_output=raw,
                           calls=adapter.records)
                cache[request_id] = record["id"] + "/" + record["mode"]
            except Exception as error:
                # No errors/headers that could leak credentials, no retries or fallback.
                row.update(status="failed", error_type=type(error).__name__, calls=adapter.records,
                           raw_output=raw)
                ledger["status"] = "stopped_after_failure"
            ledger["rows"].append(row)
            write(output / "ledger.json", ledger)
            if row["status"] == "failed":
                break
        else:
            ledger["status"] = "completed"
    finally:
        ledger["unused_closed"] = max_calls - ledger["used"]
        if ledger["status"] == "active":
            ledger["status"] = "interrupted_closed"
        write(output / "ledger.json", ledger)
    print(json.dumps({k: ledger[k] for k in ("status", "used", "unused_closed")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["offline", "benchmark-live"], nargs="?", default="offline")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "benchmark-live":
        if args.max_calls is None or args.output is None:
            parser.error("benchmark-live requires --max-calls and fresh --output")
        live(args.max_calls, args.output.resolve())
    else:
        offline((args.output or ROOT / "outputs/offline").resolve())


if __name__ == "__main__":
    main()
