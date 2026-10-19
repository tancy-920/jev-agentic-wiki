"""Honest all-attempt metrics. Missing quality/usage is not a zero score."""

import math
import statistics

QUALITY = ("Faithfulness", "Answer Relevance", "Context Precision", "Context Recall")


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def aggregate(records):
    if not records or len({r["mode"] for r in records}) != 1:
        raise ValueError("nonempty single-mode records required")
    protocols = {
        r.get("quality_protocol")
        for r in records
        if any(r.get("quality", {}).get(k) is not None for k in QUALITY)
    }
    if len(protocols) > 1:
        raise ValueError("different quality protocols must not be pooled")

    def mean(field):
        values = [r.get(field) for r in records]
        return statistics.mean(values) if all(v is not None for v in values) else None

    latencies = [r.get("latency_ms") for r in records]
    iterations = [r.get("iterations") for r in records]
    quality = {}
    strong_counts = [r.get("strong_model_calls") for r in records]
    eligible_counts = [r.get("eligible_escalation_calls") for r in records]
    escalation = (
        sum(strong_counts) / sum(eligible_counts)
        if all(v is not None for v in [*strong_counts, *eligible_counts])
        and sum(eligible_counts) > 0
        else None
    )
    for key in QUALITY:
        scores = [r.get("quality", {}).get(key) for r in records]
        for record, score in zip(records, scores):
            if score is not None and (
                type(score) not in {int, float}
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or record.get("quality_source") not in {"HUMAN_REVIEW", "INDEPENDENT_EVALUATOR"}
            ):
                raise ValueError(
                    "quality must be independently scored, not confidence or invalid values"
                )
        quality[key] = statistics.mean(scores) if all(s is not None for s in scores) else None
    return {
        **quality,
        "Attempts": len(records),
        "Verified Answer Rate": sum(r["status"] == "answered" for r in records) / len(records),
        "Avg Latency": mean("latency_ms"),
        "P50 Latency": quantile(latencies, 0.5) if all(v is not None for v in latencies) else None,
        "P95 Latency": quantile(latencies, 0.95) if all(v is not None for v in latencies) else None,
        "Latency Std": statistics.stdev(latencies)
        if len(latencies) > 1 and all(v is not None for v in latencies)
        else None,
        "Avg LLM Calls": mean("llm_calls"),
        "Avg Jev Calls": mean("jev_calls"),
        "Strong Model Escalation Rate": escalation,
        "Avg Tokens": mean("total_tokens"),
        "Avg Input Tokens": mean("input_tokens"),
        "Avg Output Tokens": mean("output_tokens"),
        "Avg Cost": mean("estimated_cost_usd"),
        "Avg Iterations": mean("iterations"),
        "P95 Iterations": quantile(iterations, 0.95)
        if all(v is not None for v in iterations)
        else None,
        "Avg Observed Actions": mean("tool_calls"),
        "Loop Rate": sum(r.get("loop_count", 0) > 0 for r in records) / len(records),
        "Retry Rate": sum(r.get("retry_count", 0) > 0 for r in records) / len(records),
        "Early Stop Rate": sum(r.get("early_stop", False) for r in records) / len(records),
        "Intent Fallback Rate": sum(r.get("intent_fallback", False) for r in records)
        / len(records),
        "quality_scored_cases": sum(
            all(r.get("quality", {}).get(k) is not None for k in QUALITY) for r in records
        ),
    }
