"""Aggregate attempted calls, keeping unknown usage/cost unknown."""


def call_metrics(records, mode):
    if any(r["mode"] != mode for r in records):
        raise ValueError("cannot mix MOCK and LIVE call metrics")
    eligible = [r for r in records if r["task"] in {"generation", "reasoning"}]
    result = {
        "mode": mode,
        "llm_calls": len(records),
        "calls_by_model": {
            model: sum(r["model"] == model for r in records)
            for model in sorted({r["model"] for r in records})
        },
        "calls_by_task": {
            task: sum(r["task"] == task for r in records)
            for task in sorted({r["task"] for r in records})
        },
        "strong_model_escalation_rate": sum(r["tier"] == "strong" for r in eligible) / len(eligible)
        if eligible
        else None,
        "escalation_denominator": "attempted generation/reasoning calls",
        "eligible_escalation_calls": len(eligible),
    }
    for field in ("input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"):
        result[field] = (
            sum(r[field] for r in records)
            if records and all(r.get(field) is not None for r in records)
            else None
        )
    return result
