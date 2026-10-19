"""Bounded engineering acceptance and an exploratory, NOT M5, intent pilot."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from jev_rag.agent.baseline_acceptance import (
    recorded_contract,
    verify_baseline,
    verify_intent_execution,
    verify_partition_checks,
)
from jev_rag.config import Config
from jev_rag.io import write_json
from jev_rag.jev.pilot import run_live_pilot
from jev_rag.llm.adapter import LiveGate
from jev_rag.llm.metrics import call_metrics
from jev_rag.llm.policy import ModelPolicy

QUESTIONS = [
    {"id": "mechanism", "question": "How does scaled dot product attention work?"},
    {
        "id": "comparison",
        "question": "Compare BERT and GPT-3 pre-training objectives, not inference speed.",
    },
]


def paired_acceptance(baseline, proposed, question):
    """Require current live graphs and exact fixed-input/code parity, not quality gold."""
    baseline_audit = verify_baseline(baseline)
    if (
        proposed.get("mode") != "LIVE"
        or proposed.get("decision_layer") != "jev"
        or proposed.get("status") != "answered"
    ):
        raise ValueError("successful current live Jev graph required")
    if recorded_contract(baseline, question) != recorded_contract(proposed, question):
        raise ValueError("paired fixed-input drift")
    if not baseline.get("code_sha256") or baseline["code_sha256"] != proposed.get("code_sha256"):
        raise ValueError("paired code drift")
    tasks = {c["task"] for c in proposed["jev_calls"] if c["status"] == "completed"}
    if tasks != set(baseline_audit["live_tasks"]):
        raise ValueError("decision tasks not exercised equally")
    all_calls = [*proposed["jev_calls"], *proposed["llm_calls"]]
    if any(
        c["mode"] != "LIVE" or c["status"] != "completed" or c["http_status"] != 200
        for c in all_calls
    ):
        raise ValueError("failed or nonlive proposed calls")
    if len(all_calls) + proposed["remaining_call_budget"] != proposed["max_provider_calls"]:
        raise ValueError("proposed call accounting mismatch")
    verify_intent_execution(
        proposed, Config.model_validate(proposed["config"]), proposed["generation_calls"]
    )
    verify_partition_checks(proposed, Config.model_validate(proposed["config"]))
    accepted = {c["chunk_id"] for c in proposed["agent"]["relevant_documents"]}
    citations = proposed["agent"].get(
        "final_citation_results", proposed["agent"]["citation_results"]
    )
    if not citations or any(
        c["status"] != "SUPPORTED"
        or c["source"] != "LIVE"
        or not c["citation_ids"]
        or not set(c["citation_ids"]).issubset(accepted)
        for c in citations
    ):
        raise ValueError("proposed citation evidence mismatch")
    return {
        "passed": True,
        "baseline": baseline_audit,
        "paired_contract_hash": baseline["comparison_contract"]["hash"],
        "code_sha256": baseline["code_sha256"],
        "human_semantic_review": "pending",
        "quality_or_performance_advantage_proven": False,
    }


def arm_summary(record, case_id, planning):
    state = record.get("agent", {})
    return {
        "case_id": case_id,
        "planning": planning,
        "decision_layer": record["decision_layer"],
        "run_id": record["run_id"],
        "status": record["status"],
        "latency_ms": record.get("agent_latency_ms"),
        "iterations": state.get("iteration"),
        "intent_fallback": state.get("intent_fallback", False),
        "scheduled_retrieval_queries": state.get("retrieval_requests"),
        "returned_candidates": state.get("retrieved_candidates"),
        "final_relevant_passages": len(state.get("relevant_documents", [])),
        "llm_calls": len(record["llm_calls"]),
        "jev_calls": len(record["jev_calls"]),
        "llm_tokens": record["llm_metrics"]["total_tokens"],
        "jev_tokens": None
        if any(
            c.get("input_tokens") is None or c.get("output_tokens") is None
            for c in record["jev_calls"]
        )
        else sum(c["input_tokens"] + c["output_tokens"] for c in record["jev_calls"]),
        "estimated_cost_usd": None,
        "contract_hash": record["comparison_contract"]["hash"],
    }


def run_study(workspace: Path, config, max_calls: int, environment, prior_studies=None):
    if not 1 <= max_calls <= 80 or config.llm.max_output_tokens > 2048:
        raise ValueError("bounded live study cap <=80 calls and <=2048 output tokens required")
    previous = [str(UUID(value)) for value in (prior_studies or [])]
    if len(set(previous)) != len(previous):
        raise ValueError("duplicate budget histories")
    prior_used = 0
    for prior_id in previous:
        prior = json.loads(
            (workspace / "results/reports/live_study" / f"{prior_id}.json").read_text(
                encoding="utf-8"
            )
        )
        if prior.get("mode") != "LIVE" or type(prior.get("used_provider_calls")) is not int:
            raise ValueError("invalid prior authorization ledger")
        prior_used += prior["used_provider_calls"]
    if prior_used + max_calls > 80:
        raise ValueError("shared 80-call authorization would be exceeded")
    identifier = str(uuid4())
    path = workspace / "results/reports/live_study" / f"{identifier}.json"
    report = {
        "type": "LIVE_M4_ACCEPTANCE_AND_EXPLORATORY_INTENT_PILOT_NOT_MAIN_BENCHMARK",
        "mode": "LIVE",
        "study_id": identifier,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_cap": 80,
        "run_cap": max_calls,
        "prior_study_ids": previous,
        "prior_used_provider_calls": prior_used,
        "used_provider_calls": 0,
        "max_output_tokens": config.llm.max_output_tokens,
        "status": "initializing",
        "m4_acceptance": {"passed": False},
        "question_fixtures": QUESTIONS,
        "case_provenance": "public engineering queries, not human-reviewed held-out gold",
        "arms": [],
        "attempt_ids": [],
        "failure_ids": [],
        "quality_benchmark_completed": False,
        "independent_ground_truth": False,
        "limitations": [
            "two questions, one observation per arm, development pilot",
            "model decisions/citation verification are not human ground truth",
            "unknown costs stay null",
            "local source routing fixed in both on/off arms to isolate planning",
            "all failures retained; no best-run selection",
        ],
    }

    def persist_budget(remaining):
        report["used_provider_calls"] = max_calls - remaining
        report["total_authorization_used"] = prior_used + report["used_provider_calls"]
        write_json(path, report)  # Persist BEFORE the HTTP attempt, even if interrupted.

    gate = LiveGate(entrypoint="benchmark-live", max_calls=max_calls, on_consume=persist_budget)
    write_json(path, report)
    from jev_rag.retrieval import HybridRetriever

    retriever = None
    records = []
    try:
        retriever = HybridRetriever(config)

        def execute(case, layer, planning):
            if gate.remaining < 1:
                raise ValueError("study budget exhausted")
            shared = config.model_copy(deep=True)
            shared.intent = config.intent.model_copy(update={"enabled": planning})
            policy = ModelPolicy(shared.llm)  # Same predeclared FAST profile in every arm.
            cap = min(16, gate.remaining)
            child = LiveGate(entrypoint="benchmark-live", max_calls=cap, parent=gate)
            report["status"] = f"running:{case['id']}:{layer}:planning={planning}"
            write_json(path, report)
            record = run_live_pilot(
                workspace,
                shared,
                policy,
                case["question"],
                cap,
                environment,
                decision_layer=layer,
                gate=child,
                retriever=retriever,
                local_source_only=True,
                return_report=True,
            )
            if not isinstance(record, dict):
                raise ValueError("pilot failed before starting")
            records.append(record)
            report["attempt_ids"].append(record["run_id"])
            report["arms"].append(arm_summary(record, case["id"], planning))
            intent_attempts = sum(
                any(
                    c.get("decision_task", c.get("task")) == "intent"
                    for c in [*r["llm_decision_calls"], *r["jev_calls"]]
                )
                for r in records
            )
            fallback_count = sum(a["intent_fallback"] for a in report["arms"])
            report["intent_fallback_metrics"] = {
                "count": fallback_count,
                "attempts": intent_attempts,
                "rate": fallback_count / intent_attempts if intent_attempts else None,
            }
            if record["status"] != "answered":
                report["failure_ids"].append(record["run_id"])
            report["llm_metrics"] = call_metrics(
                [c for r in records for c in r["llm_calls"]], "LIVE"
            )
            write_json(path, report)
            return record

        # Engineering gate first; a failure prevents further paid exploratory runs.
        baseline = execute(QUESTIONS[0], "llm", True)
        if baseline["status"] != "answered":
            report["status"] = "m4_baseline_failed_stop"
            return 1
        proposed = execute(QUESTIONS[0], "jev", True)
        report["m4_acceptance"] = paired_acceptance(baseline, proposed, QUESTIONS[0]["question"])
        write_json(path, report)
        for layer in ("jev", "llm"):
            execute(QUESTIONS[0], layer, False)
        for planning in (False, True):
            for layer in ("llm", "jev"):
                if gate.remaining < 12:
                    report["status"] = "exploratory_pilot_budget_limited"
                    return 0
                execute(QUESTIONS[1], layer, planning)
        report["status"] = "exploratory_pilot_finished"
        return 0
    except Exception as error:
        report["status"] = "stopped_safely"
        report["error_type"] = type(error).__name__  # Never secret-bearing exception text.
        return 1
    finally:
        if retriever is not None:
            retriever.close()
        report["used_provider_calls"] = max_calls - gate.remaining
        report["total_authorization_used"] = prior_used + report["used_provider_calls"]
        report["remaining_budget"] = gate.remaining
        write_json(path, report)
        print(
            f"Study: {report['status']}; new provider calls {report['used_provider_calls']}/{max_calls}; report: {path}"
        )
