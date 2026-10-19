"""Explicit M3 live Agent pilot; only invoked by benchmark_live.py after confirmation."""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from jev_rag.agent.comparison_contract import comparison_contract
from jev_rag.agent.contracts import ProviderError
from jev_rag.agent.graph import ResearchAgent
from jev_rag.agent.intent import RetrievalPlan
from jev_rag.agent.language import QueryTranslation
from jev_rag.io import sha256_file, write_json
from jev_rag.llm.adapter import CompatibleAdapter, CompatibleGenerator, LiveGate
from jev_rag.llm.credentials import load_credentials
from jev_rag.llm.decisions import LLMDecisions
from jev_rag.llm.metrics import call_metrics

from .provider import JevDecisions


def run_live_pilot(
    workspace: Path,
    config,
    policy,
    question,
    max_calls,
    environment,
    decision_layer="jev",
    *,
    gate=None,
    retriever=None,
    local_source_only=False,
    return_report=False,
    prepared_translation=None,
):
    if decision_layer not in {"jev", "llm"}:
        raise ValueError("unknown decision layer")
    config = config.model_copy(deep=True)
    config.agent.jev_mode = "live"
    try:
        llm_key = load_credentials(workspace, environment, config.llm.provider)
        jev_key = (
            load_credentials(workspace, environment, "jev") if decision_layer == "jev" else None
        )
    except (ProviderError, OSError):
        print("LIVE NOT STARTED: required provider credentials are missing; no model calls.")
        return 1
    gate = gate or LiveGate(entrypoint="benchmark-live", max_calls=max_calls)
    if not isinstance(gate, LiveGate) or gate.remaining != max_calls:
        raise ValueError("fresh bounded gate required for each pilot")
    adapter = CompatibleAdapter(
        config.llm,
        api_key=llm_key.get_secret_value(),
        gate=gate,
        structured_policy=config.structured_output,
    )
    if decision_layer == "jev":
        decisions = JevDecisions(config.jev, api_key=jev_key.get_secret_value(), gate=gate)
    else:
        decision_adapter = CompatibleAdapter(
            config.llm,
            api_key=llm_key.get_secret_value(),
            gate=gate,
            structured_policy=config.structured_output,
        )
        decisions = LLMDecisions(
            decision_adapter, policy, relevance_batch_size=config.jev.relevance_batch_size
        )
    report = {
        "type": "LIVE_JEV_AGENT_PILOT_NOT_BENCHMARK"
        if decision_layer == "jev"
        else "LIVE_LLM_BASELINE_PILOT_NOT_BENCHMARK",
        "decision_layer": decision_layer,
        "mode": "LIVE",
        "run_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(mode="json"),
        "policy": policy.snapshot(),
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "max_provider_calls": max_calls,
        "ground_truth_reviewed": False,
        "status": "failed",
        "error": None,
    }
    owns_retriever = retriever is None
    try:
        revision = subprocess.run(
            ["git", "-c", f"safe.directory={workspace.as_posix()}", "rev-parse", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = subprocess.run(
            ["git", "-c", f"safe.directory={workspace.as_posix()}", "status", "--porcelain"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        report["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        source_files = sorted((workspace / "src").rglob("*.py"))
        code_hash = hashlib.sha256()
        for source_file in source_files:
            code_hash.update(source_file.relative_to(workspace).as_posix().encode())
            code_hash.update(source_file.read_bytes())
        report["code_sha256"] = code_hash.hexdigest()
        from jev_rag.retrieval import HybridRetriever

        report["corpus_manifest_sha256"] = sha256_file(workspace / config.paths.manifest)
        report["chunks_sha256"] = sha256_file(workspace / config.paths.processed / "chunks.jsonl")
        report["comparison_contract"] = comparison_contract(
            config,
            policy,
            question,
            report["corpus_manifest_sha256"],
            report["chunks_sha256"],
            local_source_only=local_source_only,
        )
        retriever = retriever or HybridRetriever(config)
        agent = ResearchAgent(
            retriever,
            decisions,
            CompatibleGenerator(adapter, policy, config.answer, config.verification),
            max_iterations=config.agent.max_iterations,
            live_gate=gate,
            relevance_uncertainty=config.agent.relevance_uncertainty,
            threshold=config.agent.decision_threshold,
            intent_policy=config.intent,
            retrieval_top_k=min(20, config.retrieval.top_k),
            local_source_only=local_source_only,
            translation_policy=config.translation,
            verification_policy=config.verification,
        )
        started = perf_counter()
        state = agent.run(question, prepared_translation=prepared_translation)
        report["agent_latency_ms"] = (perf_counter() - started) * 1000
        report["translation_summary"] = (
            QueryTranslation.model_validate(state["query_translation"]).summarize()
            if state["query_translation"]
            else None
        )
        report["comparison_contract"] = comparison_contract(
            config,
            policy,
            question,
            report["corpus_manifest_sha256"],
            report["chunks_sha256"],
            local_source_only=local_source_only,
            translation_summary=report["translation_summary"],
        )
        report["agent"] = {
            key: state[key]
            for key in (
                "run_id",
                "mode",
                "route",
                "intent",
                "intent_fallback",
                "retrieval_requests",
                "retrieved_candidates",
                "retrieved_documents",
                "relevant_documents",
                "iteration",
                "max_iterations",
                "retry_count",
                "loop_count",
                "early_stop",
                "termination_reason",
                "events",
                "tool_calls",
                "citation_results",
                "final_citation_results",
                "citation_repair_count",
                "draft_answer",
                "draft_history",
                "final_answer",
            )
        }
        report["status"] = state["termination_reason"]
        report["plan_summary"] = (
            None
            if state["retrieval_plan"] is None
            else RetrievalPlan.model_validate(state["retrieval_plan"]).summary()
        )
        # Provider judgments remain predictions, never human-reviewed semantic gold.
        report["semantic_support_source"] = f"{decision_layer} predictions, not ground truth"
    except Exception as error:
        report["error"] = "live_agent_pilot_failure"
        report["error_type"] = type(error).__name__
    finally:
        if retriever is not None and owns_retriever:
            retriever.close()
        report["jev_calls"] = decisions.records if decision_layer == "jev" else []
        report["llm_decision_calls"] = decisions.records if decision_layer == "llm" else []
        report["generation_calls"] = adapter.records
        report["llm_calls"] = sorted(
            [*report["llm_decision_calls"], *adapter.records],
            key=lambda c: datetime.fromisoformat(c["timestamp_utc"]),
        )
        report["remaining_call_budget"] = gate.remaining
        report["generation_metrics"] = adapter.metrics()
        report["llm_metrics"] = call_metrics(report["llm_calls"], "LIVE")
        report["estimated_cost_usd"] = None
        path = workspace / "results/reports/live_agent" / f"{report['run_id']}.json"
        write_json(path, report)
        print(
            f"LIVE pilot status: {report['status']}; Jev calls {len(report['jev_calls'])}; LLM calls {len(report['llm_calls'])}."
        )
        print(f"Saved pilot (not benchmark/ground truth): {path}")
    return report if return_report else 0 if report["status"] == "answered" else 1
