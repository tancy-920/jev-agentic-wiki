"""One-call format diagnostic from preserved real evidence, not a graph acceptance."""

import json
from uuid import UUID, uuid4

from jev_rag.agent.baseline_acceptance import recorded_contract
from jev_rag.agent.contracts import DecisionRequest, ProviderError
from jev_rag.io import read_jsonl, sha256_file, write_json
from jev_rag.schema import Chunk

from .adapter import CompatibleAdapter, LiveGate
from .credentials import load_credentials
from .decisions import LLMDecisions


def diagnostic_request(workspace, record, config, question):
    recorded_contract(record, question)
    if record["mode"] != "LIVE" or record["type"] != "LIVE_LLM_BASELINE_PILOT_NOT_BENCHMARK":
        raise ValueError("real baseline context required")
    chunks_path = workspace / config.paths.processed / "chunks.jsonl"
    if record["chunks_sha256"] != sha256_file(chunks_path):
        raise ValueError("stale diagnostic evidence")
    if record["config"]["llm"] != config.llm.model_dump():
        raise ValueError("model configuration drift")
    events = record["agent"]["events"]
    if not any(
        e["node"] == "evidence" and e["executed"] and e["selected_action"] == "SUFFICIENT"
        for e in events
    ):
        raise ValueError("sufficiency context missing")
    ids = [
        i
        for e in events
        if e["node"] == "relevance" and e["executed"] and e["selected_action"] == "RELEVANT"
        for i in e["input_summary"]["evidence_chunk_ids"]
    ]
    chunks = {
        c["chunk_id"]: Chunk.model_validate(c)
        for c in read_jsonl(chunks_path)
        if c["chunk_id"] in ids
    }
    if not ids or any(i not in chunks for i in ids):
        raise ValueError("missing diagnostic evidence")
    return DecisionRequest(
        task="loop",
        query=question,
        candidates=["STOP", "ESCALATE"],
        evidence=[chunks[i] for i in ids],
        iteration=record["agent"]["iteration"],
        evidence_sufficient=True,
    )


def run_loop_diagnostic(workspace, config, policy, question, context_run, environment):
    source_path = workspace / "results/reports/live_agent" / f"{UUID(context_run)}.json"
    record = json.loads(source_path.read_text(encoding="utf-8"))
    if record["policy"] != policy.snapshot():
        raise ValueError("diagnostic policy drift")
    request = diagnostic_request(workspace, record, config, question)
    key = load_credentials(workspace, environment, config.llm.provider)
    adapter = CompatibleAdapter(
        config.llm,
        api_key=key.get_secret_value(),
        gate=LiveGate(entrypoint="benchmark-live", max_calls=1),
        structured_policy=config.structured_output,
    )
    decisions = LLMDecisions(adapter, policy, format_version="legacy-v1")
    report = dict(
        type="LIVE_M4_LOOP_FORMAT_DIAGNOSTIC_NOT_ACCEPTANCE",
        mode="LIVE",
        run_id=str(uuid4()),
        source_run_id=record["run_id"],
        source_artifact_sha256=sha256_file(source_path),
        status="failed",
        ground_truth_reviewed=False,
        graph_acceptance=False,
    )
    try:
        report["decision"] = decisions.decide(request).model_dump()
        report["status"] = "decision_received"
    except ProviderError:
        report["error"] = "decision_provider_failure"
    finally:
        report["calls"] = adapter.records
        path = workspace / "results/reports/live_agent" / f"{report['run_id']}.json"
        write_json(path, report)
        print(
            f"Isolated loop diagnostic: {report['status']}; calls {len(adapter.records)}; saved {path}"
        )
    return 0 if report["status"] == "decision_received" else 1
