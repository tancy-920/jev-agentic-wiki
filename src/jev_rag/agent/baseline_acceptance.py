"""Engineering-artifact validation, not human semantic ground truth."""

import hashlib

from jev_rag.config import Config
from jev_rag.llm.metrics import call_metrics
from jev_rag.llm.policy import ModelPolicy, QueryProfile

from .comparison_contract import comparison_contract
from .verification import checks_for, validate_draft


def verify_partition_checks(record, config):
    if config.verification.mode == "paragraph":
        return
    state = record["agent"]
    draft = validate_draft(
        state["draft_answer"], config.verification, max_claims=config.llm.max_claims
    )
    expected = []
    for paragraph_index, paragraph in enumerate(draft.claims):
        for claim, parent, span_index in checks_for(paragraph, config.verification):
            expected.append(
                (
                    paragraph_index,
                    "span" if parent else "paragraph",
                    span_index,
                    claim.text,
                    claim.citations,
                )
            )
    actual = [
        (
            r.get("paragraph_index"),
            r.get("scope"),
            r.get("span_index"),
            r.get("claim"),
            r.get("citation_ids"),
        )
        for r in state["final_citation_results"]
    ]
    if actual != expected:
        raise ValueError("missing, reordered or changed paragraph/span verification")


def recorded_contract(record, question):
    if record["question_sha256"] != hashlib.sha256(question.encode()).hexdigest():
        raise ValueError("question mismatch")
    config = Config.model_validate(record["config"])
    policy = ModelPolicy(config.llm, QueryProfile.model_validate(record["policy"]["profile"]))
    if policy.snapshot() != record["policy"]:
        raise ValueError("invalid policy snapshot")
    contract = comparison_contract(
        config,
        policy,
        question,
        record["corpus_manifest_sha256"],
        record["chunks_sha256"],
        version=record.get("comparison_contract", {}).get("version", "paired-contract-v1"),
        local_source_only=record.get("comparison_contract", {}).get("local_source_only", False),
        translation_summary=record.get("translation_summary"),
    )
    if "comparison_contract" in record and record["comparison_contract"] != contract:
        raise ValueError("invalid comparison contract")
    return contract


def verify_intent_execution(record, config, generation):
    state = record["agent"]
    original_hash = record["question_sha256"]
    fallback_hash, fallback_tag = original_hash, "ORIGINAL_QUERY_LOCAL_RETRIEVAL"
    if config.translation.enabled:
        summary = record.get("translation_summary")
        prepared = [e for e in state["events"] if e["node"] == "prepare_query" and e["executed"]]
        if (
            not summary
            or summary.get("original_sha256") != original_hash
            or summary.get("policy_sha256") != config.translation.snapshot()["hash"]
            or summary.get("source") not in {"LIVE", "IDENTITY"}
            or len(prepared) != 1
            or any(prepared[0].get(k) != v for k, v in summary.items())
        ):
            raise ValueError("faithful translation binding/event mismatch")
        fallback_hash = summary["english_sha256"]
        fallback_tag = "ORIGINAL_TRANSLATED_QUERY_LOCAL_RETRIEVAL"
    fallback = state.get("intent_fallback", False)
    if type(fallback) is not bool or (fallback and not config.intent.enabled):
        raise ValueError("invalid intent fallback flag")
    if not config.intent.enabled:
        return
    if not any(
        e["node"] == "routing" and e["source"] == "application" and e["executed"]
        for e in state["events"]
    ):
        raise ValueError("validated local routing required")
    if not fallback:
        if not any(
            c["task"] == "reasoning" and c.get("purpose") == "retrieval_planning"
            for c in generation
        ):
            raise ValueError("shared retrieval planning required")
        return
    if (
        config.intent.uncertainty != "original_query"
        or state.get("intent") is not None
        or record.get("plan_summary") is not None
        or state.get("route") != "LOCAL_RAG"
        or state.get("iteration") != 1
        or state.get("retrieval_requests") != 1
        or any(c["task"] in {"rewrite", "reasoning"} for c in generation)
    ):
        raise ValueError("original-query fallback must not adopt, plan or rewrite intent")
    uncertain = [e for e in state["events"] if e["node"] == "intent"]
    retrieval = [e for e in state["events"] if e["node"] == "retrieve" and e["executed"]]
    if (
        len(uncertain) != 1
        or uncertain[0]["executed"]
        or uncertain[0].get("error") != "uncertain_decision"
        or uncertain[0].get("application_fallback") != fallback_tag
        or not any(
            v is None or v < config.agent.decision_threshold
            for v in (uncertain[0].get("probability"), uncertain[0].get("confidence"))
        )
        or len(retrieval) != 1
        or retrieval[0].get("query_count") != 1
        or retrieval[0].get("query_sha256s") != [fallback_hash]
    ):
        raise ValueError("uncertain original-query fallback audit mismatch")


def verify_baseline(record):
    if (
        record["type"] != "LIVE_LLM_BASELINE_PILOT_NOT_BENCHMARK"
        or record["mode"] != "LIVE"
        or record["decision_layer"] != "llm"
    ):
        raise ValueError("live baseline required")
    if record["status"] != "answered" or record["agent"]["mode"] != "LIVE":
        raise ValueError("successful live graph required")
    if record["jev_calls"]:
        raise ValueError("baseline must not call Jev")
    decisions, generation = record["llm_decision_calls"], record["generation_calls"]
    calls = record["llm_calls"]
    if len(calls) != len(decisions) + len(generation) or any(
        calls.count(c) != 1 for c in [*decisions, *generation]
    ):
        raise ValueError("duplicated or missing call records")
    if any(
        c["mode"] != "LIVE" or c["status"] != "completed" or c["http_status"] != 200 for c in calls
    ):
        raise ValueError("failed/nonlive calls in successful pilot")
    config = Config.model_validate(record["config"])
    policy = ModelPolicy(config.llm, QueryProfile.model_validate(record["policy"]["profile"]))
    if record["policy"] != policy.snapshot() or any(
        c["policy_hash"] != policy.snapshot()["hash"]
        or c["model"] != policy.model(c["task"])
        or c["tier"] != policy.tier(c["task"]).value
        for c in calls
    ):
        raise ValueError("model policy mismatch")
    required_tasks = {
        "routing",
        "relevance",
        "evidence",
        "loop",
        "citation",
    }
    if config.intent.enabled:
        required_tasks = (required_tasks - {"routing"}) | {"intent"}
    elif record.get("comparison_contract", {}).get("local_source_only", False):
        required_tasks -= {"routing"}
    verify_intent_execution(record, config, generation)
    verify_partition_checks(record, config)
    if {c["decision_task"] for c in decisions} != required_tasks:
        raise ValueError("all configured decision tasks required")
    if not any(c["task"] == "generation" for c in generation) or any(
        c["task"] != "decision" or c["tier"] != "fast" for c in decisions
    ):
        raise ValueError("baseline decision tier/task mismatch")
    if not 1 <= record["agent"]["iteration"] <= record["agent"]["max_iterations"] <= 5:
        raise ValueError("iteration bound failed")
    if len(calls) + record["remaining_call_budget"] != record["max_provider_calls"]:
        raise ValueError("call budget accounting failed")
    accepted = {
        i
        for e in record["agent"]["events"]
        if e["node"] == "relevance" and e["executed"] and e["selected_action"] == "RELEVANT"
        for i in e["input_summary"]["evidence_chunk_ids"]
    }
    citations = record["agent"].get("final_citation_results", record["agent"]["citation_results"])
    if not citations or any(
        c["status"] != "SUPPORTED"
        or c["source"] != "LIVE"
        or not c["citation_ids"]
        or not set(c["citation_ids"]).issubset(accepted)
        for c in citations
    ):
        raise ValueError("model verification/citation IDs failed")
    metrics = call_metrics(calls, "LIVE")
    if record["llm_metrics"] != metrics:
        raise ValueError("aggregate metrics mismatch")
    return {
        "live_tasks": sorted({c["decision_task"] for c in decisions}),
        "intent_fallback": record["agent"].get("intent_fallback", False),
        "claims_model_verified": sum(c.get("scope") != "span" for c in citations),
        **(
            {"span_checks_model_verified": sum(c.get("scope") == "span" for c in citations)}
            if config.verification.mode != "paragraph"
            else {}
        ),
        "llm_metrics": metrics,
        "human_semantic_review_status": "pending",
    }
