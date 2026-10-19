"""Fingerprint all fixed inputs; the decision provider is deliberately excluded."""

import hashlib
import json


def comparison_contract(
    config,
    policy,
    question,
    corpus_hash,
    chunks_hash,
    *,
    version="paired-contract-v11",
    local_source_only=False,
    translation_summary=None,
):
    if version == "paired-contract-v11":
        # Historical contracts remain byte-for-byte reconstructible. The new
        # wire/parser policy is a controlled variable only in the new protocol.
        values = comparison_contract(
            config,
            policy,
            question,
            corpus_hash,
            chunks_hash,
            version="paired-contract-v10",
            local_source_only=local_source_only,
            translation_summary=translation_summary,
        )
        values.pop("hash")
        values.update(version=version, structured_output_policy=config.structured_output.snapshot())
        values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values
    if version not in {
        "paired-contract-v1",
        "paired-contract-v2",
        "paired-contract-v3",
        "paired-contract-v4",
        "paired-contract-v5",
        "paired-contract-v6",
        "paired-contract-v7",
        "paired-contract-v8",
        "paired-contract-v9",
        "paired-contract-v10",
    }:
        raise ValueError("unknown comparison contract version")
    agent = config.agent.model_dump()
    agent.pop("jev_mode", None)  # execution label, not an algorithmic parameter
    values = {
        "version": version,
        "seed": config.seed,
        "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "corpus_sha256": corpus_hash,
        "chunks_sha256": chunks_hash,
        "chunking": config.chunking.model_dump(),
        "embedding": config.embedding.model_dump(),
        "retrieval": config.retrieval.model_dump(),
        "qdrant": config.qdrant.model_dump(mode="json"),
        "agent": agent,
        "model_policy": policy.snapshot(),
        "relevance_batch_size": config.jev.relevance_batch_size,
        "allow_direct": False,
        "web_search_available": False,
        "actions": ["RETRIEVE", "REWRITE_QUERY", "GENERATE", "VERIFY", "STOP"],
    }
    if version == "paired-contract-v2":
        values["answer_policy"] = config.answer.snapshot(version="grounded-explanation-v1")
    if version in {
        "paired-contract-v3",
        "paired-contract-v4",
        "paired-contract-v5",
        "paired-contract-v6",
        "paired-contract-v7",
        "paired-contract-v8",
        "paired-contract-v9",
        "paired-contract-v10",
    }:
        values["answer_policy"] = config.answer.snapshot(
            version="grounded-explanation-v3"
            if version in {"paired-contract-v9", "paired-contract-v10"}
            else "grounded-explanation-v2"
        )
        values["intent_policy"] = config.intent.snapshot()
        values["decision_rubric_version"] = "original-query-authority-v2"
        values["retrieval_plan_execution"] = (
            "bounded-subqueries-global-rrf-original-query-rerank-v1"
        )
        values["local_routing_policy"] = "configured-local-source-plan-query-count-v1"
    if version in {
        "paired-contract-v4",
        "paired-contract-v5",
        "paired-contract-v6",
        "paired-contract-v7",
        "paired-contract-v8",
        "paired-contract-v9",
        "paired-contract-v10",
    }:
        values["local_source_only"] = local_source_only
    if version in {
        "paired-contract-v5",
        "paired-contract-v6",
        "paired-contract-v7",
        "paired-contract-v8",
        "paired-contract-v9",
        "paired-contract-v10",
    }:
        values["citation_uncertainty_repair_limit"] = 1
        values["schema_transport_version"] = "strict-required-minimal-v1"
    if version in {
        "paired-contract-v6",
        "paired-contract-v7",
        "paired-contract-v8",
        "paired-contract-v9",
        "paired-contract-v10",
    }:
        values["intent_uncertainty_policy"] = config.intent.uncertainty
        values["original_query_fallback_cycles"] = 1
    if version in {
        "paired-contract-v7",
        "paired-contract-v8",
        "paired-contract-v9",
        "paired-contract-v10",
    }:
        values["decision_rubric_version"] = "original-query-authority-v3-cross-language"
        values["language_workflow"] = "configured-generation-language-before-verification-v1"
        values["retrieval_query_language_policy"] = "english-paper-query-plan-v2-original-fallback"
    if version in {"paired-contract-v8", "paired-contract-v9", "paired-contract-v10"}:
        values["translation_policy"] = config.translation.snapshot()
        values["translation_model_rule"] = "shared-language-only-FAST-v1"
        values["retrieval_query_language_policy"] = (
            "faithful-English-full-query-fallback-and-rerank-v1"
        )
        values["translation_summary"] = translation_summary
        if translation_summary is not None and (
            not config.translation.enabled
            or translation_summary.get("original_sha256") != values["query_sha256"]
            or translation_summary.get("policy_sha256") != values["translation_policy"]["hash"]
        ):
            raise ValueError("translation contract binding mismatch")
    if version in {"paired-contract-v9", "paired-contract-v10"}:
        values["decision_rubric_version"] = "original-query-authority-v4-variant-referents"
    if version == "paired-contract-v10":
        values["verification_policy"] = config.verification.snapshot()
        values["decision_rubric_version"] = "original-query-authority-v5-parent-context"
    values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return values
