"""Offline paired-run planning: no credentials, retriever, model or provider access."""

import hashlib
import json

from jev_rag.evaluation.dataset import load_packet
from jev_rag.io import read_jsonl, sha256_file
from jev_rag.llm.policy import ModelPolicy
from jev_rag.schema import Chunk


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def runtime_hash(workspace):
    digest = hashlib.sha256()
    for path in sorted(
        [*(workspace / "src").rglob("*.py"), *(workspace / "experiments").rglob("*.py")]
    ):
        digest.update(path.relative_to(workspace).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def plan_benchmark(workspace, config, packet, *, seeds=(13, 29, 47)):
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("unique bounded integer repeated-run seeds required")
    chunks_path = workspace / config.paths.processed / "chunks.jsonl"
    chunks = {c["chunk_id"]: Chunk.model_validate(c) for c in read_jsonl(chunks_path)}
    manifest, questions, approved, audit = load_packet(packet, chunks=chunks)
    corpus_hash, chunks_hash = (
        sha256_file(workspace / config.paths.manifest),
        sha256_file(chunks_path),
    )
    if manifest["corpus_sha256"] != corpus_hash or manifest["chunks_sha256"] != chunks_hash:
        raise ValueError("current corpus differs from reviewed question sources")
    cases = []
    for q in questions:
        snapshot = ModelPolicy(config.llm, q.input_profile).snapshot()
        cases.append(
            {
                "case_id": q.id,
                "category": q.category,
                "input_profile": q.input_profile.model_dump(),
                "model_policy_sha256": snapshot["hash"],
                "generation_model": ModelPolicy(config.llm, q.input_profile).model("generation"),
            }
        )
    binding = {
        "dataset_manifest_sha256": sha256_file(packet / "manifest.json"),
        "questions_sha256": manifest["questions_sha256"],
        "reviews_sha256": sha256_file(packet / "reviews.jsonl"),
        "corpus_sha256": corpus_hash,
        "chunks_sha256": chunks_hash,
        "config_sha256": canonical_hash(config.model_dump(mode="json")),
        "runtime_source_sha256": runtime_hash(workspace),
        "seeds": list(seeds),
        "layers": ["llm", "jev"],
        "case_policies": cases,
    }
    return {
        "type": "OFFLINE_PAIRED_BENCHMARK_PLAN_NOT_RESULTS",
        "dataset_version": manifest["dataset_version"],
        "split": manifest["split"],
        "case_count": len(questions),
        "approved_case_count": len(approved),
        "review_audit": audit,
        "dataset_ready": audit["formal_release_ready"],
        "planned_agent_runs": len(questions) * len(seeds) * 2,
        "planned_translation_inputs": len(questions) * len(seeds)
        if config.translation.enabled
        else 0,
        "shared_translation_policy": "one physical result shared by both arms; logical costs charged equally",
        "binding": binding,
        "binding_sha256": canonical_hash(binding),
        "new_provider_calls": 0,
        "performance_results_generated": False,
        "estimated_request_count": None,
        "estimated_cost_usd": None,
        "live_authorization_checked": False,
        "blockers": (
            []
            if audit["formal_release_ready"]
            else ["100+ source-validated, explicitly human-reviewed held-out cases required"]
        )
        + [
            "fresh finite benchmark-live authorization not checked by this offline planner",
            "independent source review of actual LIVE output assessments still required after runs",
        ],
        "limits": [
            "planned run counts are not actual model calls, measured metrics or a bill",
            "same-source/topic cases are correlated; this is not a claim of independent observations",
            "a seed records the planned run, not guaranteed API determinism",
            "freeze again after review; changed sources/config/labels must not reuse this binding",
        ],
    }


def validate_plan(workspace, config, packet, plan, *, seeds=(13, 29, 47)):
    if plan.get("type") != "OFFLINE_PAIRED_BENCHMARK_PLAN_NOT_RESULTS":
        raise ValueError("offline plan required, not benchmark results")
    current = plan_benchmark(workspace, config, packet, seeds=seeds)
    if (
        plan.get("binding_sha256") != canonical_hash(plan.get("binding"))
        or plan["binding_sha256"] != current["binding_sha256"]
    ):
        raise ValueError("dataset/source/config/review/run-plan drift; create a new explicit plan")
    return current
