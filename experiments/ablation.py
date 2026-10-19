"""M6 fixed component matrix. MOCK by default; live dispatch needs benchmark-live."""

import json
import random
import subprocess
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID, uuid4

from jev_rag.agent.ablation import VARIANTS, MixedDecisions
from jev_rag.agent.comparison_contract import comparison_contract
from jev_rag.agent.graph import ResearchAgent
from jev_rag.agent.language import QueryTranslation
from jev_rag.agent.mock import MockDecisions, MockGenerator
from jev_rag.evaluation.dataset import load_packet
from jev_rag.evaluation.metrics import QUALITY
from jev_rag.evaluation.preflight import canonical_hash, runtime_hash
from jev_rag.io import read_jsonl, sha256_file, write_json
from jev_rag.jev.provider import JevDecisions
from jev_rag.llm.adapter import CompatibleAdapter, CompatibleGenerator, LiveGate
from jev_rag.llm.credentials import load_credentials
from jev_rag.llm.decisions import LLMDecisions
from jev_rag.llm.policy import ModelPolicy
from jev_rag.schema import Chunk

BUDGET_POLICY = {"version": "m6-fixed-tier-output-v1", "fast": 2048, "strong": 8192}
ARM_CAP = 40


class TierBudgetAdapter(CompatibleAdapter):
    """Shared M6 output policy; original M5 adapter and configuration stay unchanged."""

    def complete(self, **kwargs):
        tier = kwargs.get("tier")
        expected = self.config.model if tier == "fast" else self.config.strong_model
        if (
            tier not in {"fast", "strong"}
            or kwargs.get("model") != expected
            or (tier == "strong" and kwargs.get("task") not in {"generation", "reasoning"})
        ):
            raise ValueError("fixed model-tier-output policy required")
        original = self.config.max_output_tokens
        before = len(self.records)
        self.config.max_output_tokens = BUDGET_POLICY[tier]
        try:
            return super().complete(**kwargs)
        finally:
            self.config.max_output_tokens = original
            if len(self.records) > before:
                self.records[-1]["max_output_tokens"] = BUDGET_POLICY[tier]
                self.records[-1]["output_budget_policy"] = BUDGET_POLICY.copy()


def sources(workspace, config, packet):
    chunks_path = workspace / config.paths.processed / "chunks.jsonl"
    chunks = {c["chunk_id"]: Chunk.model_validate(c) for c in read_jsonl(chunks_path)}
    manifest, questions, approved, audit = load_packet(packet, chunks=chunks, formal=False)
    if manifest["split"] != "development" or not approved or len(questions) != len(approved):
        raise ValueError("all questions must be source-validated human-confirmed DEVELOPMENT data")
    if len(approved) > 12:
        raise ValueError("first-stage ablation is limited to a fixed small packet")
    if manifest["corpus_sha256"] != sha256_file(workspace / config.paths.manifest) or manifest[
        "chunks_sha256"
    ] != sha256_file(chunks_path):
        raise ValueError("reviewed corpus changed")
    return manifest, approved, audit


def plan_ablation(workspace, config, packet, seeds=(13, 29, 47)):
    if (
        not seeds
        or len(seeds) > 3
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or config.llm.max_output_tokens != 2048
        or not config.intent.enabled
        or not config.translation.enabled
    ):
        raise ValueError("fixed M6 base cap, intent/translation and valid repeat IDs required")
    manifest, questions, audit = sources(workspace, config, packet)
    binding = {
        "dataset_sha256": manifest["questions_sha256"],
        "manifest_sha256": sha256_file(packet / "manifest.json"),
        "reviews_sha256": sha256_file(packet / "reviews.jsonl"),
        "corpus_sha256": manifest["corpus_sha256"],
        "chunks_sha256": manifest["chunks_sha256"],
        "config": config.model_dump(mode="json"),
        "runtime_sha256": runtime_hash(workspace),
        "execution_scripts_sha256": {
            name: sha256_file(workspace / name)
            for name in ("scripts/benchmark_live.py", "scripts/m6_ablation.py")
            if (workspace / name).is_file()
        },
        "case_ids": [q.id for q in questions],
        "seeds": list(seeds),
        "variants": {k: list(v) for k, v in VARIANTS.items()},
        "output_budget_policy": BUDGET_POLICY.copy(),
        "per_arm_provider_cap": ARM_CAP,
        "model_policies": {
            q.id: ModelPolicy(config.llm, q.input_profile).snapshot() for q in questions
        },
        "order": "Latin rotation by case index plus seed index; translation once per case/repeat",
    }
    return {
        "type": "OFFLINE_M6_DEVELOPMENT_ABLATION_PLAN",
        "binding": binding,
        "binding_sha256": canonical_hash(binding),
        "review_audit": audit,
        "dataset_version": manifest["dataset_version"],
        "split": "development",
        "planned_agent_runs": len(questions) * len(seeds) * 5,
        "new_provider_calls": 0,
        "estimated_cost_usd": None,
        "limits": [
            "Predeclared small development screen, not fresh held-out quality evaluation.",
            "All three repeats are planned before results; finite budgets may pause the matrix.",
            "M6 has a new shared Pro output budget; do not reuse M5 A0/A4 results as these arms.",
            "A3-to-A4 changes intent and citation together; marginal effects depend on order/interactions.",
            "Main routing is deterministic application policy; no artificial paid router call.",
            "New generated answers require their own source review and exact human confirmation.",
        ],
    }


def validate_plan(workspace, config, packet, plan):
    current = plan_ablation(workspace, config, packet, tuple(plan["binding"]["seeds"]))
    if (
        plan.get("type") != current["type"]
        or plan.get("binding_sha256") != canonical_hash(plan.get("binding"))
        or plan["binding_sha256"] != current["binding_sha256"]
    ):
        raise ValueError("M6 source/policy/packet drift; freeze a new version, not reuse old runs")
    return current


def execution_order(seed_index, case_index):
    names = list(VARIANTS)
    shift = (seed_index + case_index) % len(names)
    return names[shift:] + names[:shift]


def run_matrix(
    workspace,
    config,
    packet,
    plan,
    *,
    gate=None,
    environment=None,
    retriever=None,
    resume=None,
    mock_translations=None,
):
    current = validate_plan(workspace, config, packet, plan)
    mode = "LIVE" if gate is not None else "MOCK"
    if mode == "LIVE" and (not isinstance(gate, LiveGate) or gate.parent is not None):
        raise ValueError("root benchmark-live gate required")
    _, questions, _ = sources(workspace, config, packet)
    report = {
        "type": "LIVE_M6_DEVELOPMENT_ABLATION" if mode == "LIVE" else "MOCK_ABLATION_NOT_RESEARCH",
        "mode": mode,
        "experiment_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "plan": current,
        "plan_binding_sha256": current["binding_sha256"],
        "config": config.model_dump(mode="json"),
        "seeds": plan["binding"]["seeds"],
        "case_ids": plan["binding"]["case_ids"],
        "variants": list(VARIANTS),
        "expected_runs": plan["planned_agent_runs"],
        "records": [],
        "translations": [],
        "complete": False,
        "status": "initializing",
        "in_progress": None,
        "quality_source": "unmeasured; no provider judgment is ground truth",
        "split": "development",
    }
    for key, arguments in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_dirty", ["status", "--porcelain"]),
    ):
        process = subprocess.run(
            ["git", "-c", f"safe.directory={workspace.as_posix()}", *arguments],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        report[key] = (
            (process.stdout.strip() if key == "git_commit" else bool(process.stdout.strip()))
            if process.returncode == 0
            else None
        )
    output = workspace / "results" / mode.lower() / "ablation" / report["experiment_id"]
    if resume:
        output = workspace / "results" / mode.lower() / "ablation" / str(UUID(resume))
        previous = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        keys = [(r["case_id"], r["seed"], r["variant"]) for r in previous["records"]]
        expected = {(q.id, s, v) for q in questions for s in report["seeds"] for v in VARIANTS}
        if (
            previous.get("plan_binding_sha256") != current["binding_sha256"]
            or previous.get("plan") != current
            or any(
                previous.get(k) != report[k]
                for k in ("case_ids", "seeds", "variants", "expected_runs", "config")
            )
            or previous.get("mode") != mode
            or previous.get("complete")
            or previous.get("status") != "budget_exhausted"
            or previous.get("in_progress") is not None
            or len(set(keys)) != len(keys)
            or not set(keys) <= expected
            or any(
                json.loads((output / "raw" / f"{r['run_id']}.json").read_text(encoding="utf-8"))
                != r
                for r in previous["records"]
            )
            or any(
                t["status"] not in {"completed", "translation_failure"}
                for t in previous["translations"]
            )
        ):
            raise ValueError(
                "only intact budget-paused matrices can resume; never retry interrupted calls"
            )
        write_json(output / "sessions" / f"{uuid4()}.json", previous)
        report = previous
        report["status"] = "resuming"
    keys = None
    if mode == "LIVE":
        keys = {
            p: load_credentials(workspace, environment or {}, p).get_secret_value()
            for p in (config.llm.provider, "jev")
        }
    owned = retriever is None
    write_json(output / "experiment.json", report)
    try:
        if retriever is None:
            from jev_rag.retrieval import HybridRetriever

            retriever = HybridRetriever(config)
        for si, seed in enumerate(report["seeds"]):
            random.seed(seed)
            run_config = config.model_copy(deep=True)
            run_config.seed = seed
            for qi, question in enumerate(questions):
                done = {
                    r["variant"]
                    for r in report["records"]
                    if r["case_id"] == question.id and r["seed"] == seed
                }
                if done == set(VARIANTS):
                    continue
                policy = ModelPolicy(run_config.llm, question.input_profile)
                cached = [
                    t
                    for t in report["translations"]
                    if t["case_id"] == question.id and t["seed"] == seed
                ]
                if len(cached) > 1:
                    raise ValueError("duplicate shared translation")
                if cached:
                    common = cached[0]
                else:
                    if mode == "LIVE" and gate.remaining < ARM_CAP + 1:
                        report["status"] = "budget_exhausted"
                        return report, output
                    common = {
                        "id": str(uuid4()),
                        "case_id": question.id,
                        "seed": seed,
                        "status": "initializing",
                        "calls": [],
                        "prepared": None,
                    }
                    report["translations"].append(common)
                    write_json(output / "experiment.json", report)
                    adapter = None
                    before = perf_counter()
                    try:
                        if mode == "LIVE":
                            adapter = TierBudgetAdapter(
                                run_config.llm,
                                api_key=keys[config.llm.provider],
                                gate=gate,
                                structured_policy=config.structured_output,
                            )
                            generator = CompatibleGenerator(
                                adapter, policy, config.answer, config.verification
                            )
                        else:
                            generator = MockGenerator(
                                translations=(mock_translations or {}).get(question.id, [])
                            )
                        prepared = generator.translate_query(question.question, config.translation)
                        common.update(status="completed", prepared=prepared.model_dump(mode="json"))
                    except Exception as error:
                        common.update(status="translation_failure", error_type=type(error).__name__)
                    finally:
                        common.update(
                            calls=adapter.records if adapter else [],
                            latency_ms=(perf_counter() - before) * 1000,
                        )
                        write_json(output / "experiment.json", report)
                prepared = (
                    QueryTranslation.model_validate(common["prepared"])
                    if common["prepared"]
                    else None
                )
                if prepared:
                    prepared.validate_for(question.question, config.translation)
                contract = comparison_contract(
                    run_config,
                    policy,
                    question.question,
                    current["binding"]["corpus_sha256"],
                    current["binding"]["chunks_sha256"],
                    local_source_only=True,
                    translation_summary=prepared.summarize() if prepared else None,
                )
                control_hash = canonical_hash(
                    {
                        "contract": contract,
                        "output_budget_policy": BUDGET_POLICY,
                        "per_arm_cap": ARM_CAP,
                    }
                )
                for variant in execution_order(si, qi):
                    if variant in done:
                        continue
                    if (
                        mode == "LIVE"
                        and common["status"] != "translation_failure"
                        and gate.remaining < ARM_CAP
                    ):
                        report["status"] = "budget_exhausted"
                        return report, output
                    run_id = str(uuid4())
                    report["in_progress"] = {
                        "run_id": run_id,
                        "case_id": question.id,
                        "seed": seed,
                        "variant": variant,
                    }
                    write_json(output / "experiment.json", report)
                    llm_calls, jev_calls, dispatches = [], [], []
                    child = None
                    before = perf_counter()
                    if common["status"] == "translation_failure":
                        state = {
                            "termination_reason": "translation_failure",
                            "iteration": 0,
                            "final_answer": None,
                            "events": [],
                        }
                    else:
                        if mode == "LIVE":
                            child = LiveGate(
                                entrypoint="benchmark-live", max_calls=ARM_CAP, parent=gate
                            )
                            adapter = TierBudgetAdapter(
                                run_config.llm,
                                api_key=keys[config.llm.provider],
                                gate=child,
                                structured_policy=config.structured_output,
                            )
                            decision_adapter = TierBudgetAdapter(
                                run_config.llm,
                                api_key=keys[config.llm.provider],
                                gate=child,
                                structured_policy=config.structured_output,
                            )
                            ld = LLMDecisions(
                                decision_adapter, policy, config.jev.relevance_batch_size
                            )
                            jd = JevDecisions(config.jev, api_key=keys["jev"], gate=child)
                            generator = CompatibleGenerator(
                                adapter, policy, config.answer, config.verification
                            )
                        else:
                            ld, jd = MockDecisions(), MockDecisions()
                            generator = MockGenerator(
                                answer_policy=config.answer, verification_policy=config.verification
                            )
                        mixed = MixedDecisions(ld, jd, variant)
                        agent = ResearchAgent(
                            retriever,
                            mixed,
                            generator,
                            max_iterations=config.agent.max_iterations,
                            threshold=config.agent.decision_threshold,
                            live_gate=child,
                            relevance_uncertainty=config.agent.relevance_uncertainty,
                            intent_policy=config.intent,
                            retrieval_top_k=min(20, config.retrieval.top_k),
                            local_source_only=True,
                            answer_policy=config.answer,
                            translation_policy=config.translation,
                            verification_policy=config.verification,
                        )
                        state = agent.run(question.question, prepared_translation=prepared)
                        dispatches = mixed.dispatches
                        if mode == "LIVE":
                            llm_calls = [*adapter.records, *decision_adapter.records]
                            jev_calls = jd.records
                    accounted = [*common["calls"], *llm_calls, *jev_calls]

                    def total(field):
                        values = [c.get(field) for c in accounted]
                        return (
                            sum(values) if values and all(v is not None for v in values) else None
                        )

                    record = {
                        "mode": mode,
                        "run_id": run_id,
                        "case_id": question.id,
                        "seed": seed,
                        "variant": variant,
                        "jev_tasks": list(VARIANTS[variant]),
                        "contract_hash": control_hash,
                        "profile": question.input_profile.model_dump(),
                        "model_policy": policy.snapshot(),
                        "status": state["termination_reason"],
                        "answer": state.get("final_answer"),
                        "latency_ms": (perf_counter() - before) * 1000 + common["latency_ms"],
                        "llm_calls": len(llm_calls) + len(common["calls"])
                        if mode == "LIVE"
                        else None,
                        "jev_calls": len(jev_calls) if mode == "LIVE" else None,
                        "input_tokens": total("input_tokens"),
                        "output_tokens": total("output_tokens"),
                        "total_tokens": total("total_tokens"),
                        "estimated_cost_usd": None,
                        "physical_provider_calls": ARM_CAP - child.remaining if child else 0,
                        "provider_records": {"llm": llm_calls, "jev": jev_calls},
                        "shared_translation_id": common["id"],
                        "decision_dispatches": dispatches,
                        "iterations": state.get("iteration"),
                        "tool_calls": len(state.get("tool_calls", [])),
                        "loop_count": state.get("loop_count", 0),
                        "retry_count": state.get("retry_count", 0),
                        "early_stop": state.get("early_stop", False),
                        "intent_fallback": state.get("intent_fallback", False),
                        "strong_model_calls": sum(c["tier"] == "strong" for c in llm_calls)
                        if mode == "LIVE"
                        else None,
                        "eligible_escalation_calls": sum(
                            c["task"] in {"generation", "reasoning"} for c in llm_calls
                        )
                        if mode == "LIVE"
                        else None,
                        "quality": dict.fromkeys(QUALITY),
                        "quality_source": "unmeasured",
                        "quality_context": {
                            "displayed_claims": (state.get("draft_answer") or {}).get("claims", [])
                            if state["termination_reason"] == "answered"
                            else [],
                            "contexts": state.get("relevant_documents", []),
                        },
                        "agent": state,
                    }
                    write_json(output / "raw" / f"{run_id}.json", record)
                    report["records"].append(record)
                    report["in_progress"] = None
                    write_json(output / "experiment.json", report)
                    print(
                        f"{mode} M6 {question.id}/{seed}/{variant}: {record['status']}; {len(report['records'])}/{report['expected_runs']}",
                        flush=True,
                    )
        report.update(status="finished", complete=True)
        return report, output
    except Exception as error:
        report.update(status="stopped_safely", error_type=type(error).__name__)
        raise
    finally:
        if owned and retriever is not None:
            retriever.close()
        write_json(output / "experiment.json", report)
