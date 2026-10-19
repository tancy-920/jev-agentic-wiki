"""Frozen application study contract, schedule and offline acceptance metrics."""

import json
from copy import deepcopy
from time import perf_counter
from uuid import uuid4

from jev_rag.evaluation.preflight import canonical_hash
from jev_rag.io import sha256_file

SCOPE = "m6-application-acceptance"
SHARED_PAIR_POLICY = {
    "version": "fresh-paired-common-planner-retrieval-v1",
    "common": "one fresh planner and local retrieval per pair; identical candidate pool",
    "latency": "paired_full_pipeline_latency_ms combines freshly measured common elapsed with arm wall time; processing_latency_ms remains observed wall time",
    "calls": "physical common calls counted once; logical per-question calls include common stage",
    "reuse": "current pair only; new pair for every repetition; no old prediction or answer cache",
}
PILOT = [
    "请解释Transformer的编码器—解码器注意力如何连接两部分，并指出查询、键和值分别来自哪一侧。",
    "这种注意力与解码器内部的自注意力有何区别？",
]


def make_plan(workspace, config):
    from frontend.backend import application_hash
    from frontend.ranking import load_selection

    selection = load_selection(workspace)
    references = workspace / "results/processed/rag_revision/acceptance_references.json"
    packet = json.loads(references.read_text(encoding="utf-8"))
    cases = packet["cases"]
    if len(cases) != 6 or len({c["question"] for c in cases}) != 6:
        raise ValueError("six unique fixed formal scenarios required")
    if any(c["exact_previous_query_matches"] for c in cases):
        raise ValueError("formal query was already run")
    if packet["chunks_sha256"] != sha256_file(workspace / "data/processed/chunks.jsonl"):
        raise ValueError("reference corpus drift")
    formal = []
    for repeat in range(2):
        for index, case in enumerate(cases):
            variants = ["A0", "A1"] if (repeat + index) % 2 == 0 else ["A1", "A0"]
            formal.extend(
                {
                    "case_id": case["id"],
                    "question": case["question"],
                    "repeat": repeat,
                    "variant": variant,
                }
                for variant in variants
            )
    return {
        "type": "FROZEN_APPLICATION_RAG_ACCEPTANCE_PLAN",
        "scope": SCOPE,
        "application_source_sha256": application_hash(workspace),
        "config_sha256": canonical_hash(config.model_dump(mode="json")),
        "references_sha256": sha256_file(references),
        "selection_sha256": sha256_file(
            workspace / "results/processed/rag_revision/selection.json"
        ),
        "weight": selection["selected_weight"],
        "max_calls": 192,
        "question_call_cap": 20,
        "max_iterations": 5,
        "quality_threshold": 0.85,
        "repair_cap": 1,
        "transport_retries": 0,
        "style": "balanced",
        "pilot": [
            {
                "question": q,
                "variant": v,
                "case_id": f"pilot-{i}",
                "context": "new_conversation" if i == 0 else "previous_pilot_turn",
            }
            for v in ["A0", "A1"]
            for i, q in enumerate(PILOT)
        ],
        "formal": formal,
        "formal_reference_confirmation_required": True,
        "pilot_actual_answer_confirmation_required_before_formal": True,
        "page_checks": [
            "authorization",
            "new_question",
            "followup",
            "new_conversation",
            "invalid_submission_recovery",
            "end_authorization",
        ],
        "warmup": "local embedding and reranker; no provider or answer/prediction cache",
        "shared_pair_policy": SHARED_PAIR_POLICY,
        "gate": {
            "minimum_correct_per_arm": 11,
            "minimum_correct_pairs": 10,
            "all_six_scenarios_paired": True,
            "minimum_paired_latency_reduction": 0.2,
            "unsupported_delivered_facts": 0,
        },
        "stop": "phase failure closes unused authorization; no repairs inside frozen study",
        "estimated_cost_usd": None,
        "limits": "Small fresh-query validation on familiar corpus, not universal/statistically significant advantage. All failures retained; runtime answered is not source-reviewed correctness.",
    }


class SharedPair:
    """Trusted formal runner only. The UI pilot still exercises ordinary independent chats."""

    def __init__(self, question):
        self.question, self.identifier = question, str(uuid4())
        self.resolution = None
        self.hits = None
        self.retrieval_key = None
        self.traces = []
        self.common_ms = 0.0
        self.common_calls = 0
        self.call_hashes = []
        self.uses = 0

    def bind(self, generator, retriever):
        if self.uses >= 2 or generator.conversation_context:
            raise ValueError("fresh independent two-arm pair required")
        physically_first = self.uses == 0
        self.uses += 1
        original_plan = generator.plan_conversation
        pair = self

        def plan(query, policy):
            if query != pair.question:
                raise ValueError("pair question changed")
            if pair.resolution is None:
                started = perf_counter()
                before = len(generator.adapter.records)
                pair.resolution = original_plan(query, policy)
                pair.common_ms += (perf_counter() - started) * 1000
                calls = generator.adapter.records[before:]
                pair.common_calls = len(calls)
                pair.call_hashes = [canonical_hash(c) for c in calls]
            generator.conversation_resolution = pair.resolution.model_copy(deep=True)
            return generator.conversation_resolution

        generator.plan_conversation = plan

        class Retriever:
            chunks = retriever.chunks

            def __init__(self):
                self.traces = []

            def search_plan(self, queries, original_query, top_k=20):
                key = canonical_hash(
                    {"queries": queries, "original": original_query, "top_k": top_k}
                )
                if pair.hits is None:
                    before = len(retriever.traces)
                    started = perf_counter()
                    pair.hits = retriever.search_plan(queries, original_query, top_k=top_k)
                    pair.common_ms += (perf_counter() - started) * 1000
                    pair.retrieval_key = key
                    pair.traces = deepcopy(retriever.traces[before:])
                elif pair.retrieval_key != key:
                    raise ValueError("new retrieval scope cannot silently reuse paired candidates")
                self.traces.extend(
                    {
                        **deepcopy(t),
                        "shared_pair_id": pair.identifier,
                        "physical_execution_in_this_arm": physically_first,
                    }
                    for t in pair.traces
                )
                return deepcopy(pair.hits)

        return Retriever(), physically_first

    def snapshot(self, physically_first):
        return {
            "policy": SHARED_PAIR_POLICY,
            "pair_id": self.identifier,
            "common_stage_ms": self.common_ms,
            "common_physical_calls": self.common_calls,
            "common_call_sha256": self.call_hashes,
            "common_executed_in_this_arm": physically_first,
            "resolution_sha256": canonical_hash(self.resolution.model_dump(mode="json"))
            if self.resolution
            else None,
            "candidate_pool_sha256": canonical_hash(self.hits) if self.hits is not None else None,
        }


def validate_plan(workspace, config, path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan != make_plan(workspace, config):
        raise ValueError("frozen application plan/source/config/reference mismatch")
    return plan


def check_references(workspace, plan):
    directory = workspace / "results/processed/rag_revision"
    packet = json.loads((directory / "acceptance_references.json").read_text(encoding="utf-8"))
    confirmation = json.loads(
        (directory / "acceptance_reference_confirmation.json").read_text(encoding="utf-8")
    )
    expected = [
        {"id": c["id"], "facts_sha256": canonical_hash(c["proposed_facts"])}
        for c in packet["cases"]
    ]
    if (
        confirmation.get("references_sha256") != plan["references_sha256"]
        or confirmation.get("human_attested") is not True
        or not confirmation.get("approval_text", "").strip()
        or confirmation.get("cases") != expected
    ):
        raise ValueError("exact six-reference human confirmation required")
    return confirmation


def check_pilot_review(workspace, plan):
    directory = workspace / "results/processed/rag_revision"
    review = json.loads((directory / "pilot_review.json").read_text(encoding="utf-8"))
    if (
        review.get("plan_sha256") != canonical_hash(plan)
        or review.get("human_attested") is not True
        or not review.get("approval_text", "").strip()
        or len(review.get("runs", [])) != 4
        or review.get("page_checks_passed") is not True
    ):
        raise ValueError("source-reviewed pilot answers and real page checks required")
    seen = set()
    for run in review["runs"]:
        path = (
            workspace
            / "results/live/application"
            / run["session_id"]
            / (run["request_id"] + ".json")
        )
        from uuid import UUID

        UUID(run["session_id"])
        UUID(run["request_id"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        key = (raw["record"]["variant"], raw["record"]["agent"]["query"])
        if (
            run.get("raw_sha256") != sha256_file(path)
            or run.get("conclusion") != "SUPPORTED_COMPLETE"
            or raw["response"].get("status") != "answered"
            or raw["response"].get("application_source_sha256") != plan["application_source_sha256"]
            or key in seen
        ):
            raise ValueError("pilot output/source review mismatch or failed answer")
        seen.add(key)
    if seen != {(r["variant"], r["question"]) for r in plan["pilot"]}:
        raise ValueError("pilot scenarios mismatch")
    return review


def submission_guard(schedule):
    """No model action can select a different question, style, order or extra paid run."""
    position = 0
    prior = None

    def guard(request):
        nonlocal position, prior
        if position >= len(schedule):
            raise ValueError("study phase is complete; extra paid submissions denied")
        row = schedule[position]
        if (
            request.question != row["question"]
            or request.variant != row["variant"]
            or request.style != "balanced"
            or request.complex_multi_hop
            or request.hard_conflicting_evidence
        ):
            raise ValueError("submission differs from frozen schedule")
        if row.get("context") == "previous_pilot_turn":
            if request.conversation_id is None or request.conversation_id != prior:
                raise ValueError("pilot followup requires its actual previous conversation")
        elif row.get("context") == "new_conversation" and request.conversation_id is None:
            raise ValueError("pilot first turn requires a conversation ID")
        elif request.conversation_id is not None and request.conversation_id == prior:
            raise ValueError("independent scenario requires a new conversation")
        prior = request.conversation_id
        position += 1

    return guard


def evaluate_formal(plan, rows):
    """All 24 source-confirmed outcomes needed; failed pairs never become fast successes."""
    expected = {(r["case_id"], r["repeat"], r["variant"]) for r in plan["formal"]}
    actual = {(r["case_id"], r["repeat"], r["variant"]) for r in rows}
    if (
        len(rows) != 24
        or actual != expected
        or any(r.get("human_attested") is not True for r in rows)
    ):
        raise ValueError("all frozen runs and exact source confirmation required")
    if any(
        type(r.get("correct_complete")) is not bool
        or type(r.get("unsupported_delivered_facts")) is not int
        or r["unsupported_delivered_facts"] < 0
        for r in rows
    ):
        raise ValueError("explicit reviewed pass/fail and nonnegative unsupported counts required")
    counts = {
        v: sum(r["correct_complete"] is True for r in rows if r["variant"] == v)
        for v in ["A0", "A1"]
    }
    by_key = {(r["case_id"], r["repeat"], r["variant"]): r for r in rows}
    pairs = []
    for case, repeat in sorted({(r["case_id"], r["repeat"]) for r in rows}):
        a, b = (by_key[(case, repeat, v)] for v in ["A0", "A1"])
        if a["correct_complete"] is True and b["correct_complete"] is True:
            if plan.get("shared_pair_policy"):
                if (
                    not a.get("pair_id")
                    or a.get("pair_id") != b.get("pair_id")
                    or not a.get("candidate_pool_sha256")
                    or a.get("candidate_pool_sha256") != b.get("candidate_pool_sha256")
                ):
                    raise ValueError("fresh paired candidate identity required")
            elapsed_key = (
                "paired_full_pipeline_latency_ms"
                if plan.get("shared_pair_policy")
                else "processing_latency_ms"
            )
            if any(
                type(r.get(elapsed_key)) not in {float, int} or r[elapsed_key] <= 0 for r in [a, b]
            ):
                raise ValueError("actual positive elapsed times required")
            pairs.append(
                {
                    "case_id": case,
                    "repeat": repeat,
                    "A0_ms": a[elapsed_key],
                    "A1_ms": b[elapsed_key],
                }
            )
    baseline = sum(p["A0_ms"] for p in pairs)
    reduction = 1 - sum(p["A1_ms"] for p in pairs) / baseline if baseline else None
    covered = {p["case_id"] for p in pairs}
    unsupported = sum(r["unsupported_delivered_facts"] for r in rows)
    passed = (
        counts["A0"] >= 11
        and counts["A1"] >= max(11, counts["A0"])
        and len(pairs) >= 10
        and covered == {r["case_id"] for r in plan["formal"]}
        and reduction is not None
        and reduction >= 0.2
        and unsupported == 0
    )
    repeats = []
    for repeat in range(2):
        part = [p for p in pairs if p["repeat"] == repeat]
        elapsed = sum(p["A0_ms"] for p in part)
        repeats.append(
            {
                "repeat": repeat,
                "paired_count": len(part),
                "paired_latency_reduction": 1 - sum(p["A1_ms"] for p in part) / elapsed
                if elapsed
                else None,
                "counts": {
                    v: sum(
                        r["correct_complete"]
                        for r in rows
                        if r["repeat"] == repeat and r["variant"] == v
                    )
                    for v in ["A0", "A1"]
                },
            }
        )
    return {
        "passed": passed,
        "counts": counts,
        "pairs": pairs,
        "repeats": repeats,
        "paired_latency_reduction": reduction,
        "latency_metric": "composed fresh common stage plus arm elapsed"
        if plan.get("shared_pair_policy")
        else "observed processing elapsed",
        "unsupported_delivered_facts": unsupported,
        "all_failures_retained": True,
        "rows": rows,
    }
