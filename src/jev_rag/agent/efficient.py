"""Opt-in evidence-control v1. Shared policy; historical/default graph is unchanged."""

import json
from functools import wraps
from time import perf_counter

from pydantic import ValidationError

from jev_rag.jev.provider import JevDecisions, choice_payload, validate_request
from jev_rag.llm.decisions import LLMDecisions

from .contracts import CoverageRequest, Decision, ProviderError
from .graph import ResearchAgent
from .intent import RetrievalPlan
from .verification import checks_for, validate_draft

PROTOCOL = {
    "version": "evidence-control-v3",
    "generation": "shared existing question-first-units-v4, <=4 cited units of <=240 chars; length is not truth",
    "translation": "opt-in token/tokens grammatical inflection; identifiers and numbers remain literal",
    "jev_tasks": ["relevance", "coverage"],
    "coverage": "up to six exact user constraint spans AND independent full-original-query gate; planner evidence needs are retrieval hints, not added user obligations",
    "shared_preprocessing": "both arms reuse initial translation, LLM intent and validated plan",
    "loop": "covered->generate; conflict->withhold; missing->rewrite within unchanged limit",
    "citation": "independent batch; unchanged per-item threshold, repair and release rules",
    "payload": "shared policy and lossless evidence catalog; no evidence truncation",
    "threshold": 0.85,
    "uncertain_intent": "unchanged original-query fallback; do not adopt guessed intent",
    "source_metadata": "heading_heuristic_v1 is not proof of exact section attribution",
}


def compact_payload(requests):
    """Reference identical evidence once; preserve every evidence field and byte of text."""
    payload = choice_payload(requests)
    catalog = {}
    policies = set()
    for key, context in payload["state"]["requests"].items():
        evidence = context.pop("evidence")
        ids = []
        for chunk in evidence:
            identifier = chunk["chunk_id"]
            if identifier in catalog and catalog[identifier] != chunk:
                raise ValueError("conflicting evidence with identical chunk ID")
            catalog[identifier] = chunk
            ids.append(identifier)
        context["evidence_ids"] = ids
        instruction = payload["questions"][key]["instructions"]
        policies.add(instruction.pop("policy"))
        instruction["context"] = (
            f"Evaluate only state.requests.{key}. Resolve its evidence_ids ONLY through "
            "state.evidence_by_id; other catalog passages are not evidence for this request. "
            "Follow state.shared_policy. Metadata marked heading_heuristic_v1 is fallible; "
            "verify any asserted exact section/figure attribution against passage text."
        )
    if len(policies) != 1:
        raise ValueError("one shared evidence policy required")
    payload["state"].update(evidence_by_id=catalog, shared_policy=policies.pop())
    return payload


class EfficientBatches:
    payload_builder = staticmethod(compact_payload)
    protocol_version = PROTOCOL["version"]

    def _send(self, requests):
        # Keep task/batch provenance even when transport fails before LLM parsing.
        before = len(self.records)
        payload = self.payload_builder(requests)
        try:
            return super()._send(requests)
        finally:
            if len(self.records) > before:
                self.records[-1].update(
                    decision_task=requests[0].task,
                    decision_count=len(requests),
                    payload_version=self.protocol_version,
                    payload_bytes=len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                )

    def decide_batch(self, requests):
        if not requests:
            return []
        requests = [validate_request(r) for r in requests]
        task = requests[0].task
        limit = {"coverage": 7, "citation": 32}.get(task)
        if limit is None or len(requests) > limit or any(r.task != task for r in requests):
            raise ValueError("bounded homogeneous coverage/citation batch required")
        # Each item has its own candidates/evidence/claim; all answer IDs are validated.
        return self._send(requests)


class EfficientJev(EfficientBatches, JevDecisions):
    pass


class EfficientLLM(EfficientBatches, LLMDecisions):
    pass


class EfficientDecisions:
    """Only decision provider differs between E0 (LLM) and E2 (Jev relevance/coverage)."""

    def __init__(self, llm, jev, variant):
        if variant not in {"E0", "E2"} or llm.source != jev.source:
            raise ValueError("known efficient variant and identical modes required")
        self.llm, self.jev, self.variant = llm, jev, variant
        self.source, self._gate = llm.source, getattr(llm, "_gate", None)
        if self.source not in {"MOCK", "LIVE"} or (
            self.source == "LIVE"
            and (self._gate is None or self._gate is not getattr(jev, "_gate", None))
        ):
            raise ValueError("shared explicit live gate required")
        self.dispatches = []

    def _provider(self, task, count):
        if task not in {"intent", "routing", "relevance", "evidence", "coverage", "citation"}:
            raise ValueError("unknown efficient decision task")
        selected = "jev" if self.variant == "E2" and task in PROTOCOL["jev_tasks"] else "llm"
        self.dispatches.append({"task": task, "provider": selected, "count": count})
        return self.jev if selected == "jev" else self.llm

    def decide(self, request):
        return self._provider(request.task, 1).decide(request)

    def decide_many(self, requests):
        if not requests or any(r.task != "relevance" for r in requests):
            raise ValueError("independent relevance batch required")
        return self._provider("relevance", len(requests)).decide_many(requests)

    def decide_batch(self, requests):
        if not requests or any(r.task != requests[0].task for r in requests):
            raise ValueError("homogeneous batch required")
        return self._provider(requests[0].task, len(requests)).decide_batch(requests)


def time_node(agent, name, method):
    @wraps(method)
    def measured(state):
        start = perf_counter()
        result = method(state)
        agent._event(
            result, "node_timing", [], executed=True, latency=(perf_counter() - start) * 1000
        )
        result["events"][-1].update(
            timed_node=name,
            protocol=getattr(agent, "protocol_version", PROTOCOL["version"]),
            status=result["termination_reason"],
        )
        return result

    return measured


class EfficientAgent(ResearchAgent):
    def __init__(self, *args, prepared_control=None, **kwargs):
        if kwargs.get("threshold", 0.85) != PROTOCOL["threshold"]:
            raise ValueError("v1 preserves the frozen 0.85 threshold")
        self._citation_predictions = None
        self.prepared_control = prepared_control
        for name in (
            "prepare_query",
            "analyze_intent",
            "plan_query",
            "route",
            "retrieve",
            "relevance",
            "sufficiency",
            "loop",
            "rewrite",
            "generate",
            "verify",
            "conflict",
            "finish",
        ):
            setattr(self, name, time_node(self, name, getattr(self, name)))
        super().__init__(*args, **kwargs)

    def _plan(self, state, *, rewriting=False):
        if rewriting or self.prepared_control is None:
            return super()._plan(state, rewriting=rewriting)
        control = self.prepared_control
        if (
            control["query"] != state["query"]
            or control["retrieval_query"] != state["retrieval_query"]
        ):
            raise ValueError("shared control source mismatch")
        plan = RetrievalPlan.model_validate(control["plan"]).validate_for(
            state["query"], state["intent"], self.intent_policy
        )
        state["retrieval_plan"] = plan.model_dump()
        self._event(state, "plan_query", [], executed=True)
        state["events"][-1].update(
            plan.summary(), shared_preprocessing=True, provider_role="llm_planner"
        )

    def _batch(self, state, requests):
        start = perf_counter()
        try:
            raw = self.decisions.decide_batch(requests)
            if not isinstance(raw, (list, tuple)) or len(raw) != len(requests):
                raise ValueError("batch result count mismatch")
            predictions = [Decision.model_validate(d) for d in raw]
            if any(
                d.source != self.mode or d.choice not in r.candidates
                for d, r in zip(predictions, requests, strict=True)
            ):
                raise ValueError("batch source or choice mismatch")
        except (ProviderError, ValueError, ValidationError) as error:
            reason = "provider_failure" if isinstance(error, ProviderError) else "invalid_decision"
            state["termination_reason"], state["next_node"] = reason, "finish"
            self._event(
                state, "decision_batch", [], error=reason, latency=(perf_counter() - start) * 1000
            )
            state["events"][-1].update(task=requests[0].task, decision_count=len(requests))
            return None
        self._event(
            state, "decision_batch", [], executed=True, latency=(perf_counter() - start) * 1000
        )
        state["events"][-1].update(task=requests[0].task, decision_count=len(requests))
        return predictions

    def _coverage_scopes(self, state):
        plan = (
            RetrievalPlan.model_validate(state["retrieval_plan"])
            if state["retrieval_plan"]
            else None
        )
        # Full-query check remains mandatory even if the planner drops an essential requirement.
        # Only literal user requirements can become mandatory gates. Do not reward or
        # penalize expansion by the fallible planner. Above six spans use full-query
        # checking only, without silently discarding original requirements.
        constraints = list(dict.fromkeys(plan.user_constraints)) if plan else []
        scopes = [("requirement", need) for need in constraints] if len(constraints) <= 6 else []
        scopes.append(("original_query", state["query"]))
        return scopes

    def sufficiency(self, state):
        evidence = self._evidence(state)
        if not evidence:
            state["evidence_sufficient"] = False
            self._event(state, "coverage", ["MISSING"], executed=True, action="MISSING")
            state["next_node"] = "loop"
            return state
        scopes = self._coverage_scopes(state)
        requests = [
            CoverageRequest(
                query=state["query"],
                requirement=need,
                scope=scope,
                translated_query=state["retrieval_query"] if state["query_translation"] else None,
                candidates=["COVERED", "MISSING", "CONFLICT"],
                evidence=evidence,
                iteration=state["iteration"],
                evidence_sufficient=None,
            )
            for scope, need in scopes
        ]
        predictions = self._batch(state, requests)
        if predictions is None:
            return state
        choices = []
        for request, decision in zip(requests, predictions, strict=True):
            confident = all(
                v is not None and v >= self.threshold
                for v in (decision.probability, decision.confidence)
            )
            self._event(
                state,
                "coverage",
                request.candidates,
                decision,
                executed=confident,
                error=None if confident else "uncertain_decision",
                evidence_ids=[c.chunk_id for c in evidence],
            )
            state["events"][-1].update(
                requirement=request.requirement, coverage_scope=request.scope
            )
            choices.append(decision.choice)
        # Validate every item before using any prediction. Uncertain coverage stops, never guessed.
        if any(
            v is None or v < self.threshold
            for d in predictions
            for v in (d.probability, d.confidence)
        ):
            state["termination_reason"], state["next_node"] = "low_confidence", "finish"
        elif "CONFLICT" in choices:
            state["termination_reason"], state["next_node"] = "evidence_conflict", "finish"
        else:
            state["evidence_sufficient"] = all(c == "COVERED" for c in choices)
            state["next_node"] = "loop"
        return state

    def loop(self, state):
        if state["evidence_sufficient"]:
            action, following, reason = "STOP", "generate", None
            state["early_stop"] = state["iteration"] < state["max_iterations"]
        elif state["iteration"] >= state["max_iterations"]:
            action, following, reason = "STOP", "finish", "max_iterations"
        elif state["intent_fallback"]:
            action, following, reason = "STOP", "finish", "insufficient_evidence"
        else:
            action, following, reason = "RETRY", "rewrite", None
            state["loop_count"] += 1
            state["retry_count"] += 1
        self._event(state, "loop", [action], executed=True, action=action)
        state["events"][-1]["policy"] = PROTOCOL["version"]
        state["next_node"], state["termination_reason"] = following, reason
        return state

    def verify(self, state):
        draft = validate_draft(state["draft_answer"], self.verification_policy)
        by_id = {c.chunk_id: c for c in self._evidence(state)}
        requests = []
        for paragraph in draft.claims:
            for claim, parent, _ in checks_for(paragraph, self.verification_policy):
                if not claim.citations or any(key not in by_id for key in claim.citations):
                    continue  # Historical validator handles invalid IDs; never send invented evidence.
                request = self._request(
                    state,
                    task="citation",
                    candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"],
                    evidence=[by_id[key] for key in dict.fromkeys(claim.citations)],
                    claim=claim,
                    **({"parent_claim": parent} if parent is not None else {}),
                )
                requests.append(request)
        predictions = self._batch(state, requests) if requests else []
        if predictions is None:
            return state
        self._citation_predictions = iter(predictions)
        try:
            # Keep existing all-or-nothing release, uncertainty repair and contradiction behavior.
            return super().verify(state)
        finally:
            self._citation_predictions = None

    def _decide(self, state, task, candidates, *args, **kwargs):
        if task == "intent" and self.prepared_control is not None:
            if (
                self.prepared_control["query"] != state["query"]
                or self.prepared_control["retrieval_query"] != state["retrieval_query"]
            ):
                raise ValueError("shared intent source mismatch")
            kwargs["supplied"] = self.prepared_control["intent_decision"]
        if task == "citation" and self._citation_predictions is not None:
            kwargs["supplied"] = next(self._citation_predictions)
        return super()._decide(state, task, candidates, *args, **kwargs)
