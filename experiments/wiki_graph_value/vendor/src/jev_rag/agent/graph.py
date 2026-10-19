import hashlib
from time import perf_counter
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from jev_rag.llm.adapter import LiveGate
from jev_rag.llm.answering import AnswerPolicy
from jev_rag.retrieval import merge_search_hits
from jev_rag.schema import Chunk

from .contracts import (
    ContextualCitationRequest,
    Decision,
    DecisionRequest,
    FocusedDecisionRequest,
    ProviderError,
    TranslatedDecisionRequest,
    TranslatedFocusedDecisionRequest,
)
from .intent import INTENTS, IntentPolicy, RetrievalPlan, clarification
from .language import QueryTranslation, TranslationPolicy
from .state import AgentState
from .verification import VerificationPolicy, checks_for, validate_draft


class ResearchAgent:
    """Shared orchestration; LIVE providers require an explicit shared call-budget gate."""

    def __init__(
        self,
        retriever,
        decisions,
        generator,
        max_iterations=5,
        threshold=0.85,
        allow_direct=False,
        live_gate=None,
        relevance_uncertainty="exclude",
        intent_policy=None,
        retrieval_top_k=20,
        local_source_only=False,
        answer_policy=None,
        translation_policy=None,
        verification_policy=None,
    ):
        if not 1 <= max_iterations <= 5:
            raise ValueError("max_iterations must be between 1 and 5")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if relevance_uncertainty not in {"exclude", "stop"}:
            raise ValueError("invalid relevance uncertainty policy")
        self.relevance_uncertainty = relevance_uncertainty
        generator_policy = getattr(generator, "answer_policy", None)
        self.answer_policy = answer_policy or generator_policy or AnswerPolicy()
        generator_verification = getattr(generator, "verification_policy", None)
        self.verification_policy = (
            verification_policy or generator_verification or VerificationPolicy()
        )
        if (
            not isinstance(self.verification_policy, VerificationPolicy)
            or (
                generator_verification is not None
                and generator_verification != self.verification_policy
            )
            or (allow_direct and self.verification_policy.mode != "paragraph")
        ):
            raise ValueError("shared grounded verification policy required")
        if not isinstance(self.answer_policy, AnswerPolicy) or (
            generator_policy is not None and generator_policy != self.answer_policy
        ):
            raise ValueError("agent and generator must share a typed answer policy")
        self.intent_policy = intent_policy or IntentPolicy()
        self.translation_policy = translation_policy or TranslationPolicy()
        if not isinstance(self.translation_policy, TranslationPolicy) or (
            self.translation_policy.enabled
            and not callable(getattr(generator, "translate_query", None))
        ):
            raise ValueError("translation requires a typed policy and generator")
        if not isinstance(self.intent_policy, IntentPolicy):
            raise ValueError("typed intent policy required")
        if type(retrieval_top_k) is not int or not 1 <= retrieval_top_k <= 20:
            raise ValueError("retrieval_top_k must be between 1 and 20")
        self.retrieval_top_k = retrieval_top_k
        if type(local_source_only) is not bool or (local_source_only and allow_direct):
            raise ValueError("local-source control cannot allow direct generation")
        self.local_source_only = local_source_only
        if self.intent_policy.enabled and (
            allow_direct
            or not callable(getattr(generator, "plan", None))
            or not callable(getattr(generator, "generate_focused", None))
        ):
            raise ValueError("intent planning requires grounded generation and a typed planner")
        if decisions.source == generator.source == "MOCK":
            self.mode = "MOCK"
        elif decisions.source == generator.source == "LIVE":
            if (
                not isinstance(live_gate, LiveGate)
                or getattr(decisions, "_gate", None) is not live_gate
                or getattr(getattr(generator, "adapter", None), "_gate", None) is not live_gate
            ):
                raise ValueError(
                    "LIVE graph requires shared benchmark-live call budget; default is MOCK"
                )
            self.mode = "LIVE"
        else:
            raise ValueError("cannot mix MOCK and LIVE providers or use unknown sources")
        self.retriever, self.decisions, self.generator = retriever, decisions, generator
        self.max_iterations, self.threshold, self.allow_direct = (
            max_iterations,
            threshold,
            allow_direct,
        )
        builder = StateGraph(AgentState)
        for name in (
            "prepare_query",
            "analyze_intent",
            "plan_query",
            "route_query",
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
            builder.add_node(name, getattr(self, "route" if name == "route_query" else name))
        entry = "analyze_intent" if self.intent_policy.enabled else "route_query"
        builder.add_edge(START, "prepare_query" if self.translation_policy.enabled else entry)
        nodes = {
            name: name
            for name in (
                "analyze_intent",
                "plan_query",
                "route_query",
                "retrieve",
                "relevance",
                "sufficiency",
                "loop",
                "rewrite",
                "generate",
                "verify",
                "conflict",
                "finish",
            )
        }
        for name in (
            "prepare_query",
            "analyze_intent",
            "plan_query",
            "route_query",
            "retrieve",
            "relevance",
            "sufficiency",
            "loop",
            "rewrite",
            "generate",
            "verify",
            "conflict",
        ):
            builder.add_conditional_edges(name, lambda state: state["next_node"], nodes)
        builder.add_edge("finish", END)
        self.graph = builder.compile()

    def run(self, query: str, *, prepared_translation=None):
        if not query.strip() or len(query) > 6000:
            raise ValueError("query must contain 1-6000 characters")
        state = AgentState(
            run_id=str(uuid4()),
            mode=self.mode,
            query=query,
            retrieval_query=query,
            query_translation=None,
            rewritten_query=None,
            route=None,
            intent=None,
            intent_fallback=False,
            retrieval_plan=None,
            retrieval_requests=0,
            retrieved_candidates=0,
            retrieved_documents=[],
            relevant_documents=[],
            evidence_sufficient=None,
            reasoning_steps=[],
            tool_calls=[],
            events=[],
            iteration=0,
            max_iterations=self.max_iterations,
            loop_count=0,
            retry_count=0,
            early_stop=False,
            draft_answer=None,
            draft_history=[],
            citation_results=[],
            final_citation_results=[],
            citation_repair_count=0,
            verification_feedback=[],
            final_answer=None,
            termination_reason=None,
            next_node="route_query",
        )
        if prepared_translation is not None:
            if not self.translation_policy.enabled:
                raise ValueError("prepared translation requires enabled policy")
            translation = QueryTranslation.model_validate(prepared_translation).validate_for(
                query, self.translation_policy
            )
            if translation.source not in {self.mode, "IDENTITY"}:
                raise ValueError("prepared translation mode mismatch")
            state["query_translation"] = translation.model_dump()
        # Keep inherited tracing settings from uploading query/evidence/state to external services.
        with tracing_context(enabled=False):
            return self.graph.invoke(state, {"recursion_limit": 100})

    def prepare_query(self, state):
        started = perf_counter()
        shared = state["query_translation"] is not None
        try:
            translation = QueryTranslation.model_validate(
                state["query_translation"]
                if shared
                else self.generator.translate_query(state["query"], self.translation_policy)
            ).validate_for(state["query"], self.translation_policy)
            if translation.source not in {self.mode, "IDENTITY"}:
                raise ValueError("translation mode mismatch")
            state["query_translation"] = translation.model_dump()
            state["retrieval_query"] = translation.english_query
            self._event(
                state, "prepare_query", [], executed=True, latency=(perf_counter() - started) * 1000
            )
            state["events"][-1].update(translation.summarize(), shared_preprocessing=shared)
            state["next_node"] = "analyze_intent" if self.intent_policy.enabled else "route_query"
        except (ProviderError, ValueError, ValidationError):
            self._event(
                state,
                "prepare_query",
                [],
                error="translation_failure",
                latency=(perf_counter() - started) * 1000,
            )
            state["termination_reason"], state["next_node"] = "translation_failure", "finish"
        return state

    def _event(
        self,
        state,
        node,
        candidates,
        decision=None,
        executed=False,
        latency=None,
        error=None,
        action=None,
        evidence_ids=None,
    ):
        event = {
            "run_id": state["run_id"],
            "step_id": len(state["events"]) + 1,
            "node": node,
            "input_summary": {
                "query_chars": len(state["query"]),
                "evidence_count": len(evidence_ids)
                if evidence_ids is not None
                else len(state["relevant_documents"]),
                "evidence_chunk_ids": evidence_ids,
                "iteration": state["iteration"],
            },
            "candidate_actions": list(candidates),
            "selected_action": decision.choice if decision else action,
            "probability": decision.probability if decision else None,
            "confidence": decision.confidence if decision else None,
            "latency_ms": latency,
            "source": decision.source if decision else "application",
            "executed": executed,
            "error": error,
            "mode": state["mode"],
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
        }
        state["events"] = [*state["events"], event]

    def _decide(
        self,
        state,
        task,
        candidates,
        evidence=None,
        claim=None,
        supplied=None,
        uncertain_fallback=None,
        parent_claim=None,
    ):
        started = perf_counter()
        request = self._request(
            state,
            task=task,
            candidates=list(candidates),
            evidence=evidence or [],
            claim=claim,
            **({"parent_claim": parent_claim} if parent_claim is not None else {}),
        )
        decision = None
        try:
            raw = self.decisions.decide(request) if supplied is None else supplied
            decision = Decision.model_validate(raw)
            if decision.source != self.decisions.source or decision.choice not in candidates:
                raise ValueError("invalid decision source or choice")
            if any(
                v is None or v < self.threshold for v in (decision.probability, decision.confidence)
            ):
                self._event(
                    state,
                    task,
                    candidates,
                    decision,
                    False,
                    (perf_counter() - started) * 1000,
                    "uncertain_decision",
                    evidence_ids=[c.chunk_id for c in request.evidence],
                )
                if uncertain_fallback is not None:
                    state["events"][-1]["application_fallback"] = "EXCLUDE_UNCERTAIN_PASSAGE"
                    return uncertain_fallback
                state["termination_reason"], state["next_node"] = "low_confidence", "finish"
                return None
        except (ProviderError, ValidationError, ValueError) as error:
            reason = "provider_failure" if isinstance(error, ProviderError) else "invalid_decision"
            self._event(
                state,
                task,
                candidates,
                None,
                False,
                (perf_counter() - started) * 1000,
                reason,
                evidence_ids=[c.chunk_id for c in request.evidence],
            )
            state["termination_reason"], state["next_node"] = reason, "finish"
            return None
        self._event(
            state,
            task,
            candidates,
            decision,
            True,
            (perf_counter() - started) * 1000,
            evidence_ids=[c.chunk_id for c in request.evidence],
        )
        return decision.choice

    def _request(self, state, **kwargs):
        fixed = dict(
            query=state["query"],
            iteration=state["iteration"],
            evidence_sufficient=state["evidence_sufficient"],
        )
        if "parent_claim" in kwargs:
            return ContextualCitationRequest(
                **fixed,
                **kwargs,
                translated_query=state["retrieval_query"] if state["query_translation"] else None,
                focus=RetrievalPlan.model_validate(state["retrieval_plan"])
                if state["retrieval_plan"]
                else None,
            )
        if state["retrieval_plan"] is not None:
            cls = (
                TranslatedFocusedDecisionRequest
                if state["query_translation"]
                else FocusedDecisionRequest
            )
            return cls(
                **fixed,
                **kwargs,
                focus=RetrievalPlan.model_validate(state["retrieval_plan"]),
                **(
                    {"translated_query": state["retrieval_query"]}
                    if state["query_translation"]
                    else {}
                ),
            )
        if state["query_translation"]:
            return TranslatedDecisionRequest(
                **fixed, **kwargs, translated_query=state["retrieval_query"]
            )
        return DecisionRequest(**fixed, **kwargs)

    def _observe(self, state, action, summary):
        state["tool_calls"] = [
            *state["tool_calls"],
            {
                "action": action,
                "iteration": state["iteration"],
                "source": "application",
                "mode": state["mode"],
                "executed": True,
            },
        ]
        state["reasoning_steps"] = [
            *state["reasoning_steps"],
            {"action": action, "observation": summary, "iteration": state["iteration"]},
        ]

    def _evidence(self, state):
        return [Chunk.model_validate(c) for c in state["relevant_documents"]]

    def analyze_intent(self, state):
        choice = self._decide(state, "intent", INTENTS)
        if (
            choice is None
            and state["termination_reason"] == "low_confidence"
            and self.intent_policy.uncertainty == "original_query"
        ):
            # Reject the uncertain interpretation, not the original user query.
            # No guessed intent, planner, rewrite, extra source or direct answer.
            state["intent_fallback"] = True
            state["events"][-1]["application_fallback"] = (
                "ORIGINAL_TRANSLATED_QUERY_LOCAL_RETRIEVAL"
                if state["query_translation"]
                else "ORIGINAL_QUERY_LOCAL_RETRIEVAL"
            )
            state["termination_reason"], state["next_node"] = None, "route_query"
            return state
        if choice == "CLARIFY" or (
            choice is None and state["termination_reason"] == "low_confidence"
        ):
            state["termination_reason"] = "clarification_required" if choice else "intent_uncertain"
            state["final_answer"] = f"{self.mode} — " + clarification(
                state["query"], self.answer_policy.language
            )
            state["next_node"] = "finish"
        elif choice is not None:
            state["intent"], state["next_node"] = choice, "plan_query"
        return state

    def _plan(self, state, *, rewriting=False):
        started = perf_counter()
        previous = RetrievalPlan.model_validate(state["retrieval_plan"]) if rewriting else None
        plan = RetrievalPlan.model_validate(
            self.generator.plan(
                state["query"],
                state["intent"],
                self.intent_policy,
                evidence=self._evidence(state),
                iteration=state["iteration"] if rewriting else None,
                previous_plan=previous,
                **(
                    {"translated_query": state["retrieval_query"]}
                    if state["query_translation"]
                    else {}
                ),
            )
        ).validate_for(state["query"], state["intent"], self.intent_policy)
        state["retrieval_plan"] = plan.model_dump()
        self._event(
            state,
            "rewrite" if rewriting else "plan_query",
            [],
            executed=True,
            latency=(perf_counter() - started) * 1000,
        )
        state["events"][-1].update(plan.summary())
        state["events"][-1].update(source=self.generator.source, provider_role="llm_planner")

    def plan_query(self, state):
        try:
            self._plan(state)
            state["next_node"] = "route_query"
        except (ProviderError, ValueError, ValidationError):
            self._event(state, "plan_query", [], error="planning_failure")
            state["termination_reason"], state["next_node"] = "planning_failure", "finish"
        return state

    def route(self, state):
        if self.intent_policy.enabled and not state["intent_fallback"]:
            # Only a local paper source is currently configured. Do not pay for a
            # second classifier of identical actions: D1 execution follows the
            # validated query count, and multi-query now has distinct semantics.
            plan = RetrievalPlan.model_validate(state["retrieval_plan"])
            state["route"] = "MULTI_HOP_RAG" if len(plan.queries) > 1 else "LOCAL_RAG"
            self._event(state, "routing", [state["route"]], executed=True, action=state["route"])
            state["events"][-1]["routing_policy"] = "configured-local-source-plan-query-count-v1"
            state["next_node"] = "retrieve"
            return state
        if self.local_source_only or state["intent_fallback"]:
            state["route"], state["next_node"] = "LOCAL_RAG", "retrieve"
            self._event(state, "routing", ["LOCAL_RAG"], executed=True, action="LOCAL_RAG")
            state["events"][-1]["routing_policy"] = "configured-local-source-plan-query-count-v1"
            return state
        candidates = ["LOCAL_RAG", "MULTI_HOP_RAG"]
        if self.allow_direct:
            candidates.append("DIRECT_LLM")
        choice = self._decide(state, "routing", candidates)
        if choice:
            state["route"] = choice
            state["next_node"] = "generate" if choice == "DIRECT_LLM" else "retrieve"
        return state

    def retrieve(self, state):
        if state["iteration"] >= state["max_iterations"]:
            state["termination_reason"], state["next_node"] = "max_iterations", "finish"
            return state
        state["iteration"] += 1
        try:
            if self.intent_policy.enabled and not state["intent_fallback"]:
                queries = RetrievalPlan.model_validate(state["retrieval_plan"]).queries
                state["retrieval_requests"] += len(queries)
                if callable(getattr(self.retriever, "search_plan", None)):
                    hits = self.retriever.search_plan(
                        queries, state["retrieval_query"], top_k=self.retrieval_top_k
                    )
                else:
                    hits = merge_search_hits(
                        [self.retriever.search(q) for q in queries], top_k=self.retrieval_top_k
                    )
            else:
                queries = [state["rewritten_query"] or state["retrieval_query"]]
                state["retrieval_requests"] += 1
                hits = self.retriever.search(queries[0])
            state["retrieved_candidates"] += len(hits)
            if self.intent_policy.enabled and len(hits) > self.retrieval_top_k:
                raise ValueError("retriever exceeded the global passage budget")
            chunks = [Chunk.model_validate(hit["chunk"]).model_dump() for hit in hits]
            state["retrieved_documents"] = list({c["chunk_id"]: c for c in chunks}.values())
            state["relevant_documents"], state["evidence_sufficient"] = [], None
            state["draft_answer"] = None
            state["verification_feedback"] = []
            self._observe(
                state, "RETRIEVE", {"retrieved_chunks": len(chunks), "query_count": len(queries)}
            )
            self._event(state, "retrieve", ["RETRIEVE"], executed=True, action="RETRIEVE")
            state["events"][-1].update(
                query_count=len(queries), retrieved_chunk_count=len(state["retrieved_documents"])
            )
            if state["intent_fallback"]:
                state["events"][-1]["query_sha256s"] = [
                    hashlib.sha256(q.encode()).hexdigest() for q in queries
                ]
            state["next_node"] = "relevance"
        except (
            ProviderError,
            ValidationError,
            ValueError,
            OSError,
            ResponseHandlingException,
            UnexpectedResponse,
            KeyError,
            TypeError,
        ):
            self._event(
                state, "retrieve", ["RETRIEVE"], error="retrieval_failure", action="RETRIEVE"
            )
            state["termination_reason"], state["next_node"] = "retrieval_failure", "finish"
        return state

    def relevance(self, state):
        accepted = []
        batch = None
        if state["retrieved_documents"] and hasattr(self.decisions, "decide_many"):
            try:
                batch = self.decisions.decide_many(
                    [
                        self._request(
                            state,
                            task="relevance",
                            candidates=["RELEVANT", "IRRELEVANT"],
                            evidence=[Chunk.model_validate(item)],
                        )
                        for item in state["retrieved_documents"]
                    ]
                )
                if (
                    not isinstance(batch, (list, tuple))
                    or len(batch) != len(state["retrieved_documents"])
                    or any(d is None for d in batch)
                ):
                    raise ValueError("invalid relevance batch")
            except (ProviderError, ValueError, ValidationError):
                self._event(state, "relevance", ["RELEVANT", "IRRELEVANT"], error="batch_failure")
                state["termination_reason"], state["next_node"] = "provider_failure", "finish"
                return state
        for index, item in enumerate(state["retrieved_documents"]):
            chunk = Chunk.model_validate(item)
            choice = self._decide(
                state,
                "relevance",
                ["RELEVANT", "IRRELEVANT"],
                [chunk],
                supplied=batch[index] if batch is not None else None,
                uncertain_fallback="IRRELEVANT"
                if self.relevance_uncertainty == "exclude"
                else None,
            )
            if choice is None:
                return state
            if choice == "RELEVANT":
                accepted.append(item)
        state["relevant_documents"] = accepted
        self._event(state, "relevance_summary", [], executed=True)
        state["events"][-1].update(
            accepted_count=len(accepted),
            rejected_count=len(state["retrieved_documents"]) - len(accepted),
        )
        state["next_node"] = "sufficiency"
        return state

    def sufficiency(self, state):
        evidence = self._evidence(state)
        if not evidence:
            state["evidence_sufficient"] = False
            self._event(state, "evidence", ["INSUFFICIENT"], executed=True, action="INSUFFICIENT")
        else:
            choice = self._decide(state, "evidence", ["SUFFICIENT", "INSUFFICIENT"], evidence)
            if choice is None:
                return state
            state["evidence_sufficient"] = choice == "SUFFICIENT"
        state["next_node"] = "loop"
        return state

    def loop(self, state):
        if not state["evidence_sufficient"] and state["iteration"] >= state["max_iterations"]:
            state["termination_reason"], state["next_node"] = "max_iterations", "finish"
            return state
        candidates = (
            ["STOP", "ESCALATE"]
            if state["evidence_sufficient"] or state["intent_fallback"]
            else ["CONTINUE", "RETRY", "STOP", "ESCALATE"]
        )
        choice = self._decide(state, "loop", candidates, self._evidence(state))
        if choice is None:
            return state
        if choice == "STOP" and state["evidence_sufficient"]:
            state["early_stop"] = state["iteration"] < state["max_iterations"]
            state["next_node"] = "generate"
        elif choice in {"CONTINUE", "RETRY"}:
            state["loop_count"] += 1
            state["retry_count"] += choice == "RETRY"
            state["next_node"] = "rewrite"
        else:
            state["termination_reason"] = (
                "escalated" if choice == "ESCALATE" else "insufficient_evidence"
            )
            state["next_node"] = "finish"
        return state

    def rewrite(self, state):
        if state["intent_fallback"]:
            state["termination_reason"], state["next_node"] = "insufficient_evidence", "finish"
            return state
        try:
            if self.intent_policy.enabled:
                self._plan(state, rewriting=True)
                self._observe(
                    state, "REWRITE_QUERY", {"query_count": len(state["retrieval_plan"]["queries"])}
                )
                state["next_node"] = "route_query"
                return state
            rewrite = self.generator.rewrite(
                state["query"], self._evidence(state), state["iteration"]
            )
            if not isinstance(rewrite, str) or not rewrite.strip() or len(rewrite) > 6000:
                raise ValueError("invalid rewrite")
            state["rewritten_query"] = rewrite
            self._observe(state, "REWRITE_QUERY", {"rewritten_query_chars": len(rewrite)})
            self._event(state, "rewrite", ["REWRITE_QUERY"], executed=True, action="REWRITE_QUERY")
            state["next_node"] = "retrieve"
        except (ProviderError, ValueError):
            self._event(state, "rewrite", ["REWRITE_QUERY"], error="rewrite_failure")
            state["termination_reason"], state["next_node"] = "rewrite_failure", "finish"
        return state

    def generate(self, state):
        try:
            draft = validate_draft(
                self.generator.generate_repair(
                    state["query"], self._evidence(state), state["verification_feedback"]
                )
                if state["verification_feedback"]
                else self.generator.generate_focused(
                    state["query"],
                    self._evidence(state),
                    RetrievalPlan.model_validate(state["retrieval_plan"]),
                )
                if self.intent_policy.enabled and not state["intent_fallback"]
                else self.generator.generate(state["query"], self._evidence(state)),
                self.verification_policy,
            )
            state["draft_answer"] = draft.model_dump()
            state["draft_history"] = [*state["draft_history"], draft.model_dump()]
            self._observe(state, "GENERATE", {"claims": len(draft.claims)})
            self._event(state, "generate", ["GENERATE"], executed=True, action="GENERATE")
            state["next_node"] = "verify"
        except (ProviderError, ValueError, ValidationError):
            state["termination_reason"], state["next_node"] = "generation_failure", "finish"
        return state

    def verify(self, state):
        draft, evidence = (
            validate_draft(state["draft_answer"], self.verification_policy),
            self._evidence(state),
        )
        by_id = {c.chunk_id: c for c in evidence}
        supported, results, unsupported = [], [], False
        direct = state["route"] == "DIRECT_LLM"
        check_items = [
            (paragraph_index, paragraph, claim, parent, span_index)
            for paragraph_index, paragraph in enumerate(draft.claims)
            for claim, parent, span_index in checks_for(paragraph, self.verification_policy)
        ]
        batch = None
        batch_method = getattr(self.decisions, "decide_citations", None)
        if (getattr(self, "batch_citations", False) and callable(batch_method)
            and len(check_items) >= 1 and not direct
            and all(item[2].citations and all(k in by_id for k in item[2].citations)
                    for item in check_items)):
            started = perf_counter()
            try:
                requests = [
                    self._request(state, task="citation",
                                  candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"],
                                  evidence=[by_id[k] for k in dict.fromkeys(claim.citations)],
                                  claim=claim)
                    for _, _, claim, _, _ in check_items
                ]
                hook = getattr(self, "_batch_source_and_coverage", None)
                batch = hook(state, requests) if callable(hook) else batch_method(requests)
                if not isinstance(batch, (list, tuple)) or len(batch) != len(check_items) or any(
                    d is None for d in batch
                ):
                    raise ValueError("invalid source batch")
            except (ProviderError, ValueError):
                self._event(state, "citation_batch", [], error="batch_failure",
                            latency=(perf_counter() - started) * 1000)
                state["termination_reason"], state["next_node"] = "provider_failure", "finish"
                return state
            self._event(state, "citation_batch", [], executed=True,
                        latency=(perf_counter() - started) * 1000)
        for check_index, (paragraph_index, paragraph, claim, parent, span_index) in enumerate(check_items):
            context = (
                {
                    "paragraph_index": paragraph_index,
                    "scope": "span" if parent else "paragraph",
                    "span_index": span_index,
                }
                if self.verification_policy.mode != "paragraph"
                else {}
            )
            if direct and not claim.citations:
                supported.append(claim.text)
                results.append(
                    {"claim": claim.text, "status": "UNCITED_DIRECT", "source": "application"}
                )
                continue
            if not claim.citations or any(key not in by_id for key in claim.citations):
                results.append(
                    {
                        "claim": claim.text,
                        "status": "INVALID_CITATION",
                        "source": "application",
                        **context,
                    }
                )
                unsupported = True
                continue
            cited = [by_id[key] for key in dict.fromkeys(claim.citations)]
            choice = self._decide(
                state,
                "citation",
                ["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"],
                cited,
                claim,
                parent_claim=parent,
                supplied=batch[check_index] if batch is not None else None,
            )
            if choice is None:
                results.append(
                    {
                        "claim": claim.text,
                        "citation_ids": claim.citations,
                        "status": "UNVERIFIED",
                        "source": "application",
                        **context,
                    }
                )
                state["citation_results"] = [*state["citation_results"], *results]
                if (
                    state["termination_reason"] == "low_confidence"
                    and state["citation_repair_count"] < 1
                    and callable(getattr(self.generator, "generate_repair", None))
                ):
                    state["citation_repair_count"] += 1
                    state["retry_count"] += 1
                    state["loop_count"] += 1
                    state["verification_feedback"] = [
                        {
                            "claim": claim.text,
                            "status": "UNVERIFIED",
                            "instruction": "Evidence support was uncertain; preserve qualifiers and ground the revised answer without asserting unsupported conclusions.",
                        }
                    ]
                    state["termination_reason"], state["next_node"] = None, "generate"
                    self._event(
                        state, "citation_repair", ["GENERATE"], executed=True, action="GENERATE"
                    )
                return state
            results.append(
                {
                    "claim": claim.text,
                    "citation_ids": claim.citations,
                    "status": choice,
                    "source": self.decisions.source,
                    **context,
                }
            )
            if choice == "CONTRADICTED":
                state["citation_results"], state["next_node"] = (
                    [*state["citation_results"], *results],
                    "conflict",
                )
                return state
            if choice == "SUPPORTED":
                if parent is None:
                    supported.append(claim.text + " " + " ".join(c.citation for c in cited))
            else:
                unsupported = True
        state["citation_results"] = [*state["citation_results"], *results]
        self._observe(state, "VERIFY", {"checked_claims": len(results), "unsupported": unsupported})
        if unsupported:
            if (
                state["iteration"] < state["max_iterations"]
                and not direct
                and not state["intent_fallback"]
            ):
                state["retry_count"] += 1
                state["loop_count"] += 1
                state["next_node"] = "rewrite"
            else:
                state["termination_reason"], state["next_node"] = "citation_failure", "finish"
        else:
            state["final_citation_results"] = results
            label = self.answer_policy.heading(self.mode)
            state["final_answer"] = label + "\n\n" + "\n\n".join(supported)
            state["termination_reason"] = "direct_uncited" if direct else "answered"
            state["next_node"] = "finish"
        return state

    def conflict(self, state):
        self._event(state, "conflict", ["STOP"], executed=True, action="STOP")
        state["termination_reason"], state["next_node"] = "conflicting_evidence", "finish"
        return state

    def finish(self, state):
        if state["final_answer"] is None:
            state["final_answer"] = self.answer_policy.withheld(
                self.mode, state["termination_reason"]
            )
        self._event(state, "finish", ["STOP"], executed=True, action="STOP")
        return state
