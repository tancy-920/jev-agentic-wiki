"""Opt-in independent support/certainty/attribution gates, never a semantic proof."""

from jev_rag.jev.provider import RUBRICS
from jev_rag.schema import Chunk

from .contracts import FidelityCitationRequest
from .efficient import EfficientAgent, EfficientJev, EfficientLLM, compact_payload
from .graph import ResearchAgent
from .verification import checks_for, validate_draft

DIMENSIONS = ("support", "certainty", "attribution")
FIDELITY_PROTOCOL = {
    "version": "source-fidelity-v1",
    "citation_dimensions": list(DIMENSIONS),
    "acceptance": "all facets SUPPORTED with original p/c >=0.85; facets are model predictions, not ground truth",
    "batch": "three independent questions per claim, one lossless catalog, <=30 items",
    "certainty": "each displayed unit preserves conjecture, observation, purpose and necessity strength on its own",
    "rendering": "paper/page citation with section explicitly unverified; never treat heuristic heading as certified",
    "repair": "one complete semantic repair for uncertain/unsupported facets, then withhold; explicit contradiction stops; no transport retries",
    "historical_defaults": "unchanged",
}

FACET_INSTRUCTIONS = {
    "support": RUBRICS["citation"][0],
    "certainty": (
        "Judge ONLY whether this entire claim preserves the evidence's epistemic and causal strength. "
        "Read each source sentence's actual scope. A source conjecture must remain a conjecture "
        "in THIS displayed claim, even when a different claim later restores a qualifier. "
        "Do not mistake a reported observation for a conjecture merely because the next sentence "
        "proposes a conjectural explanation. Design motivation does not prove necessity, sufficiency, "
        "exclusive use or a universal causal guarantee. A statement of design purpose is allowed "
        "as purpose, without asserting the mechanism is experimentally established. "
        "An assumption must not become an unconditional fact. Do not force a conjecture disclaimer "
        "onto a definition, formula, observed result or documented procedure that the source states directly. "
        "Choose NOT_SUPPORTED for any ungrounded strengthening OR misassigned uncertainty; "
        "choose CONTRADICTED only for an explicit conflict, not absence of support. "
        "Ignore unrelated source conjectures; judge the relationship this claim actually asserts."
    ),
    "attribution": (
        "Judge ONLY explicit source attribution in this claim: exact paper/method identity, "
        "section, figure, table, page, experiment and pretraining versus finetuning context. "
        "A supplied heuristic section label is not proof that an assertion occurs in that section. "
        "Require passage text to support any exact asserted section/figure attribution. "
        "Do not import unretrieved material or another request's evidence. "
        "If the claim makes no exact attribution assertion beyond a correctly identified paper, "
        "do not require invented section/figure details; SUPPORTED means no unsupported attribution "
        "assertion in this claim, not that its scientific content is certified. "
        "Choose NOT_SUPPORTED for attribution without evidence and CONTRADICTED only for explicit conflict."
    ),
}


def fidelity_requests(request):
    values = request.model_dump(mode="json")
    # Contextual and translated request variants are closed again in the new schema.
    return [FidelityCitationRequest(**values, fidelity_dimension=d) for d in DIMENSIONS]


def fidelity_payload(requests):
    payload = compact_payload(requests)
    for key, context in payload["state"]["requests"].items():
        dimension = context.get("fidelity_dimension")
        if dimension is None:
            continue
        instruction = payload["questions"][key]["instructions"]
        instruction["question"] = FACET_INSTRUCTIONS[dimension]
        criteria = {
            "SUPPORTED": "Every assertion within this facet's scope is supported; other facets are checked independently.",
            "NOT_SUPPORTED": "At least one assertion within this facet's scope is not established; absence is not contradiction.",
            "CONTRADICTED": "The evidence explicitly conflicts with an assertion within this facet's scope.",
        }
        payload["questions"][key]["criteria"] = {
            c: criteria[c] for c in payload["questions"][key]["criteria"]
        }
    return payload


def collapse_facets(predictions, threshold=0.85, *, dimensions=DIMENSIONS):
    """Select a real veto prediction; never synthesize an aggregate confidence score."""
    if (
        not dimensions
        or dimensions[0] != "support"
        or len(set(dimensions)) != len(dimensions)
        or not set(dimensions) <= set(DIMENSIONS)
        or len(predictions) != len(dimensions)
    ):
        raise ValueError("exactly the declared independent facets required")
    if any(p.choice not in {"SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"} for p in predictions):
        raise ValueError("unknown citation facet choice")
    for prediction in predictions:
        if prediction.choice == "CONTRADICTED" and all(
            v is not None and v >= threshold
            for v in (prediction.probability, prediction.confidence)
        ):
            return prediction  # An explicit confident conflict must not become a repair.
    for prediction in predictions:
        if any(v is None or v < threshold for v in (prediction.probability, prediction.confidence)):
            return prediction  # Uncertainty retains the existing repair/stop behavior.
    for choice in ("NOT_SUPPORTED",):
        for prediction in predictions:
            if prediction.choice == choice:
                return prediction
    return predictions[0]  # Support is usable only because every other facet also passed.


class FidelityLLM(EfficientLLM):
    payload_builder = staticmethod(fidelity_payload)
    protocol_version = FIDELITY_PROTOCOL["version"]


class FidelityJev(EfficientJev):
    payload_builder = staticmethod(fidelity_payload)
    protocol_version = FIDELITY_PROTOCOL["version"]


class PageCitedChunk(Chunk):
    @property
    def citation(self):
        # Exact section attribution is unavailable in the current index. Preserve
        # raw metadata for audit while declining to render it as a certified heading.
        return f"[{self.title}, 章节未核实, p. {self.page}]"


class FidelityAgent(EfficientAgent):
    protocol_version = FIDELITY_PROTOCOL["version"]

    def _evidence(self, state):
        return [PageCitedChunk(**c.model_dump()) for c in super()._evidence(state)]

    def _citation_requests(self, request):
        return fidelity_requests(request)

    def _additional_checks(self, state, draft, by_id):
        return []

    def verify(self, state):
        draft = validate_draft(state["draft_answer"], self.verification_policy)
        by_id = {c.chunk_id: c for c in self._evidence(state)}
        requests, claims, groups = [], [], []
        for paragraph in draft.claims:
            for claim, parent, _ in checks_for(paragraph, self.verification_policy):
                if not claim.citations or any(key not in by_id for key in claim.citations):
                    state["citation_results"].append(
                        {"claim": claim.text, "status": "INVALID_CITATION", "source": "application"}
                    )
                    state["termination_reason"], state["next_node"] = "citation_failure", "finish"
                    return state  # Invalid citations cannot trigger provider calls or rewriting.
                request = self._request(
                    state,
                    task="citation",
                    candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"],
                    evidence=[by_id[key] for key in dict.fromkeys(claim.citations)],
                    claim=claim,
                    **({"parent_claim": parent} if parent is not None else {}),
                )
                group = self._citation_requests(request)
                groups.append(group)
                requests.extend(group)
                claims.append(claim)
        for claim, group in self._additional_checks(state, draft, by_id):
            claims.append(claim)
            groups.append(group)
            requests.extend(group)
        if len(requests) > 30:
            state["termination_reason"], state["next_node"] = "invalid_decision", "finish"
            return state
        predictions = self._batch(state, requests) if requests else []
        if predictions is None:
            return state
        selected = []
        grouped, offset = [], 0
        for claim, group in zip(claims, groups, strict=True):
            facets = predictions[offset : offset + len(group)]
            offset += len(group)
            dimensions = [r.fidelity_dimension for r in group]
            grouped.append((dimensions, facets))
            decision = collapse_facets(facets, self.threshold, dimensions=dimensions)
            selected.append(decision)
            self._event(state, "source_fidelity", [], executed=False, evidence_ids=claim.citations)
            state["events"][-1].update(
                claim=claim.text,
                check_scope="complete_answer" if hasattr(group[0], "answer_claims") else "claim",
                facets={d: p.model_dump() for d, p in zip(dimensions, facets, strict=True)},
                veto_choice=decision.choice,
                semantic_accuracy_certified=False,
            )

        def confident(p):
            return all(v is not None and v >= self.threshold for v in (p.probability, p.confidence))

        if any(p.choice == "CONTRADICTED" and confident(p) for p in predictions):
            state["citation_results"] += [
                {
                    "claim": c.text,
                    "citation_ids": c.citations,
                    "status": p.choice,
                    "source": self.mode,
                }
                for c, p in zip(claims, selected, strict=True)
            ]
            state["next_node"] = "conflict"
            return state
        failed = [
            (c, p, grouped[i])
            for i, (c, p) in enumerate(zip(claims, selected, strict=True))
            if p.choice != "SUPPORTED" or not confident(p)
        ]
        if failed:
            state["citation_results"] += [
                {
                    "claim": c.text,
                    "citation_ids": c.citations,
                    "status": p.choice if confident(p) else "UNVERIFIED",
                    "source": self.mode,
                }
                for c, p in zip(claims, selected, strict=True)
            ]
            self._observe(state, "VERIFY", {"checked_claims": len(selected), "unsupported": True})
            if state["citation_repair_count"] < 1 and callable(
                getattr(self.generator, "generate_repair", None)
            ):
                state["citation_repair_count"] += 1
                state["retry_count"] += 1
                state["loop_count"] += 1
                state["verification_feedback"] = [
                    {
                        "verification_kind": (
                            "retrieved_consistency"
                            if getattr(groups[i][0], "retrieved_consistency", False)
                            else "answer_coverage"
                            if hasattr(groups[i][0], "answer_claims")
                            else "claim_support"
                        ),
                        "claim": c.text,
                        "status": p.choice if confident(p) else "UNVERIFIED",
                        "failed_facets": [
                            d
                            for d, f in zip(dimensions, facets, strict=True)
                            if f.choice != "SUPPORTED" or not confident(f)
                        ],
                        "instruction": "This is fallible model feedback, not ground truth. Repair the complete answer using cited evidence; preserve scope and uncertainty, every original requirement, and all factual citations. No unchecked post-verification edits.",
                    }
                    for i, (c, p) in enumerate(zip(claims, selected, strict=True))
                    if p.choice != "SUPPORTED" or not confident(p)
                    for dimensions, facets in [grouped[i]]
                ]
                state["termination_reason"], state["next_node"] = None, "generate"
                self._event(
                    state, "citation_repair", ["GENERATE"], executed=True, action="GENERATE"
                )
            else:
                state["termination_reason"] = (
                    "low_confidence"
                    if any(not confident(p) for _, p, _ in failed)
                    else "citation_failure"
                )
                state["next_node"] = "finish"
            return state
        self._citation_predictions = iter(selected)
        try:
            return ResearchAgent.verify(self, state)
        finally:
            self._citation_predictions = None
