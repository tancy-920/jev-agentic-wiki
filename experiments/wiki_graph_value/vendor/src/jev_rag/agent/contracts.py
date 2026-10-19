from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from jev_rag.schema import Chunk

from .intent import IntentPolicy, RetrievalPlan


class ProviderError(RuntimeError):
    """Transport/provider failure; messages must not contain secrets or prompts."""


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: str
    probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    source: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000)
    citations: list[str] = Field(default_factory=list, max_length=20)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(min_length=1, max_length=30)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["intent", "routing", "relevance", "evidence", "loop", "citation", "answer_budget"]
    query: str
    candidates: list[str]
    evidence: list[Chunk] = Field(default_factory=list)
    iteration: int = 0
    evidence_sufficient: bool | None = None
    claim: Claim | None = None


class FocusedDecisionRequest(DecisionRequest):
    """New graph context without altering hashes of historical base requests."""

    focus: RetrievalPlan


class TranslatedDecisionRequest(DecisionRequest):
    translated_query: str = Field(min_length=1, max_length=6000)


class TranslatedFocusedDecisionRequest(FocusedDecisionRequest):
    translated_query: str = Field(min_length=1, max_length=6000)


class ContextualCitationRequest(DecisionRequest):
    """Parent is linguistic context, not evidence or an authorization."""

    parent_claim: Claim
    translated_query: str | None = Field(default=None, min_length=1, max_length=6000)
    focus: RetrievalPlan | None = None


class CoverageRequest(DecisionRequest):
    """Opt-in v2 coverage; the original query remains authoritative."""

    task: Literal["coverage"] = "coverage"
    requirement: str = Field(min_length=1, max_length=6000)
    scope: Literal["requirement", "original_query"]
    translated_query: str | None = Field(default=None, min_length=1, max_length=6000)


class FidelityCitationRequest(DecisionRequest):
    """Opt-in independent citation facets; no facet is a gold support label."""

    task: Literal["citation"] = "citation"
    fidelity_dimension: Literal["support", "certainty", "attribution"]
    translated_query: str | None = Field(default=None, min_length=1, max_length=6000)
    focus: RetrievalPlan | None = None
    parent_claim: Claim | None = None


class SourceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    unit_id: str
    chunk_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=20000)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnchoredCitationRequest(FidelityCitationRequest):
    anchors: list[SourceAnchor] = Field(min_length=1, max_length=4)


class BasisCitationRequest(FidelityCitationRequest):
    """Explicit experimental fact-to-source mapping, not a support certificate."""

    basis_anchors: list[SourceAnchor] = Field(min_length=1, max_length=4)
    parent_claim: Claim


class AnswerCoverageRequest(FidelityCitationRequest):
    fidelity_dimension: Literal["support"] = "support"
    answer_claims: list[Claim] = Field(min_length=1, max_length=4)


class RetrievedConsistencyRequest(FidelityCitationRequest):
    """Explicit additional veto over all accepted evidence, including uncited passages."""

    fidelity_dimension: Literal["support"] = "support"
    retrieved_consistency: Literal[True] = True


class DecisionProvider(Protocol):
    source: str

    def decide(self, request: DecisionRequest) -> Decision | dict: ...


class Generator(Protocol):
    source: str

    def plan(
        self,
        query: str,
        intent: str,
        policy: IntentPolicy,
        *,
        evidence: list[Chunk] | None = None,
        iteration: int | None = None,
        previous_plan: RetrievalPlan | None = None,
    ) -> RetrievalPlan | dict: ...

    def rewrite(self, query: str, evidence: list[Chunk], iteration: int) -> str: ...

    def generate(self, query: str, evidence: list[Chunk]) -> Draft | dict: ...

    def generate_focused(
        self, query: str, evidence: list[Chunk], plan: RetrievalPlan
    ) -> Draft | dict: ...


class Retriever(Protocol):
    def search(self, query: str) -> list[dict]: ...
