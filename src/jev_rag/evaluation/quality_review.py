"""Independent, source-bound human scoring; never reuse agent judge predictions."""

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from jev_rag.agent.contracts import Claim
from jev_rag.schema import Chunk

from .dataset import EvaluationQuestion, QuestionReview, case_hash
from .metrics import QUALITY

PROTOCOL = "human-quality-rubric-v1"


class QualityInput(BaseModel):
    """Original question, actual displayed answer and final generation-visible context."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["MOCK", "LIVE"]
    run_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    question: EvaluationQuestion
    answer: str | None
    displayed_claims: list[Claim]
    contexts: list[Chunk]

    @model_validator(mode="after")
    def actual_answer_evidence(self):
        ids = [c.chunk_id for c in self.contexts]
        if len(ids) != len(set(ids)):
            raise ValueError("unique final context IDs required")
        if self.status != "answered" and self.displayed_claims:
            raise ValueError("withheld drafts cannot be scored as displayed factual claims")
        if any(c.text not in (self.answer or "") for c in self.displayed_claims):
            raise ValueError("displayed claims must occur in actual final answer")
        if any(not set(c.citations).issubset(ids) for c in self.displayed_claims):
            raise ValueError("displayed citation outside actual final context")
        return self

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


class ClaimReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Independent reviewer identifies factual assertions; paragraphs are not
    # assumed atomic and provider verification spans are not inherited as gold.
    text: str = Field(min_length=1)
    supported: StrictBool | None = None
    rationale: str = Field(min_length=1)


class ContextReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    relevant: StrictBool | None = None


class FactReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_index: int = Field(ge=0, strict=True)
    supporting_context_ids: list[str] | None = None


class QualityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Literal["human-quality-rubric-v1"] = PROTOCOL
    run_id: str
    input_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    status: Literal["pending", "approved"] = "pending"
    origin: Literal["AI_DRAFT", "HUMAN_AUTHORED"] = "AI_DRAFT"
    reviewer: str = ""
    reviewed_at: datetime | None = None
    approval_text: str = ""
    human_attested: StrictBool = False
    rationale: str = ""
    all_factual_claims_covered: StrictBool = False
    claims: list[ClaimReview] = Field(default_factory=list)
    # A custom human rubric, NOT the RAGAS embedding-based answer-relevance metric.
    intent_addressed: Annotated[int, Field(strict=True, ge=0, le=2)] | None = None
    constraints_respected: Annotated[int, Field(strict=True, ge=0, le=2)] | None = None
    on_topic: Annotated[int, Field(strict=True, ge=0, le=2)] | None = None
    contexts: list[ContextReview] = Field(default_factory=list)
    facts: list[FactReview] = Field(default_factory=list)

    @model_validator(mode="after")
    def approval(self):
        if self.status == "pending":
            if self.human_attested or self.reviewed_at is not None or self.approval_text:
                raise ValueError("draft cannot attest human approval")
        elif (
            not self.human_attested
            or not all(v.strip() for v in (self.reviewer, self.approval_text, self.rationale))
            or self.reviewed_at is None
            or self.reviewed_at.utcoffset() is None
        ):
            raise ValueError("explicit dated human approval and rationale required")
        for value in (self.intent_addressed, self.constraints_respected, self.on_topic):
            if value is not None and type(value) is not int:
                raise ValueError("rubric dimensions must be integer categories, not confidence")
        return self


def score_review(raw_input, raw_review, raw_question_review):
    inputs = QualityInput.model_validate(raw_input)
    review = QualityReview.model_validate(raw_review)
    question_review = QuestionReview.model_validate(raw_question_review)
    if (
        review.run_id != inputs.run_id
        or review.input_sha256 != inputs.fingerprint()
        or question_review.case_id != inputs.question.id
        or question_review.input_sha256 != case_hash(inputs.question)
    ):
        raise ValueError("answer, question, source or review hash mismatch")
    context_ids = {c.chunk_id for c in inputs.contexts}
    context_map = {c.chunk_id: c for c in review.contexts}
    fact_map = {f.fact_index: f for f in review.facts}
    if (
        len(context_map) != len(review.contexts)
        or set(context_map) != context_ids
        or len(fact_map) != len(review.facts)
        or set(fact_map) != set(range(len(inputs.question.expected_facts)))
        or len({c.text for c in review.claims}) != len(review.claims)
        or any(c.text not in (inputs.answer or "") for c in review.claims)
        or (review.claims and inputs.status != "answered")
        or any(
            f.supporting_context_ids is not None
            and (
                len(set(f.supporting_context_ids)) != len(f.supporting_context_ids)
                or not set(f.supporting_context_ids).issubset(context_ids)
            )
            for f in review.facts
        )
    ):
        raise ValueError(
            "independent review must cover exact final contexts/facts and actual claims"
        )
    scores = {k: None for k in QUALITY}
    if review.status == "approved" and question_review.status == "approved":
        labels = [c.supported for c in review.claims]
        if review.all_factual_claims_covered and labels and all(v is not None for v in labels):
            scores["Faithfulness"] = sum(labels) / len(labels)
        rubric = [review.intent_addressed, review.constraints_respected, review.on_topic]
        if all(v is not None for v in rubric):
            scores["Answer Relevance"] = sum(rubric) / 6
        relevance = [c.relevant for c in review.contexts]
        if relevance and all(v is not None for v in relevance):
            scores["Context Precision"] = sum(relevance) / len(relevance)
        facts = [f.supporting_context_ids for f in review.facts]
        if not inputs.question.expected_abstention and facts and all(v is not None for v in facts):
            scores["Context Recall"] = sum(bool(v) for v in facts) / len(facts)
    return {
        "mode": inputs.mode,
        "run_id": inputs.run_id,
        "quality": scores,
        "quality_source": "HUMAN_REVIEW"
        if review.status == "approved" and question_review.status == "approved"
        else "unmeasured",
        "quality_protocol": PROTOCOL,
        "quality_input_sha256": inputs.fingerprint(),
        "annotation_origin": review.origin,
        "faithfulness_denominator": len(review.claims),
        "context_precision_denominator": len(review.contexts),
        "context_recall_denominator": len(review.facts),
        "semantic_review_certified_by_code": False,
    }


def quality_input(record, question):
    if "quality_context" not in record:
        raise ValueError("historical record lacks final-context snapshot; do not invent it")
    snapshot = record["quality_context"]
    return QualityInput(
        mode=record["mode"],
        run_id=record["run_id"],
        status=record["status"],
        question=question,
        answer=record["answer"],
        displayed_claims=snapshot["displayed_claims"],
        contexts=snapshot["contexts"],
    )
