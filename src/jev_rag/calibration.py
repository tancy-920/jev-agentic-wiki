"""Development-only decision cases and explicit human-review release gate."""

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_rag.agent.contracts import DecisionRequest, FocusedDecisionRequest
from jev_rag.jev.provider import validate_request


def input_hash(request):
    return hashlib.sha256(
        json.dumps(request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class CalibrationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"dev-[a-z]+-\d{3}")
    split: Literal["development"] = "development"
    query_family: str = Field(min_length=1)
    origin: Literal["local_retrieval", "constructed_state", "draft_claim"]
    request: FocusedDecisionRequest | DecisionRequest
    input_sha256: str

    @model_validator(mode="after")
    def validate_case(self):
        validate_request(self.request)
        if self.input_sha256 != input_hash(self.request):
            raise ValueError("case input hash mismatch")
        return self


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    input_sha256: str
    status: Literal["pending", "approved", "ambiguous", "rejected"] = "pending"
    gold_choice: str | None = None
    reviewer: str = ""
    reviewed_at: datetime | None = None
    rationale: str = ""
    human_attested: bool = False

    @model_validator(mode="after")
    def validate_review(self):
        if self.status == "pending":
            if self.gold_choice is not None or self.human_attested or self.reviewed_at is not None:
                raise ValueError("pending review cannot contain certified labels")
        else:
            if not self.human_attested or not self.reviewer.strip() or not self.rationale.strip():
                raise ValueError(
                    "completed review requires human attestation, reviewer and rationale"
                )
            if self.reviewed_at is None or self.reviewed_at.utcoffset() is None:
                raise ValueError("completed review requires timezone-aware review timestamp")
            if (self.status == "approved") != (self.gold_choice is not None):
                raise ValueError("only approved cases may contain gold labels")
        return self


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    text: str = Field(min_length=1)


class DraftAnnotation(BaseModel):
    """AI-authored proposal; cannot certify human review or populate gold."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    input_sha256: str
    provenance: Literal["AI_ANALYSIS"] = "AI_ANALYSIS"
    label_status: Literal["decided", "ambiguous", "invalid"]
    proposed_choice: str | None = None
    alternative_choices: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    recommendation: Literal["smoke_only", "revise", "exclude"]
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    human_review_status: Literal["pending"] = "pending"
    human_attested: Literal[False] = False
    gold_choice: None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.label_status == "decided" and self.proposed_choice is None:
            raise ValueError("decided draft requires a proposed choice")
        if self.label_status == "invalid" and self.proposed_choice is not None:
            raise ValueError("invalid input cannot receive a decision label")
        if self.label_status == "ambiguous" and len(set(self.alternative_choices)) < 2:
            raise ValueError("ambiguous draft must identify competing choices")
        if self.label_status != "decided" and self.recommendation == "smoke_only":
            raise ValueError("ambiguous/invalid input needs revision or exclusion")
        return self


def audit_draft_annotations(cases, annotations):
    cases = [CalibrationCase.model_validate(c) for c in cases]
    annotations = [DraftAnnotation.model_validate(a) for a in annotations]
    case_map = {c.id: c for c in cases}
    annotation_map = {a.case_id: a for a in annotations}
    if (
        not cases
        or len(case_map) != len(cases)
        or len(annotation_map) != len(annotations)
        or set(case_map) != set(annotation_map)
    ):
        raise ValueError("exactly one draft annotation per case required")
    for case_id, annotation in annotation_map.items():
        case = case_map[case_id]
        if annotation.input_sha256 != case.input_sha256:
            raise ValueError("stale draft annotation")
        choices = [*annotation.alternative_choices]
        if annotation.proposed_choice is not None:
            choices.append(annotation.proposed_choice)
        if any(c not in case.request.candidates for c in choices):
            raise ValueError("draft choice outside candidates")
        passages = {c.chunk_id: c.text for c in case.request.evidence}
        if any(
            s.chunk_id not in passages or s.text not in passages[s.chunk_id]
            for s in annotation.evidence_spans
        ):
            raise ValueError("draft evidence span is not present in case evidence")
    return {
        "type": "AI_PREANNOTATION_AUDIT_PENDING_HUMAN_REVIEW",
        "case_count": len(cases),
        "counts": {
            s: sum(a.label_status == s for a in annotations)
            for s in ("decided", "ambiguous", "invalid")
        },
        "recommendations": {
            r: sum(a.recommendation == r for a in annotations)
            for r in ("smoke_only", "revise", "exclude")
        },
        "human_reviewed": False,
        "release_ready": False,
        "threshold_selected": None,
    }


def confirmed_draft_reviews(cases, annotations, *, reviewer, reviewed_at):
    """Materialize already received human approval, retaining AI-assisted provenance."""
    audit_draft_annotations(cases, annotations)
    reviews = []
    for raw in annotations:
        draft = DraftAnnotation.model_validate(raw)
        status = {"decided": "approved", "ambiguous": "ambiguous", "invalid": "rejected"}[
            draft.label_status
        ]
        reviews.append(
            HumanReview(
                case_id=draft.case_id,
                input_sha256=draft.input_sha256,
                status=status,
                gold_choice=draft.proposed_choice if status == "approved" else None,
                reviewer=reviewer,
                reviewed_at=reviewed_at,
                human_attested=True,
                rationale="用户集中确认 AI 助写草案；" + draft.rationale,
            )
        )
    audit, approved = audit_reviews(cases, reviews)
    eligible_ids = {
        DraftAnnotation.model_validate(a).case_id
        for a in annotations
        if DraftAnnotation.model_validate(a).recommendation == "smoke_only"
    }
    eligible = [a for a in approved if a["case"]["id"] in eligible_ids]
    audit.update(
        annotation_method="AI-assisted, human-confirmed",
        approved_semantic_label_count=len(approved),
        smoke_eligible_count=len(eligible),
        held_out_independent_annotation=False,
        threshold_selection_ready=False,
    )
    return reviews, audit, eligible


def audit_reviews(cases, reviews):
    cases = [CalibrationCase.model_validate(c) for c in cases]
    reviews = [HumanReview.model_validate(r) for r in reviews]
    case_map = {c.id: c for c in cases}
    review_map = {r.case_id: r for r in reviews}
    if (
        not cases
        or len(case_map) != len(cases)
        or len(review_map) != len(reviews)
        or set(case_map) != set(review_map)
    ):
        raise ValueError("unique cases and exactly matching review IDs required")
    approved = []
    for case_id, case in case_map.items():
        review = review_map[case_id]
        if review.input_sha256 != case.input_sha256:
            raise ValueError("review is stale or linked to another input")
        if review.status == "approved":
            if review.gold_choice not in case.request.candidates:
                raise ValueError("gold choice outside allowed candidates")
            approved.append(
                {"case": case.model_dump(mode="json"), "review": review.model_dump(mode="json")}
            )
    counts = {
        s: sum(r.status == s for r in reviews)
        for s in ("pending", "approved", "ambiguous", "rejected")
    }
    return {
        "type": "DEVELOPMENT_HUMAN_REVIEW_AUDIT_NOT_BENCHMARK",
        "counts": counts,
        "approved_by_task": {
            t: sum(
                case_map[r.case_id].request.task == t and r.status == "approved" for r in reviews
            )
            for t in ("routing", "relevance", "evidence", "loop", "citation")
        },
        "release_ready": counts["pending"] == 0 and bool(approved),
        "semantic_review_verified_by_code": False,
        "threshold_selected": None,
    }, approved
