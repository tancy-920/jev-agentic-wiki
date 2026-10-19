import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_rag.io import read_jsonl, sha256_file
from jev_rag.llm.policy import QueryProfile

CATEGORIES = {
    "easy",
    "medium",
    "hard",
    "multi_hop",
    "comparative",
    "conflicting_evidence",
    "out_of_knowledge",
}


class EvidenceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    chunk_id: str
    quote: str = Field(min_length=1, max_length=10000)


class EvaluationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"q\d{3,5}")
    question: str = Field(min_length=1, max_length=6000)
    category: Literal[
        "easy",
        "medium",
        "hard",
        "multi_hop",
        "comparative",
        "conflicting_evidence",
        "out_of_knowledge",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    gold_documents: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    requires_retrieval: bool = True
    expected_abstention: bool = False
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list)
    input_profile: QueryProfile = Field(default_factory=QueryProfile)
    provenance: Literal["AI_DRAFT", "HUMAN_AUTHORED"] = "AI_DRAFT"

    @model_validator(mode="after")
    def valid_text(self):
        if not self.question.strip() or any(not fact.strip() for fact in self.expected_facts):
            raise ValueError("empty question/fact")
        if len(set(self.gold_documents)) != len(self.gold_documents):
            raise ValueError("duplicate gold documents")
        if self.expected_abstention and self.gold_documents:
            raise ValueError("abstention cannot claim gold supporting documents")
        return self


def case_hash(question):
    return hashlib.sha256(
        json.dumps(question.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class QuestionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    input_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    status: Literal["pending", "approved", "excluded"] = "pending"
    reviewer: str = ""
    reviewed_at: datetime | None = None
    rationale: str = ""
    human_attested: bool = False

    @model_validator(mode="after")
    def human_release(self):
        if self.status == "pending":
            if self.human_attested or self.reviewed_at is not None:
                raise ValueError("draft cannot attest human review")
        elif (
            not self.human_attested
            or not self.reviewer.strip()
            or not self.rationale.strip()
            or self.reviewed_at is None
            or self.reviewed_at.utcoffset() is None
        ):
            raise ValueError("completed review requires real human attestation and dated rationale")
        return self


def audit_questions(questions, reviews, *, chunks=None, formal=False, split="development"):
    questions = [EvaluationQuestion.model_validate(q) for q in questions]
    reviews = [QuestionReview.model_validate(r) for r in reviews]
    by_id, by_review = {q.id: q for q in questions}, {r.case_id: r for r in reviews}
    if (
        not questions
        or len(by_id) != len(questions)
        or len(by_review) != len(reviews)
        or set(by_id) != set(by_review)
    ):
        raise ValueError("unique cases and exactly matching reviews required")
    normalized = [" ".join(q.question.casefold().split()) for q in questions]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate questions")
    accepted = []
    for identifier, question in by_id.items():
        review = by_review[identifier]
        if review.input_sha256 != case_hash(question):
            raise ValueError("stale review; question/reference facts changed")
        if chunks is not None:
            for anchor in question.evidence_anchors:
                chunk = chunks.get(anchor.chunk_id)
                if (
                    chunk is None
                    or chunk.paper_id != anchor.paper_id
                    or anchor.quote not in chunk.text
                ):
                    raise ValueError("evidence anchor is absent/stale")
        if review.status == "approved":
            if not question.reference_answer.strip() or not question.expected_facts:
                raise ValueError("approved question needs reference answer and expected facts")
            if (
                question.requires_retrieval
                and not question.expected_abstention
                and (
                    not question.gold_documents
                    or not question.evidence_anchors
                    or set(question.gold_documents)
                    != {a.paper_id for a in question.evidence_anchors}
                )
            ):
                raise ValueError("reviewed facts need actual corpus evidence anchors")
            accepted.append(question)
    counts = {s: sum(r.status == s for r in reviews) for s in ("pending", "approved", "excluded")}
    categories = {q.category for q in accepted}
    ready = (
        counts["pending"] == 0
        and len(accepted) >= 100
        and CATEGORIES <= categories
        and split == "held_out"
        and chunks is not None
    )
    if formal and not ready:
        raise ValueError(
            "formal benchmark needs 100+ human-reviewed held-out cases, category coverage and validated evidence"
        )
    return {
        "type": "DATASET_REVIEW_AUDIT",
        "counts": counts,
        "case_count": len(questions),
        "approved_categories": sorted(categories),
        "formal_release_ready": ready,
        "quality_scores_generated": False,
    }, accepted


def load_packet(directory: Path, *, chunks=None, formal=False):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    question_path = directory / "questions.jsonl"
    if sha256_file(question_path) != manifest["questions_sha256"]:
        raise ValueError("dataset hash changed")
    questions = [EvaluationQuestion.model_validate(q) for q in read_jsonl(question_path)]
    reviews = [QuestionReview.model_validate(r) for r in read_jsonl(directory / "reviews.jsonl")]
    audit, accepted = audit_questions(
        questions, reviews, chunks=chunks, formal=formal, split=manifest["split"]
    )
    if manifest.get("candidate_only") and accepted:
        confirmation = json.loads(
            (directory / "human_confirmation.json").read_text(encoding="utf-8")
        )
        if (
            confirmation.get("status") != "approved"
            or confirmation.get("human_attested") is not True
            or not str(confirmation.get("approval_text", "")).strip()
            or confirmation.get("split") != "held_out"
            or confirmation.get("held_out_eligible") is not True
            or confirmation.get("new_api_authorization") is not False
            or any(
                confirmation.get(key) != sha256_file(directory / name)
                for key, name in (
                    ("manifest_sha256", "manifest.json"),
                    ("questions_sha256", "questions.jsonl"),
                    ("reviews_sha256", "reviews.jsonl"),
                    ("pending_reviews_sha256", "reviews.pending.jsonl"),
                    ("ai_proposals_sha256", "ai_proposals.jsonl"),
                    ("review_sha256", "REVIEW.md"),
                )
            )
        ):
            raise ValueError("held-out human confirmation missing/stale; no silent gold release")
    return manifest, questions, accepted, audit
