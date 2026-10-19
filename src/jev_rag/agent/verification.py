"""Bounded exact-text verification partitions; semantic atomicity is not certified."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Claim, Draft


class VerificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["paragraph", "paragraph_and_spans"] = "paragraph"
    max_spans_per_paragraph: int = Field(default=8, ge=1, le=8)
    max_total_spans: int = Field(default=16, ge=1, le=24)

    def snapshot(self):
        values = {"version": "exact-covered-span-verification-v1", **self.model_dump()}
        values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values


class PartitionedClaim(Claim):
    spans: list[Claim] = Field(min_length=1, max_length=8)


class PartitionedDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[PartitionedClaim] = Field(min_length=1, max_length=30)


def validate_draft(value, policy, *, max_claims=30):
    model = PartitionedDraft if policy.mode == "paragraph_and_spans" else Draft
    draft = model.model_validate(value)
    if len(draft.claims) > max_claims:
        raise ValueError("paragraph budget exceeded")
    if policy.mode == "paragraph":
        return draft
    count = 0
    for paragraph in draft.claims:
        count += len(paragraph.spans)
        if len(paragraph.spans) > policy.max_spans_per_paragraph:
            raise ValueError("paragraph span budget exceeded")
        if not paragraph.citations or len(set(paragraph.citations)) != len(paragraph.citations):
            raise ValueError("unique paragraph citations required")
        # No omission, paraphrase or unchecked transition text during decomposition.
        if "".join(span.text for span in paragraph.spans) != paragraph.text:
            raise ValueError("spans must exactly cover the displayed paragraph in order")
        union = set()
        for span in paragraph.spans:
            if (
                not span.text.strip()
                or not span.citations
                or len(set(span.citations)) != len(span.citations)
            ):
                raise ValueError("nonempty cited spans required")
            if not set(span.citations).issubset(paragraph.citations):
                raise ValueError("span citations must belong to paragraph")
            union.update(span.citations)
        if union != set(paragraph.citations):
            raise ValueError("paragraph citations must be used by spans")
    if count > policy.max_total_spans:
        raise ValueError("total span budget exceeded")
    return draft


def checks_for(paragraph, policy):
    parent = Claim.model_validate(paragraph.model_dump(exclude={"spans"}))
    yield parent, None, None
    if policy.mode == "paragraph_and_spans":
        for index, span in enumerate(paragraph.spans):
            yield span, parent, index
