"""Bounded intent policy and non-authoritative retrieval plans, shared by both agents."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["FACT_LOOKUP", "CONCEPT_EXPLANATION", "COMPARISON", "MULTI_HOP", "CLARIFY"]
INTENTS = list(Intent.__args__)
EvidenceNeed = Annotated[str, Field(min_length=1, max_length=500)]
SearchQuery = Annotated[str, Field(min_length=1, max_length=6000)]
ConstraintSpan = Annotated[str, Field(min_length=1, max_length=6000)]


class IntentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # Legacy constructor/config snapshots remain replayable; base.yaml enables v1.
    enabled: bool = False
    max_queries: int = Field(default=2, ge=1, le=2)
    uncertainty: Literal["clarify", "original_query"] = "clarify"

    def snapshot(self):
        # Preserve old artifact fingerprints for the historical strict policy.
        values = (
            {"version": "bounded-intent-plan-v1", **self.model_dump(exclude={"uncertainty"})}
            if self.uncertainty == "clarify"
            else {"version": "bounded-intent-plan-v2", **self.model_dump()}
        )
        values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intent: Intent
    goal: str = Field(min_length=1, max_length=1000)
    evidence_needs: list[EvidenceNeed] = Field(min_length=1, max_length=6)
    queries: list[SearchQuery] = Field(min_length=1, max_length=2)
    user_constraints: list[ConstraintSpan] = Field(default_factory=list, max_length=20)

    def validate_for(self, original_query: str, selected_intent: str, policy: IntentPolicy):
        if self.intent != selected_intent or self.intent == "CLARIFY":
            raise ValueError("planner cannot override the selected intent")
        limit = policy.max_queries if self.intent in {"COMPARISON", "MULTI_HOP"} else 1
        if len(self.queries) > limit or len(set(self.queries)) != len(self.queries):
            raise ValueError("invalid query budget or duplicate queries")
        for query in self.queries:
            validate_search_query(query)
        if any(not need.strip() or len(need) > 500 for need in self.evidence_needs):
            raise ValueError("invalid evidence needs")
        if not self.goal.strip():
            raise ValueError("empty goal")
        if any(not c.strip() or c not in original_query for c in self.user_constraints):
            raise ValueError("constraints must be exact user-query spans")
        # Syntax and literal anchors are checked, not semantic equivalence/completeness.
        return self

    def summary(self):
        return {
            "intent": self.intent,
            "plan_sha256": hashlib.sha256(
                json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "query_count": len(self.queries),
            "evidence_need_count": len(self.evidence_needs),
            "constraint_count": len(self.user_constraints),
            "semantic_validity": "model_proposal_not_ground_truth",
        }


def validate_search_query(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 6000:
        raise ValueError("invalid search query")
    # Strings are exclusively local search text, never tools, URLs or commands.
    return query


def clarification(query, language="auto"):
    if language == "zh-CN" or any("\u4e00" <= c <= "\u9fff" for c in query):
        return "请补充你要了解的研究对象或比较维度：你更想了解定义、工作原理、方法对比，还是跨论文综合？"
    return "Which concept or methods do you mean, and do you want a definition, mechanism, comparison, or synthesis across papers?"
