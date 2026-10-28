"""Typed application planning with history used exclusively to resolve referents."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frontend.dialogue_policy import ClarificationMove, DialogueMove
from jev_rag.agent.intent import RetrievalPlan, SearchQuery
from jev_rag.agent.language import TranslationDraft
from jev_rag.terms import TermChoice

CONVERSATION_POLICY = {
    "version": "application-joint-plan-context-v7-original-question-authority",
    "history_turns": 3,
    "resolution": "one shared planning call; no separate intent prediction",
    "history_is_evidence": False,
    "unreferenced_product_retrieval": "discard proposed standalone rewrite; validate retrieval view against exact original; keep proposal in audit record",
    "original_question_preserved": True,
    "unresolved_referent": "clarify",
    "reranker_query": "full English retrieval view from the same planning call; not Chinese against English cross encoder",
    "query_budget": "code selects the first distinct proposed local queries within the intent limit; preserves original requirements and all evidence needs",
    "proposal_query_cap": 6,
    "attention_symbol_bundles": "product-only QKV/QK/KV may retain ordered symbols separated by whitespace, comma or slash; original question unchanged",
    "technical_noun_inflections": "product-only bounded ordinary technical nouns may be singular/plural; opaque names, versions, numbers and protected clauses remain guarded",
}


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: UUID
    question: str = Field(min_length=1, max_length=3000)
    answer: str | None = Field(default=None, max_length=12000)
    status: str = Field(min_length=1, max_length=100)
    failure_summary: str | None = Field(default=None, max_length=1500)
    dialogue_reply: str | None = Field(default=None, max_length=1600)
    context_omissions: list[Literal["answer", "failure_summary", "dialogue_reply"]] = Field(default_factory=list, max_length=3)


class ConversationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    standalone_question: str = Field(min_length=1, max_length=3000)
    context_turn_ids: list[UUID] = Field(default_factory=list, max_length=3)
    action: Literal["RETRIEVE", "CLARIFY"]
    plan: RetrievalPlan | None
    retrieval_translation: TranslationDraft | None

    @model_validator(mode="after")
    def consistent_action(self):
        if not self.standalone_question.strip():
            raise ValueError("empty standalone question")
        if (self.action == "RETRIEVE") != (self.plan is not None):
            raise ValueError("retrieval action requires a plan; clarification forbids it")
        if (self.action == "RETRIEVE") != (self.retrieval_translation is not None):
            raise ValueError("retrieval requires an English view; clarification forbids it")
        if len(set(self.context_turn_ids)) != len(self.context_turn_ids):
            raise ValueError("duplicate context references")
        return self

    def validate_context(self, query, context, policy, *, allow_name_spacing=False, term_corrections=()):
        allowed = {t.request_id for t in context}
        if not set(self.context_turn_ids) <= allowed:
            raise ValueError("invented conversation references")
        if not self.context_turn_ids and self.standalone_question != query:
            raise ValueError("standalone query changed without context references")
        if self.plan:
            self.plan.validate_for(query, self.plan.intent, policy)
            self.retrieval_translation.validate_for(
                self.standalone_question, allow_name_spacing=allow_name_spacing,
                term_corrections=term_corrections,
                allow_technical_inflections=allow_name_spacing,
            )
        return self

    def bounded_queries(self, policy):
        """Authorize a bounded subset of search suggestions, never rewrite facts."""
        if self.plan is None:
            return self
        limit = policy.max_queries if self.plan.intent in {"COMPARISON", "MULTI_HOP"} else 1
        queries = list(dict.fromkeys(self.plan.queries))[:limit]
        if queries == self.plan.queries:
            return self
        values = self.model_dump(mode="json")
        values["plan"]["queries"] = queries
        return type(self).model_validate(values)


class ProposedRetrievalPlan(RetrievalPlan):
    """Bounded search suggestions, never an executable retrieval authorization."""

    queries: list[SearchQuery] = Field(min_length=1, max_length=6)


class ProposedConversationPlan(ConversationPlan):
    plan: ProposedRetrievalPlan | None


class TermConversationPlan(ProposedConversationPlan):
    term_choices: list[TermChoice] = Field(default_factory=list, max_length=8)


def evaluation_query(original, resolved):
    if original == resolved:
        return original
    return (
        f"Original user question: {original}\n"
        f"Conversation referent resolution (fallible interpretation, NOT evidence): {resolved}\n"
        "Judge the original question with these referents; do not invent additional requirements."
    )


class ChatConversationPlan(ProposedConversationPlan):
    """Product-only dialogue schema; legacy research planning stays unchanged."""

    action: Literal["RETRIEVE", "CLARIFY", "CHAT"]
    context_turn_ids: list[UUID] = Field(default_factory=list, max_length=6)
    dialogue: DialogueMove | None = None
    clarification: ClarificationMove | None = None

    @model_validator(mode="after")
    def consistent_dialogue(self):
        if (self.action == "CHAT") != (self.dialogue is not None):
            raise ValueError("only CHAT requires a dialogue move")
        if self.action != "CLARIFY" and self.clarification is not None:
            raise ValueError("clarification move requires CLARIFY")
        return self

    def validate_context(self, query, context, policy, **kwargs):
        super().validate_context(query, context, policy, **kwargs)
        if self.action == "CHAT":
            from frontend.dialogue_policy import knowledge_request, social_kind

            contextual = self.dialogue.reply is not None and self.dialogue.kind in {
                "conversation", "feedback", "support", "greeting", "thanks", "pause",
                "acknowledge", "failure_explanation", "learning_support",
            }
            if ((not contextual and social_kind(query) != self.dialogue.kind)
                or knowledge_request(query) or self.standalone_question != query):
                raise ValueError("mixed or technical messages cannot bypass knowledge verification")
        return self


class TermChatConversationPlan(ChatConversationPlan):
    term_choices: list[TermChoice] = Field(default_factory=list, max_length=8)
