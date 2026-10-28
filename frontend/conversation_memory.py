"""Local bounded, source-linked context selection; history never becomes evidence.

No model summary, embedding, API call, or automatic long-term profile is used.
The existing session receipts provide recovery; closed grants remain closed.
"""

import json
import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frontend.dialogue_policy import DIALOGUE_POLICY


class MemoryQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    message_id: UUID
    user_quote: str = Field(min_length=1, max_length=500)
    topic: str = Field(default="", max_length=500)


class TaskMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_updates: list[MemoryQuote] = Field(default_factory=list, max_length=6)
    preferences: dict[str, MemoryQuote] = Field(default_factory=dict, max_length=6)
    unresolved: list[MemoryQuote] = Field(default_factory=list, max_length=6)
    focus: MemoryQuote | None = None


PREFERENCE_OPTIONS = {
    "detail": {"concise": "简短直接", "balanced": "适中", "detailed": "详细展开"},
    "order": {"conclusion": "先给结论", "example": "先举例子", "formula": "先看公式"},
    "followup": {"needed": "仅必要时追问", "avoid": "不主动追加问题"},
    "support": {"warm": "温和鼓励", "direct": "直接说明，不额外安慰"},
}


class SavedPreference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: Literal["detail", "order", "followup", "support"]
    value: str = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def allowed_choice(self):
        if self.value not in PREFERENCE_OPTIONS[self.key]:
            raise ValueError("only explicit communication preferences can be saved")
        return self


def load_preferences(workspace):
    path = workspace / ".local/chat-preferences.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf8"))
    if not isinstance(data, list) or len(data) > 4:
        raise ValueError("invalid local preference store")
    parsed = [SavedPreference.model_validate(p) for p in data]
    if len({p.key for p in parsed}) != len(parsed):
        raise ValueError("duplicate saved preference")
    return {p.key: p.value for p in parsed}


def preference_view(values):
    return {key: PREFERENCE_OPTIONS[key][value] for key, value in values.items()}


def update_task_memory(memory, turn):
    """Remember original user quotes, never derive facts from assistant prose."""
    query = turn.question.strip()
    if len(query) > 500:
        return memory  # Do not cut a condition into a different statement.
    quote = MemoryQuote(message_id=turn.request_id, user_quote=query,
                        topic=memory.focus.user_quote if memory.focus else query)
    patterns = {
        "detail": r"简短|详细|通俗|少.{0,3}公式",
        "order": r"先.{0,6}(?:结论|公式|例子)",
        "followup": r"(?:不要|别|不用).{0,8}(?:反问|追问|问我|选择题)|必要.{0,5}追问",
        "support": r"(?:不要|别|不用).{0,6}安慰|鼓励我|陪我聊",
    }
    explicit_preference = re.search(r"^(?:请|以后|接下来|这次|本次|不要|别|不用|我希望|我想要|先给|先看|先举)", query)
    for key, pattern in patterns.items():
        if explicit_preference and re.search(pattern, query):
            memory.preferences[key] = quote
    polarity = re.sub(r"是不是|能不能|有没有", "", query)
    if re.search(r"只(?:考虑|讨论|看|限)|我指的是|我问的是|更正|纠正|不是.+而是|假设|固定.+(?:输入|条件)|I mean|correction", polarity, re.I):
        memory.scope_updates = [q for q in memory.scope_updates if q.message_id != quote.message_id][-5:] + [quote]
    # Social/clarification turns do not silently resolve an unanswered question.
    if turn.status not in {"chat_completed", "clarification_required", "cancelled"}:
        memory.focus = MemoryQuote(message_id=turn.request_id, user_quote=query, topic=query)
        if turn.status == "answered":
            memory.unresolved = [q for q in memory.unresolved if q.user_quote != query]
        elif not any(q.message_id == quote.message_id for q in memory.unresolved):
            memory.unresolved = memory.unresolved[-5:] + [quote]
    return memory


def task_view(memory, query):
    """Filter old task quotes by topic; they are attributed constraints, not facts."""
    def relevant(quote):
        names = set(re.findall(r"[a-z][a-z0-9_-]*", query.casefold())) - {"mock", "i", "mean"}
        if names:
            return bool(names & set(re.findall(r"[a-z][a-z0-9_-]*", (quote.topic + quote.user_quote).casefold())))
        return True  # Chinese follow-ups may have no literal Latin referent.
    return {
        "scope_updates": [q.model_dump(mode="json") for q in memory.scope_updates if relevant(q)][-3:],
        "preferences": {k: q.model_dump(mode="json") for k, q in memory.preferences.items()},
        "unresolved": [q.model_dump(mode="json") for q in memory.unresolved if relevant(q)][-3:],
        "history_is_evidence": False,
        "interpretation": "Original user wording only; current message overrides remembered preferences/conditions. "
                          "Scope quotes apply to their attributed topic, not every future topic. "
                          "Unresolved means not delivered, not proven false. No automatic retry is authorized.",
    }


def enrich_state(state, memory, query, preferences=None):
    result = {**state, "task_memory": task_view(memory, query),
              "saved_preferences": preference_view(preferences or {})}
    # Drop whole oldest attributed entries; never truncate negation/conditions.
    task = result["task_memory"]
    while len(json.dumps(result, ensure_ascii=False)) > DIALOGUE_POLICY["max_state_characters"]:
        if task["unresolved"]:
            task["unresolved"].pop(0)
        elif len(task["scope_updates"]) > 1:
            task["scope_updates"].pop(0)
        elif task["preferences"]:
            task["preferences"].pop(next(iter(task["preferences"])))
        elif task["scope_updates"]:
            task["scope_updates"].pop(0)
        else:
            # Existing state is already bounded. Omit metadata rather than
            # discarding the explicit saved preference choices.
            result.pop("user_wording_preferences", None)
            result.pop("user_scope_updates", None)
            task.pop("interpretation", None)
            if len(json.dumps(result, ensure_ascii=False)) > DIALOGUE_POLICY["max_state_characters"]:
                result.pop("focus", None)
                result.pop("last_outcome", None)
            break
    return result


def _terms(text):
    lowered = text.casefold()
    stopwords = {"what", "why", "how", "is", "are", "the", "a", "an", "of", "and", "or",
                 "to", "it", "this", "that", "explain", "please", "mock"}
    latin = set(re.findall(r"[a-z][a-z0-9_-]*", lowered)) - stopwords
    chinese = re.findall(r"[\u4e00-\u9fff]+", lowered)
    return latin | ({part[i:i+2] for part in chinese for i in range(len(part)-1)}
                    - {"什么", "么是", "为何", "怎么", "如何", "为什", "么会", "解释", "一下", "请问"})


def retain_turn(turns, turn):
    if any(t.request_id == turn.request_id for t in turns):
        return
    turns.append(turn)
    del turns[:-DIALOGUE_POLICY["max_retained_turns"]]


def select_context(turns, query, *, enabled=False, task_memory=None):
    if not enabled:
        return tuple(turns[-3:])
    turns = turns[-DIALOGUE_POLICY["max_retained_turns"]:]
    pinned_ids = set()
    if task_memory is not None:
        from frontend.conversation import ConversationTurn

        view = task_view(task_memory, query)
        quotes = [*view["unresolved"], *view["scope_updates"]]
        if task_memory.focus and not re.search(r"[A-Za-z]", query):
            quotes.append(task_memory.focus.model_dump(mode="json"))
        known = {str(t.request_id) for t in turns}
        pinned = {}
        for quote in quotes:
            if quote["message_id"] not in known:
                pinned[quote["message_id"]] = ConversationTurn(
                    request_id=quote["message_id"], question=quote["user_quote"],
                    status="remembered_user_context", answer=None)
        turns = [*list(pinned.values())[-3:], *turns]
        pinned_ids = {q["message_id"] for q in quotes[-3:]}
    if not turns:
        return ()
    # Preserve recent corrections verbatim. Add older related questions, not a
    # model's reconstruction of facts. Ambiguous pronouns are still clarified.
    recent = list(range(max(0, len(turns)-3), len(turns)))
    terms = _terms(query)
    older = sorted(range(max(0, len(turns)-3)),
                   key=lambda i: (len(terms & _terms(turns[i].question)), i), reverse=True)
    selected = set(recent)
    for i in range(len(turns)):
        if str(turns[i].request_id) in pinned_ids and len(selected) < DIALOGUE_POLICY["max_context_turns"]:
            selected.add(i)
    for i in older:
        if len(selected) >= DIALOGUE_POLICY["max_context_turns"]:
            break
        if terms & _terms(turns[i].question):
            selected.add(i)
    # Prefer newer exact messages within a strict input size; never clip a
    # condition inside a turn or silently present a partial answer as complete.
    chosen, used = [], 0
    for i in sorted(selected, reverse=True):
        turn = turns[i]
        size = len(turn.question) + len(turn.answer or "") + len(turn.failure_summary or "") + len(turn.dialogue_reply or "")
        if used + size > DIALOGUE_POLICY["max_context_characters"]:
            omissions = list(turn.context_omissions)
            if turn.answer is not None:
                omissions.append("answer")
            if turn.failure_summary is not None:
                omissions.append("failure_summary")
            if turn.dialogue_reply is not None:
                omissions.append("dialogue_reply")
            turn = turn.model_copy(update={"answer": None, "failure_summary": None,
                                           "dialogue_reply": None,
                                           "context_omissions": list(dict.fromkeys(omissions))})
            size = len(turn.question)
        if used + size <= DIALOGUE_POLICY["max_context_characters"]:
            chosen.append((i, turn))
            used += size
    return tuple(t for _, t in sorted(chosen))


def session_state(turns, query):
    """Extract attributed conversation state, not a technical summary or user profile."""
    from frontend.dialogue_policy import social_kind

    context = select_context(turns, query, enabled=True)
    technical = [t for t in context if t.status != "chat_completed" and social_kind(t.question) is None]
    latest = technical[-1] if technical else None
    terms = _terms(query)
    focus = max(technical, key=lambda t: (len(terms & _terms(t.question)),
                                         technical.index(t))) if technical else None
    # Explicit wording only; never infer an enduring preference or a fact.
    preferences = [t for t in turns if re.fullmatch(
        r"(?:请|以后|接下来|本次|这次).{0,25}(?:简短|详细|先.{0,8}公式|先.{0,8}结论|通俗|举例).{0,25}",
        t.question.strip())][-2:]
    result = {
        "version": DIALOGUE_POLICY["version"],
        "scope": "current_conversation_only",
        "history_is_evidence": False,
        "focus": ({"message_id": str(focus.request_id), "user_question": focus.question}
                  if focus else None),
        "last_outcome": ({"message_id": str(latest.request_id), "status": latest.status,
                          "failure_summary": latest.failure_summary}
                         if latest else None),
        "user_wording_preferences": [
            {"message_id": str(t.request_id), "user_quote": t.question} for t in preferences],
        "selected_message_ids": [str(t.request_id) for t in context],
        "retained_turns": len(turns),
        "user_scope_updates": [
            {"message_id": str(t.request_id), "user_quote": t.question}
            for t in context if len(t.question) <= 500 and re.search(
                r"不是|我指的是|我问的是|更正|纠正|I mean|instead|correction", t.question, re.IGNORECASE)
        ][-2:],
    }
    if len(json.dumps(result, ensure_ascii=False)) > DIALOGUE_POLICY["max_state_characters"]:
        # Omit whole quotes, never clip negations/conditions into a new meaning.
        if result["focus"]:
            result["focus"].pop("user_question", None)
        result["user_scope_updates"] = []
        result["user_wording_preferences"] = []
    return result
