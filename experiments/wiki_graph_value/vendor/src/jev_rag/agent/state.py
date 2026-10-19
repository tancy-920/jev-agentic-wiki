from typing import TypedDict


class AgentState(TypedDict):
    run_id: str
    mode: str
    query: str
    retrieval_query: str
    query_translation: dict | None
    rewritten_query: str | None
    route: str | None
    intent: str | None
    intent_fallback: bool
    retrieval_plan: dict | None
    retrieval_requests: int
    retrieved_candidates: int
    retrieved_documents: list[dict]
    relevant_documents: list[dict]
    evidence_sufficient: bool | None
    reasoning_steps: list[dict]
    tool_calls: list[dict]
    events: list[dict]
    iteration: int
    max_iterations: int
    loop_count: int
    retry_count: int
    early_stop: bool
    draft_answer: dict | None
    draft_history: list[dict]
    citation_results: list[dict]
    final_citation_results: list[dict]
    citation_repair_count: int
    verification_feedback: list[dict]
    final_answer: str | None
    termination_reason: str | None
    next_node: str
