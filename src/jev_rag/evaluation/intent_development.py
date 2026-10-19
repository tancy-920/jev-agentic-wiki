"""Public development fixtures, NOT reviewed ground truth or held-out test data."""

from jev_rag.agent.language import digest

CASES = [
    {
        "id": "mechanism_zh",
        "question": "请解释缩放点积注意力为什么要除以键维度的平方根；不要声称它保证提高所有任务的准确率。",
        "proposed_intent": "CONCEPT_EXPLANATION",
        "annotation_reason": "Explicit why/mechanism request; preserve the prohibition on universal accuracy guarantees.",
    },
    {
        "id": "comparison_zh",
        "question": "比较 BERT 和 GPT-3 的预训练目标，不要比较推理速度。",
        "proposed_intent": "COMPARISON",
        "annotation_reason": "Two named models and an explicit comparison dimension; speed is excluded.",
    },
    {
        "id": "ambiguity_zh",
        "question": "它和另一个方法有什么区别？",
        "proposed_intent": "CLARIFY",
        "annotation_reason": "Neither referent is supplied; interpreting them would invent user intent.",
    },
]


def draft_cases():
    return [
        {
            **case,
            "question_sha256": digest(case["question"]),
            "annotation_status": "AI_ASSISTED_DRAFT_PENDING_HUMAN_REVIEW",
        }
        for case in CASES
    ]


def selection_record(decision, threshold):
    accepted = all(
        v is not None and v >= threshold for v in (decision.confidence, decision.probability)
    )
    return {
        **decision.model_dump(),
        "accepted": accepted,
        "application_outcome": decision.choice if accepted else "FULL_QUERY_CONSERVATIVE_FALLBACK",
        "decision_accuracy": None,
        "ground_truth_reviewed": False,
    }
