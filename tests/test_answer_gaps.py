"""MOCK shared gap recovery and failure disclosure; never contact providers."""

from types import SimpleNamespace

import pytest

from frontend.answer_gaps import gap_diagnostic, observed_gaps
from frontend.answering import relational_question
from frontend.retrieval import OverviewIndex, ProductResearchAgent
from frontend.view import failure_diagnostic
from jev_rag.agent.contracts import Decision
from jev_rag.agent.intent import IntentPolicy
from jev_rag.agent.mock import MockDecisions, MockGenerator
from jev_rag.schema import Chunk


def observation(
    kind="SOURCE_GAP", quote="MOCK question", valid=True, note="MOCK private verifier note"
):
    return {
        "application_check_scope": "claim_source",
        "bounded_verifier_feedback": [
            {
                "quote_binding_valid": valid,
                "observation": dict(kind=kind, answer_quote=quote, note=note),
            }
        ],
    }


def test_mock_only_bound_current_findings_and_query_quotes_are_disclosed():
    records = [
        observation(),
        observation(valid=False),
        observation(kind="NONE"),
        observation(quote="MOCK rejected assertion"),
    ]
    findings = observed_gaps(records, "MOCK question")
    assert findings == [
        dict(kind="SOURCE_GAP", scope="claim_source", question_quote="MOCK question"),
        dict(kind="SOURCE_GAP", scope="claim_source", question_quote=None),
    ]
    assert "private" not in str(findings) and "rejected assertion" not in str(findings)


@pytest.mark.parametrize(
    "query",
    ["MOCK 缓存可以作用于双向注意力吗？", "MOCK 量化可以用于训练吗？", "MOCK 位置编码能否外推？"],
)
def test_mock_source_gap_gets_one_targeted_supplement_before_repair(query):
    records = []
    plans = []
    searches = []
    c = Chunk(
        chunk_id="MOCK-source",
        paper_id="MOCK-paper",
        title="MOCK",
        authors=[],
        year=2024,
        section="MOCK",
        page=1,
        text="MOCK source premise",
        token_count=3,
        source_url="https://example.com",
    )

    class Judge(MockDecisions):
        count = 0

        def decide_citations(self, requests):
            self.count += 1
            records.append(
                observation(quote=query) if self.count == 1 else observation(kind="NONE")
            )
            return [
                Decision(
                    choice="SUPPORTED",
                    confidence=0.6 if self.count == 1 else 0.95,
                    probability=0.6 if self.count == 1 else 0.95,
                    source="MOCK",
                )
                for _ in requests
            ]

    class Generator(MockGenerator):
        def plan_supplement(self, original, previous, evidence, feedback, policy):
            plans.append(feedback)
            assert original == query
            return previous.model_copy(update={"queries": ["MOCK missing applicability premise"]})

    def search(q):
        searches.append(q)
        return [{"chunk": c.model_dump()}]

    agent = ProductResearchAgent(
        SimpleNamespace(search=search),
        Judge(),
        Generator(),
        overview_index=OverviewIndex([]),
        local_source_only=True,
        intent_policy=IntentPolicy(enabled=True),
        max_iterations=2,
    )
    agent.verification_feedback_records = lambda: records
    state = agent.run(query)
    assert state["termination_reason"] == "answered"
    assert len(plans) == 1 and len(searches) == 2
    assert state["citation_repair_count"] == 1
    assert any("untrusted_verifier_observations" in item for item in plans[0])
    assert sum(e["node"] == "gap_targeted_supplement" for e in state["events"]) == 1
    assert state["iteration"] == 2


@pytest.mark.parametrize(
    "before,iteration,remaining,kind,expected",
    [
        (0, 1, 4, "SOURCE_GAP", True),
        (0, 1, 3, "SOURCE_GAP", False),
        (1, 1, 10, "SOURCE_GAP", False),
        (0, 2, 10, "SOURCE_GAP", False),
        (0, 1, 10, "UNCERTAIN", False),
        (0, 1, 10, "SCOPE_OR_WORDING", False),
        (0, 1, 10, "MISSING_REQUIREMENT", True),
    ],
)
def test_mock_gap_does_not_expand_repair_or_budget(before, iteration, remaining, kind, expected):
    agent = object.__new__(ProductResearchAgent)
    agent.generator = SimpleNamespace(plan_supplement=lambda: None)
    state = dict(
        run_id="MOCK-gap-run",
        mode="MOCK",
        relevant_documents=[],
        query="MOCK question",
        citation_repair_count=1,
        next_node="generate",
        iteration=iteration,
        max_iterations=2,
        verification_feedback=[{"MOCK": "feedback"}],
        events=[],
    )
    agent._handle_verification_gaps(
        state, [observation(kind=kind)], before, SimpleNamespace(remaining=remaining, parent=None)
    )
    assert (state["next_node"] == "rewrite") == expected
    assert state["citation_repair_count"] == 1


def test_mock_latest_attempt_not_stale_gap_determines_final_notice():
    old = dict(
        node="verification_gap_summary",
        findings=[dict(kind="SOURCE_GAP", question_quote="MOCK old")],
    )
    latest = dict(node="verification_gap_summary", findings=[])
    row = dict(
        status="answer_incomplete",
        agent={
            "events": [
                old,
                dict(node="gap_targeted_supplement", executed=True),
                dict(
                    node="original_query_completion",
                    selected_action="SUPPORTED",
                    error="answer_incomplete_or_uncertain",
                ),
                latest,
            ]
        },
    )
    result = failure_diagnostic(row)
    assert result["uncertain_judgment"] is True and result["supplement_attempted"] is True
    assert result["question_parts"] == [] and result["candidate_answer_released"] is False
    assert "SOURCE_GAP" not in str(result) and "MOCK old" not in str(result)
    assert "正确或错误" in result["details"][0]


def test_mock_provider_failure_does_not_claim_evidence_gap():
    row = dict(
        status="provider_failure",
        agent={
            "events": [
                dict(
                    node="verification_gap_summary",
                    findings=[dict(kind="SOURCE_GAP", question_quote="MOCK")],
                )
            ]
        },
    )
    assert "details" not in failure_diagnostic(row)
    assert gap_diagnostic({"agent": {"events": []}}) is None


@pytest.mark.parametrize(
    "query", ["可以应用于双向注意力吗？", "量化能否用于训练？", "适用哪些条件？"]
)
def test_mock_applicability_questions_get_condition_scope(query):
    assert relational_question(query)
