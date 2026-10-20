"""MOCK literal math-scope guards; no provider or model-quality claims."""

import json

import httpx
import pytest

from frontend.answering import FocusedAnswerPolicy, FocusedGenerator
from frontend.backend import ApplicationLLMDecisions
from frontend.quantity_scope import condition_attachments, quantity_scope_conflicts
from jev_rag.agent.contracts import AnswerCoverageRequest, Claim, DecisionRequest
from jev_rag.config import LLMConfig
from jev_rag.llm.adapter import CompatibleAdapter
from jev_rag.llm.policy import ModelPolicy
from jev_rag.schema import Chunk


@pytest.mark.parametrize("text,role", [
    ("N为层数。每层成本为 2*N*d。", "layer"),
    ("单请求字节约为 2*B*T*d，其中B为batch size。", "request"),
    ("每个头开销为 H*d，其中H为头数。", "head"),
    ("单设备内存为 P×m，其中P为设备数。", "device"),
    ("Per-layer cost is 2*N*d, where N is the number of layers.", "layer"),
    ("每层缓存张量字节约为 2*L*B*T*H_kv*d_head*s，其中L为层数，B为batch。", "layer"),
])
def test_explicit_count_conflict_is_bound_to_literal_same_claim(text, role):
    findings = quantity_scope_conflicts(text)
    assert len(findings) == 1 and findings[0]["role"] == role
    item = findings[0]
    assert text[item["start"]:item["end"]] == item["formula_quote"]
    assert item["definition_quote"] in text and item["scope_quote"] in text


@pytest.mark.parametrize("text", [
    "N为层数。全部层总成本为 2*N*d。",
    "N为层数。每层成本为 2*d。",
    "N为层数，N=1。每层成本为 2*N*d。",
    "N为层数，N=1.0。每层成本为 2*N*d。",
    "N为层数。每层成本为 2*N*d/N。",
    "N为层数。每层成本为 2*N*d + x。",
    "N为层数。每层成本为 F(2*N*d)。",
    "每层成本为 2*N*d。",  # Undefined variables are never guessed from letters.
    "N为头数。每层成本为 2*N*d。",  # Other axis: perfectly possible.
    "N为层数。不能把每层成本称为 2*N*d。",
    "N为层数。‘每层成本为 2*N*d’是错误写法。",
    "N为层数。每层成本为 2*N*d，这是错误写法。",
    "N为层数，N为头数。每层成本为 2*N*d。",  # Ambiguous definition.
])
def test_unmatched_ambiguous_correct_and_nonasserted_syntax_is_not_blocked(text):
    assert quantity_scope_conflicts(text) == []


@pytest.mark.parametrize("text", [
    "q与k固定，位置因子才只依赖n−m。",
    "x保持不变，归一化结果只依赖其他操作数。",
    "With x fixed, the factor only depends on y.",
])
def test_condition_attachment_preserves_exact_object_and_condition_without_veto(text):
    [item] = condition_attachments(text)
    assert text[item["start"]:item["end"]] == item["answer_quote"]
    assert item["condition_quote"] in text and item["object_quote"] in text
    assert not item["semantic_support_certified"]
    assert quantity_scope_conflicts(text) == []


def source():
    return Chunk(chunk_id="MOCK-source", paper_id="MOCK-paper", title="MOCK", authors=[], year=2024,
                 section="MOCK", source_url="", text="MOCK total cost=2*N*d; N is the number of layers.", token_count=20)


@pytest.mark.parametrize("whole", [False, True])
def test_both_judge_payloads_keep_literal_condition_subjects(whole):
    config = LLMConfig()
    judge = ApplicationLLMDecisions(CompatibleAdapter(config), ModelPolicy(config),
        strong_completion=True, verification_feedback_enabled=True, semantic_scope_audit=True)
    claim = Claim(text="MOCK x固定，因子F才只依赖y。", citations=[source().chunk_id])
    common = dict(task="citation", query="MOCK?", claim=claim, evidence=[source()],
                  candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"])
    request = AnswerCoverageRequest(**common, answer_claims=[claim]) if whole else DecisionRequest(**common)
    payload = judge.payload_builder([request])
    [observation] = payload["state"]["requests"]["q0"]["literal_scope_observations"]
    assert observation["condition_attachments"][0]["object_quote"] == "因子F"
    assert not observation["semantic_support_certified"]
    assert not judge.adapter.records  # Building literal views consumes no provider requests.


@pytest.mark.parametrize("repair_ok", [False, True])
def test_actual_product_path_blocks_scope_error_and_shares_one_complete_repair(repair_ok):
    from frontend.conversation import ConversationPlan, TranslationDraft
    from frontend.retrieval import OverviewIndex, ProductResearchAgent
    from jev_rag.agent.intent import IntentPolicy, RetrievalPlan
    from jev_rag.agent.mock import MockDecisions

    query = "Explain the MOCK layer cost"
    wrong = "每层成本为 2*N*d，其中N为层数。"
    correct = "全部层总成本为 2*N*d，单层成本为2*d，其中N为层数。"
    wires = []

    def handler(request):
        wires.append(json.loads(request.content))
        text = correct if repair_ok and len(wires) == 2 else wrong
        value = {"claims": [{"text": text, "citations": [source().chunk_id], "addresses": [query]}]}
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message",
            "content": [{"type": "output_text", "text": json.dumps(value)}]}]})

    config = LLMConfig(api_format="responses")
    gen = FocusedGenerator(CompatibleAdapter(config, transport=httpx.MockTransport(handler)),
                           ModelPolicy(config), FocusedAnswerPolicy())
    gen.product_name_spacing = True
    plan = RetrievalPlan(intent="CONCEPT_EXPLANATION", goal=query, evidence_needs=[query],
                         queries=[query], user_constraints=[query])

    def planning(query, policy):
        result = ConversationPlan(standalone_question=query, action="RETRIEVE", plan=plan,
                                  retrieval_translation=TranslationDraft(english_query=query))
        gen.conversation_resolution = result
        return result

    gen.plan_conversation = planning

    class Retriever:
        def search(self, query):
            return [{"chunk": source().model_dump()}]

    decisions = MockDecisions()
    state = ProductResearchAgent(Retriever(), decisions, gen, overview_index=OverviewIndex([]),
        local_source_only=True, intent_policy=IntentPolicy(enabled=True), threshold=.75).run(query)
    assert len(wires) == 2 and state["citation_repair_count"] == 1 and state["retry_count"] == 1
    first = gen.adapter.records[0]
    assert first["application_validation_errors"][0]["type"] == "literal_quantity_scope_conflict"
    assert first["literal_quantity_scope_audit"]["units"][0]["conflicts"]
    repair_input = json.loads(wires[1]["input"])
    assert repair_input["previous_draft"]["claims"][0]["text"] == wrong
    assert "LOCAL_QUANTITY_SCOPE_CONFLICT" in wires[1]["input"]
    if repair_ok:
        assert state["termination_reason"] == "answered"
        assert len(decisions.requests) == 2  # Fresh source and whole MOCK checks.
        assert state["draft_history"][0]["claims"][0]["text"] == correct
    else:
        assert state["termination_reason"] == "generation_failure"
        assert not decisions.requests and not state["draft_history"]
