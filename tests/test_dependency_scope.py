"""MOCK cross-topic dependency contracts; no model quality claims or network."""
import json

import httpx
import pytest

from frontend.answering import FocusedAnswerPolicy, FocusedGenerator, application_judge_payload
from jev_rag.agent.contracts import AnswerCoverageRequest, DecisionRequest
from jev_rag.agent.intent import RetrievalPlan
from jev_rag.config import LLMConfig
from jev_rag.llm.adapter import CompatibleAdapter
from jev_rag.llm.policy import ModelPolicy
from jev_rag.schema import Chunk


@pytest.mark.parametrize("contract", ["ordinary", "product", "compact"])
@pytest.mark.parametrize("question,source,claim", [
    ("MOCK rotary dependence?", "MOCK s=q^T R_delta k; fix q,k to isolate positional dependence.",
     "MOCK score depends only on distance."),
    ("MOCK softmax dependence?", "MOCK softmax: a_i=exp(z_i)/sum_j exp(z_j).",
     "MOCK weight depends only on its own logit."),
    ("MOCK cache capacity dependence?", "MOCK bytes=2*L*B*T*H*d*s, all operands may vary.",
     "MOCK cache size depends only on length."),
])
def test_mock_shared_dependency_contract_reaches_generation_and_both_checks(question, source, claim, contract):
    chunk = Chunk(chunk_id="MOCK-source", paper_id="MOCK-paper", title="MOCK", authors=[], year=2024,
                  section="MOCK", source_url="", text=source, token_count=20)
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "output": [{
            "type": "message", "content": [{"type": "output_text", "text": json.dumps({
                "claims": [{"text": claim, "citations": [chunk.chunk_id], "addresses": [question]}]})}]}]})

    config = LLMConfig(api_format="responses")
    gen = FocusedGenerator(CompatibleAdapter(config, transport=httpx.MockTransport(handler)),
                           ModelPolicy(config), FocusedAnswerPolicy(language="zh-CN"))
    gen.product_name_spacing = contract != "ordinary"
    gen.product_compact_contract = contract == "compact"
    focus = RetrievalPlan(intent="CONCEPT_EXPLANATION", goal=question,
                          evidence_needs=[question], queries=[question], user_constraints=[question])
    draft = gen.generate_focused(question, [chunk], focus)
    # A proposal is not silently rewritten, accepted, or certified by the contract.
    assert draft.claims[0].text == claim
    citation = DecisionRequest(task="citation", query=question,
        candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"], claim=draft.claims[0], evidence=[chunk])
    whole = AnswerCoverageRequest(task="citation", query=question,
        candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"], answer_claims=draft.claims, evidence=[chunk])
    source_payload = application_judge_payload([citation])
    whole_payload = application_judge_payload([whole])
    for wire in (sent[0], source_payload, whole_payload):
        serialized = json.dumps(wire)
        assert "DEPENDENCY SCOPE" in serialized
        assert "held fixed" in serialized and "omitted operand" in serialized
        assert "opening conclusion" in serialized
        assert "per layer" in serialized
    assert claim in json.dumps(source_payload) and claim in json.dumps(whole_payload)
    assert len(sent) == 1
    if contract != "ordinary" and ("softmax" in source or "*" in source):
        witnesses = json.loads(sent[0]["input"])["local_arithmetic_witnesses"]
        assert witnesses and all(not w["acceptance_certified"] for w in witnesses)


