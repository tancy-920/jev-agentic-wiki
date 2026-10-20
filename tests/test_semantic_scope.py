"""MOCK finite semantic audit consistency, never model accuracy evidence."""
import json
from itertools import product

import httpx
import pytest

from frontend.arithmetic_witnesses import arithmetic_witnesses
from frontend.backend import ApplicationLLMDecisions
from frontend.semantic_scope import CHECKS, validate_checks
from jev_rag.agent.contracts import AnswerCoverageRequest, Claim, DecisionRequest, ProviderError
from jev_rag.config import LLMConfig
from jev_rag.llm.adapter import CompatibleAdapter
from jev_rag.llm.policy import ModelPolicy
from jev_rag.schema import Chunk


def test_local_arithmetic_witnesses_really_preserve_target_and_aggregate():
    from types import SimpleNamespace

    assert arithmetic_witnesses([SimpleNamespace(text="MOCK definition only")]) == []
    witnesses = arithmetic_witnesses([SimpleNamespace(text="MOCK softmax and total=A*B")])
    assert len(witnesses) == 2
    softmax, product_case = witnesses
    assert softmax["before"][0] == softmax["after"][0]
    assert softmax["before"][1:] != softmax["after"][1:]
    assert softmax["target_weights_equal"]
    assert softmax["target_weight_before"] == pytest.approx(.2)
    assert softmax["target_weight_after"] == pytest.approx(.2)
    assert product_case["before"] != product_case["after"]
    assert product_case["products_equal"]
    assert all(w["provider_calls"] == 0 and not w["acceptance_certified"] for w in witnesses)


@pytest.mark.parametrize("values", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("choice", ["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"])
def test_mock_all_finite_combinations(values, choice):
    checks = dict(zip(CHECKS, values, strict=True))
    if choice == "SUPPORTED" and not all(values):
        with pytest.raises(ValueError, match="invalid_semantic"):
            validate_checks(checks, choice)
    else:
        assert validate_checks(checks, choice) == checks


@pytest.mark.parametrize("bad", [None, {}, {**dict.fromkeys(CHECKS, True), "extra": True},
                                   dict.fromkeys(CHECKS, 1), dict.fromkeys(CHECKS, "true")])
def test_mock_missing_or_coerced_checks_rejected(bad):
    with pytest.raises(ValueError):
        validate_checks(bad, "SUPPORTED")


@pytest.mark.parametrize("choice,checks,valid", [
    ("SUPPORTED", dict.fromkeys(CHECKS, True), True),
    ("SUPPORTED", {**dict.fromkeys(CHECKS, True), "change_is_justified": False}, False),
    ("NOT_SUPPORTED", {**dict.fromkeys(CHECKS, True), "aggregation_scope_matches": False}, True),
    ("SUPPORTED", None, False),
])
@pytest.mark.parametrize("whole", [False, True])
def test_mock_real_judge_wire_cannot_accept_inconsistent_checks(choice, checks, valid, whole):
    wires = []

    def handler(request):
        wires.append(json.loads(request.content))
        item = dict(type="choice", choice=choice, confidence=.9,
                    probabilities={k: .9 if k == choice else .05 for k in
                                   ("SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED")},
                    feedback=dict(kind="NONE", answer_quote="", source_quote="", source_chunk_id="", note=""))
        if checks is not None:
            item["semantic_scope_checks"] = checks
        if whole:
            item["requirement_matches"] = {"r0": {"addressed": True, "answer_unit_ids": ["A01"]}}
        return httpx.Response(200, json={"status": "completed", "output": [{
            "type": "message", "content": [{"type": "output_text", "text": json.dumps({"answers": {"q0": item}})}]}]})

    config = LLMConfig(api_format="responses")
    adapter = CompatibleAdapter(config, transport=httpx.MockTransport(handler))
    judge = ApplicationLLMDecisions(adapter, ModelPolicy(config), strong_completion=True,
                                   verification_feedback_enabled=True, semantic_scope_audit=True)
    source = Chunk(paper_id="MOCK", chunk_id="MOCK-c", title="MOCK", authors=[], year=2024,
                   section="MOCK", text="MOCK formula total=A*B", token_count=10, source_url="")
    request = DecisionRequest(task="citation", query="MOCK scope?", claim=Claim(
        text="MOCK per-item value=A*B", citations=[source.chunk_id]), evidence=[source],
        candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"])
    if whole:
        request = AnswerCoverageRequest(task="citation", query=request.query,
            claim=request.claim, answer_claims=[request.claim], evidence=request.evidence, candidates=request.candidates)
    if valid:
        assert judge.decide(request).choice == choice
    else:
        with pytest.raises(ProviderError):
            judge.decide(request)
    assert len(wires) == 1  # no transport retry or additional judge
    props = wires[0]["text"]["format"]["schema"]["properties"]["answers"]["properties"]["q0"]["properties"]
    assert list(props).index("semantic_scope_checks") < list(props).index("choice")
    assert "Several changed operands may compensate" in wires[0]["instructions"]
    assert adapter.records[-1]["semantic_scope_checks"][0]["ground_truth"] is False
    assert adapter.records[-1]["semantic_scope_checks"][0]["validation_passed"] is valid

