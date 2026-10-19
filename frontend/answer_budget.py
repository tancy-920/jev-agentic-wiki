"""Opt-in preparation routing; never a support verdict or permission to skip checks."""

from jev_rag.agent.contracts import Decision, DecisionRequest, ProviderError

VERSION = "answer-budget-v3-optional-call-reservation"


def select_budget(query, evidence, *, intent=None, provider=None, force_deep=False,
                  available_calls=None, expected_source=None, threshold=0.85):
    trace = {"version": VERSION, "selected": "STANDARD", "provider": "local", "fallback": None,
             "verification": "source_batch_and_independent_whole", "threshold": threshold}
    if force_deep or intent == "MULTI_HOP":
        trace["selected"] = "DEEP"
        trace["reason"] = "explicit_complexity"
        return trace
    if not evidence:
        trace["fallback"] = "empty_evidence"
        return trace
    if intent != "FACT_LOOKUP":
        # Explanations cannot use LIGHT. Explicit complex requests were handled
        # above; do not pay to rediscover STANDARD for ordinary explanations.
        trace["reason"] = "intent_requires_standard"
        return trace
    if provider is None:
        # Local rules do not certify evidence clarity, so cannot authorize LIGHT.
        trace["reason"] = "local_uncertainty"
        return trace
    if available_calls is not None and available_calls < 4:
        # The optional classifier may choose STANDARD. Reserve generation and
        # both final checks before spending on a route that cannot ensure LIGHT.
        trace["reason"] = "reserve_generation_and_checks"
        trace["fallback"] = "insufficient_budget_for_optional_classifier"
        return trace
    request = DecisionRequest(task="answer_budget", query=query,
                              candidates=["LIGHT", "STANDARD", "DEEP"], evidence=list(evidence)[:3])
    trace["provider"] = "jev"
    try:
        prediction = Decision.model_validate(provider.decide(request))
        trace["prediction"] = prediction.model_dump()
        confidence = prediction.confidence
        probability = prediction.probability
        if expected_source is not None and prediction.source != expected_source:
            trace["fallback"] = "prediction_source_mismatch"
        elif (prediction.choice not in request.candidates or confidence is None or confidence < threshold
            or probability is None or probability < threshold):
            trace["fallback"] = "invalid_or_uncertain_prediction"
        elif prediction.choice == "LIGHT" and intent != "FACT_LOOKUP":
            trace["fallback"] = "light_requires_fact_lookup"
        else:
            trace["selected"] = prediction.choice
    except (ProviderError, ValueError):
        trace["fallback"] = "provider_failure"
    return trace
