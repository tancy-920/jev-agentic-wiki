"""Bounded verifier observations for navigation and honest failure explanations.

An observation never certifies a fact or authorizes a tool. Only original-query
quotes may be shown; rejected answer prose and free-form verifier notes stay out
of user-facing explanations.
"""

GAP_MESSAGES = {
    "SOURCE_GAP": "核验提示：部分结论缺少支持它的前提或适用条件，需要补充依据。",
    "MISSING_REQUIREMENT": "核验提示：原问题的一部分尚未得到明确回答。",
    "SCOPE_OR_WORDING": "核验提示：回答的适用范围或限定条件还不够明确。",
    "CONTRADICTION": "核验提示：候选表述与所引证据可能不一致，需要进一步核对。",
    "UNCERTAIN": "核验未能确定候选回答是否充分，暂不能认定它正确或错误。",
}


def observed_gaps(records, query):
    """Use only already-bound findings from the current verification attempt."""
    result = []
    for record in records:
        if record.get("application_check_scope") not in {"claim_source", "answer_completion"}:
            continue
        for item in record.get("bounded_verifier_feedback", []):
            issue = item.get("observation", {})
            kind = issue.get("kind")
            if item.get("quote_binding_valid") is not True or kind not in GAP_MESSAGES:
                continue
            quote = issue.get("answer_quote")
            finding = {
                "kind": kind,
                "scope": record["application_check_scope"],
                "question_quote": quote
                if isinstance(quote, str) and quote and quote in query
                else None,
            }
            if finding not in result:
                result.append(finding)
    return result[:6]


def gap_diagnostic(row):
    events = row.get("agent", {}).get("events", [])
    # Last attempt only: an earlier gap may have been fixed by the repair.
    latest = next(
        (e for e in reversed(events) if e.get("node") == "verification_gap_summary"), None
    )
    if latest is None:
        return None
    findings = latest.get("findings", [])
    details = list(
        dict.fromkeys(GAP_MESSAGES[f["kind"]] for f in findings if f.get("kind") in GAP_MESSAGES)
    )
    checks = [e for e in events if e.get("node") in {"citation", "original_query_completion"}]
    uncertain = bool(
        checks
        and checks[-1].get("error") in {"uncertain_decision", "answer_incomplete_or_uncertain"}
        and checks[-1].get("selected_action") == "SUPPORTED"
    )
    if not details:
        details = (
            [GAP_MESSAGES["UNCERTAIN"]]
            if uncertain
            else ["本次核验没有提供可定位的具体缺口，不能据此断言知识库缺少答案。"]
        )
    return {
        "details": details,
        "question_parts": list(
            dict.fromkeys(f["question_quote"] for f in findings if f.get("question_quote"))
        ),
        "uncertain_judgment": uncertain,
        "supplement_attempted": any(
            e.get("node") == "gap_targeted_supplement" and e.get("executed") for e in events
        ),
        "candidate_answer_released": False,
        "verifier_observations_are_fallible": True,
    }
