"""Product-only syntax normalization, never prediction/schema-field repair.

Historical benchmark JSON parsing remains strict and unchanged. The complete
body may be one JSON Markdown fence; all its JSON, schema, evidence and semantic
checks still apply. Decision envelopes additionally permit one absent root brace
after all internal values/objects are complete. No prose extraction, eval, retry
or guessed missing fields.
"""

import json
import re

from jev_rag.llm.structured import parse_json


def parse_product_decision(output, record):
    """Only close one absent root brace after all nested objects are complete.

    Never complete a value, string, array or inner object. The caller must still
    validate every question ID, candidate, probability and confidence.
    """
    try:
        return parse_json(output)
    except json.JSONDecodeError as error:
        body = output.strip() if isinstance(output, str) else ""
        # A complete envelope plus exactly one surplus root brace is unambiguous.
        # Strict parsing still rejects duplicate keys, non-finite values and
        # trailing prose/another JSON value; no prediction field is changed.
        if len(body) <= 65536 and error.msg == "Extra data" and body.endswith("}"):
            value = parse_json(body[:-1])
            if not isinstance(value, dict) or set(value) != {"answers"}:
                raise error
            record["product_decision_envelope_recovery"] = {
                "version": "single-surplus-root-brace-v1", "removed": "}",
                "prediction_fields_modified": False, "additional_provider_calls": 0,
                "original_output_preserved": True,
            }
            return value
        if len(body) > 65536 or error.pos != len(output) or not body.endswith("}"):
            raise
        stack, quoted, escaped = [], False, False
        for char in body:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or stack.pop() != ("{" if char == "}" else "["):
                    raise error
        if quoted or stack != ["{"] or not body.startswith("{"):
            raise error
        value = parse_json(body + "}")
        if not isinstance(value, dict) or set(value) != {"answers"}:
            raise error
        record["product_decision_envelope_recovery"] = {
            "version": "single-missing-root-brace-v1", "appended": "}",
            "prediction_fields_modified": False, "additional_provider_calls": 0,
            "original_output_preserved": True,
        }
        return value


def parse_product_json(output, record=None):
    if not isinstance(output, str) or len(output) > 262144:
        raise ValueError("bounded_product_json_required")
    body = output.strip()
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", body, re.IGNORECASE)
    wrapper = "single_json_fence" if match else "none"
    if match:
        body = match[1]
    if record is not None:
        record["product_json_normalization"] = {
            "version": "complete-single-json-fence-v1",
            "wrapper": wrapper,
            "content_rewritten": False,
            "additional_provider_calls": 0,
        }
    return parse_json(body)


def relocate_requirement_matches(data, record):
    """Move an explicitly supplied field from feedback, without guessing values.

    Product-only wire placement normalization. All ID/quote/verdict/schema checks
    still run. Two locations are ambiguous even if equal, and are rejected.
    """
    for key, item in data.get("answers", {}).items():
        if not isinstance(item, dict) or not isinstance(item.get("feedback"), dict):
            continue
        feedback = item["feedback"]
        if "requirement_matches" not in feedback:
            continue
        if "requirement_matches" in item:
            raise ValueError("ambiguous_requirement_matches_placement")
        item["requirement_matches"] = feedback.pop("requirement_matches")
        record.setdefault("product_field_relocations", []).append({
            "question_id": key, "field": "requirement_matches",
            "from": "feedback", "to": "prediction", "values_modified": False,
            "additional_provider_calls": 0,
        })
    return data


def remove_null_plan_annotation(data, record):
    """Drop only allowlisted empty nonoperative annotations before validation.

    Preserve the original wire output. No required field, query, enum, number,
    nonempty annotation or arbitrary extra key is filled, changed or accepted.
    The unique-member parser MUST run first; this is not syntax repair.
    """
    plan = data.get("plan") if isinstance(data, dict) else None
    allowed = ("queries_note", "evidence_needs_note")
    removed = [key for key in allowed if isinstance(plan, dict) and key in plan
               and (plan[key] is None or type(plan[key]) is str and plan[key] == "")]
    if removed:
        original = {key: plan[key] for key in removed}
        plan = {key: value for key, value in plan.items() if key not in removed}
        data = {**data, "plan": plan}
        record["planner_annotation_normalization"] = {
            "policy": "allowlisted-empty-plan-annotations-v2",
            "removed": [{"path": ["plan", key], "original_value": original[key]} for key in removed],
            "core_fields_changed": False, "additional_provider_calls": 0,
        }
    return data
