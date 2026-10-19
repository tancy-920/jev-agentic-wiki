"""Bounded verifier observations; never truth, instructions or authorization."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def literal_assertion_views(text):
    """Exact comma/colon-delimited relation qualifiers with full parent context.

    This is navigation, never a truth rule. Do not split inside mathematical
    parentheses or brackets, and do not paraphrase the proposed assertion.
    """
    stack, start, pieces = [], 0, []
    closing = {"）": "（", ")": "(", "]": "[", "}": "{", "】": "【"}
    for index, char in enumerate(text):
        if char in closing.values():
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
        elif not stack and char in "，：。；;\n":
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    marker = re.compile(r"即|因此|所以|故|从而|意味着|并未|未(?:报告|给出|说明|描述|证明)|"
                        r"不(?:存在|可能|可行|需要|引入|产生)|没有|无需|必然|必须|只能|只需|"
                        r"(?:只|仅)(?:依赖|取决于)|(?:only|solely)\s+depend|"
                        r"(?:乘|除|缩放).*(?:量级|尺度|方差|范数)|"
                        r"(?:scal|multipl|divid).*(?:variance|norm)", re.IGNORECASE)
    def has_top_level_marker(piece):
        depth, outside = [], []
        for char in piece:
            if char in closing.values():
                depth.append(char)
            elif char in closing:
                if depth and depth[-1] == closing[char]:
                    depth.pop()
            elif not depth:
                outside.append(char)
        return marker.search("".join(outside))
    return list(dict.fromkeys(p for p in pieces if p and p != text.strip() and has_top_level_marker(p)))


def evidence_audit_targets(request, *, assertion_views=False):
    """Literal navigation only: never infer support from words or selected IDs."""
    from frontend.answering import literal_clause_views
    from jev_rag.agent.contracts import AnswerCoverageRequest

    available = {c.chunk_id for c in request.evidence}
    if isinstance(request, AnswerCoverageRequest):
        targets = {f"A{i+1:02d}": {"text": claim.text,
                "source_ids": [s for s in claim.citations if s in available]}
                for i, claim in enumerate(request.answer_claims)}
    else:
        targets = {f"c{i}": {"text": text,
            "source_ids": [s for s in request.claim.citations if s in available]}
            for i, text in enumerate(literal_clause_views(request.claim.text))}
    # Exact local views, not inferred propositions. Keep original targets and
    # complete surrounding sentences. Function-call parentheses are not branches.
    additions = {}
    for key, item in targets.items():
        matches = list(re.finditer(r"（([^（）]+)）|(?<![\w])\(([^()]+)\)", item["text"]))
        for index, match in enumerate(matches):
            text = match.group(1) if match.group(1) is not None else match.group(2)
            if not text.strip():
                continue
            additions[f"{key}p{index}"] = {"text": text, "context": item["text"],
                "parent_target": key, "source_ids": list(item["source_ids"])}
    if assertion_views:
        if not isinstance(request, AnswerCoverageRequest):
            # A derivation's hypothesis may precede its consequence clause.
            # Retain the exact SAME source unit, never a different answer unit.
            for item in targets.values():
                item['context'] = request.claim.text
        for key, item in targets.items():
            for index, text in enumerate(literal_assertion_views(item["text"])):
                additions[f"{key}a{index}"] = {"text": text,
                    "context": (item["text"] if isinstance(request, AnswerCoverageRequest) else request.claim.text),
                    "parent_target": key, "source_ids": list(item["source_ids"])}
        for key, item in targets.items():
            formulas = list(dict.fromkeys(m.group() for m in re.finditer(
                r"‖[^‖\n]{1,120}‖(?:²|\^2)?|\|\|[^|\n]{1,120}\|\|(?:²|\^2)?", item["text"])))
            for index, text in enumerate(formulas):
                additions[f"{key}f{index}"] = {"text": text, "kind": "literal_formula",
                    "context": (item["text"] if isinstance(request, AnswerCoverageRequest) else request.claim.text),
                    "parent_target": key, "source_ids": list(item["source_ids"])}
    if len(additions) > 16:
        raise ValueError("too many literal parenthetical audit targets")
    return {**targets, **additions}


def evidence_audit_schema(targets, span_inventory=None):
    from frontend.premise_groups import allowed_premise_ids

    schema = {"type": "object", "additionalProperties": False,
            "properties": {key: {"type": "object", "additionalProperties": False,
                "properties": {
                    "source_ids": {"type": "array", "maxItems": 8,
                        "items": {"type": "string", "enum": item["source_ids"] or [""]}},
                    "assessment": {"type": "string", "pattern": r"^[\s\S]{1,240}$"},
                    "support": {"type": "string", "enum": ["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"]}},
                "required": ["source_ids", "assessment", "support"]}
                for key, item in targets.items()}, "required": list(targets)}
    for key, item in targets.items():
        if item.get("kind") == "literal_formula":
            literal = item["text"]
            schema["properties"][key]["properties"]["assessment"]["pattern"] = "^" + re.escape(literal) + r"[\s\S]{0," + str(240-len(literal)) + "}$"
    if span_inventory is not None:
        for key, item in schema["properties"].items():
            allowed = allowed_premise_ids(targets[key], span_inventory)
            item["properties"].pop("source_ids")
            item["properties"] = {"premise_span_ids": {"type": "array", "maxItems": 8,
                "items": {"type": "string", "enum": allowed or [""]}}, **item["properties"]}
            item["required"] = ["premise_span_ids", "assessment", "support"]
    return schema


def premise_target_wire_views(targets, inventory, *, numeric_context=False):
    """Show exact allowed fragment IDs beside each target, without widening it."""
    from copy import deepcopy

    from frontend.premise_groups import allowed_premise_ids

    result = deepcopy(targets)
    for key, item in result.items():
        item["allowed_premise_span_ids"] = allowed_premise_ids(item, inventory)
        item.pop("source_ids")
        if numeric_context and re.search(
            r"(?:乘|除|缩放).*(?:尺度|范数)|(?:scal|multipl|divid).*norm", item["text"], re.IGNORECASE):
            from jev_rag.agent.math_tools import calculation_probe

            item["local_dimension_power_calculations"] = [
                case["dimension_power_law"] for case in calculation_probe("centered_vector_scaling")["cases"]]
            item["calculation_scope"] = (
                "Compare the actual hypothesis and operation with these conditional powers of d. "
                "This is a calculation, not a support verdict or original source quotation. "
                "An O(1) RMS norm has power0; sqrt(d) power0.5; d power1. "
                "Use the matching variance case; an illustrative label does not validate a mismatched consequence.")
    return result


def bind_audit_premise_spans(values, targets, inventory):
    """Lossless navigation binding only, never infer or alter the verdict."""
    from copy import deepcopy

    if not isinstance(values, dict) or set(values) != set(targets):
        raise ValueError("invalid premise span targets")
    clean, records = deepcopy(values), {}
    for key, item in clean.items():
        if not isinstance(item, dict) or set(item) != {"premise_span_ids", "assessment", "support"}:
            raise ValueError("invalid premise span finding")
        ids = item.pop("premise_span_ids", None)
        if (not isinstance(ids, list) or len(ids) > 8
            or any(not isinstance(i, str) or i not in inventory for i in ids)
            or len(set(ids)) != len(ids)):
            raise ValueError("invalid premise span IDs")
        selected = [inventory[i] for i in ids]
        if any(s.get('premise_group') != targets[key].get('premise_group') for s in selected):
            raise ValueError('premise span citation group mismatch')
        parents = list(dict.fromkeys(s["chunk_id"] for s in selected))
        if any(p not in targets[key]["source_ids"] for p in parents):
            raise ValueError("premise span source binding mismatch")
        item["source_ids"] = parents
        records[key] = dict(selected_ids=ids, exact_original_spans=deepcopy(selected),
                            semantic_support_certified=False)
    return clean, records


def validate_evidence_audit(values, targets, choice):
    if not isinstance(values, dict) or set(values) != set(targets):
        raise ValueError("invalid evidence audit targets")
    for key, item in values.items():
        if not isinstance(item, dict) or set(item) != {"source_ids", "assessment", "support"}:
            raise ValueError("invalid evidence audit fields")
        ids, note, support = item["source_ids"], item["assessment"], item["support"]
        if (not isinstance(ids, list) or len(ids) > 8
            or not all(isinstance(s, str) and s in targets[key]["source_ids"] for s in ids)
            or len(set(ids)) != len(ids) or not isinstance(note, str)
            or not note.strip() or len(note) > 240
            or not isinstance(support, str) or support not in {"SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"}
            or (support == "SUPPORTED" and not ids)
            or (targets[key].get("kind") == "literal_formula" and not note.startswith(targets[key]["text"]))
            or (choice == "SUPPORTED" and support != "SUPPORTED")):
            raise ValueError("invalid evidence audit binding or consistency")
    return values


class VerificationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal[
        "NONE",
        "UNCERTAIN",
        "SOURCE_GAP",
        "CONTRADICTION",
        "MISSING_REQUIREMENT",
        "SCOPE_OR_WORDING",
    ]
    answer_quote: str = Field(max_length=240)
    source_chunk_id: str = Field(max_length=240)
    source_quote: str = Field(max_length=600)
    note: str = Field(max_length=600)


def bind_answer_quote(quote, claims):
    """Bind an exact quote, allowing only omitted separators BETWEEN units.

    Never normalize characters inside a unit, paraphrase, reorder, or infer
    content. Return the actual newline-separated original substring for audit.
    """
    texts = [c.text for c in claims]
    target = "\n".join(texts)
    if quote in target:
        return quote
    joined = "".join(texts)
    starts = [m.start() for m in re.finditer("(?=" + re.escape(quote) + ")", joined)] if quote else []
    if len(starts) != 1:
        return None
    start, stop = starts[0], starts[0] + len(quote)
    offsets, position = [], 0
    for i, text in enumerate(texts):
        offsets.extend(range(position + i, position + i + len(text)))
        position += len(text)
    return target[offsets[start]:offsets[stop - 1] + 1]


def bind_requirement_matches(values, claims):
    """Unit IDs locate exact written text; they never authorize a verdict."""
    units = {f'A{i+1:02d}': claim.text for i, claim in enumerate(claims)}
    result = {}
    for key, value in values.items():
        if (not isinstance(value, dict) or set(value) != {'addressed', 'answer_unit_ids'}
            or type(value['addressed']) is not bool or not isinstance(value['answer_unit_ids'], list)):
            return None
        identifiers = value['answer_unit_ids']
        if (any(not isinstance(i, str) or i not in units for i in identifiers)
            or len(set(identifiers)) != len(identifiers) or (value['addressed'] and not identifiers)):
            return None
        result[key] = [{'answer_unit_id': i, 'exact_text': units[i]} for i in identifiers]
    return result


def bound_issue(value, request):
    issue = VerificationIssue.model_validate(value)
    from jev_rag.agent.contracts import AnswerCoverageRequest

    target = (
        "\n".join(c.text for c in request.answer_claims)
        if isinstance(request, AnswerCoverageRequest)
        else request.claim.text
        if request.claim
        else ""
    )
    if not isinstance(target, str):
        target = "\n".join(c.text for c in target.claims)
    if (
        issue.answer_quote
        and issue.answer_quote not in target
        and not (issue.kind == "MISSING_REQUIREMENT" and issue.answer_quote in request.query)
    ):
        return None
    sources = {c.chunk_id: c.text for c in request.evidence}
    mapped_quote = None
    if issue.source_chunk_id or issue.source_quote:
        if not issue.source_quote.strip() or issue.source_chunk_id not in sources:
            return None
        source = sources[issue.source_chunk_id]
        if issue.source_quote not in source:
            words = re.findall(r"\S+", issue.source_quote)
            if not words:
                return None
            pattern = "(?=(" + r"\s+".join(re.escape(word) for word in words) + "))"
            matches = [m.span(1) for m in re.finditer(pattern, source)]
            if len(matches) != 1 or matches[0][1] - matches[0][0] > 1200:
                return None
            start, end = matches[0]
            mapped_quote = {
                "model_source_quote": issue.source_quote,
                "source_quote": source[start:end],
                "source_quote_start": start,
                "source_quote_end": end,
                "source_quote_binding": "unique_whitespace_only_to_exact_original",
            }
    if issue.kind != "NONE" and not issue.note.strip():
        return None
    return {**issue.model_dump(), **(mapped_quote or {})}
