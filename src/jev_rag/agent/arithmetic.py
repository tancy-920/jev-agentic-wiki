"""Conservative literal-operation guard; not a general mathematical truth verifier.

Only explicit divide/multiply operations on dot products and comparable monomial
factors are checked. PDF subscript spelling is heuristic. Ambiguous scopes,
negations, unsupported expressions and missing extraction remain model checks.
"""

import math
import re
from fractions import Fraction

OPERATION = re.compile(
    r"(?P<div>除以|\bdivid(?:e|es|ed|ing)\b\s+(?:each\s+)?by\s*)|(?P<mul>乘以|\bmultip(?:ly|lies|lied)\b\s+(?:each\s+)?by\s*)",
    re.I,
)
ENTITY = re.compile(r"点积|\bdot[ -]products?\b", re.I)
NEGATION = re.compile(r"不|未|并非|无需|\b(?:not|without|never|don't|doesn't|no)\b", re.I)
ATOM = re.compile(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,31}")
PROTOCOL = {
    "version": "literal-operation-guard-v1",
    "scope": "explicit dot-product divide/multiply, bounded monomial factors only",
    "subscripts": "single-letter underscore subscript normalized to flattened PDF spelling; heuristic",
    "acceptance": "guard only vetoes comparable literal mismatches; no pass certifies semantic correctness",
    "ambiguity": "negations, other subjects, multiple source factors and unsupported algebra defer to full support judge",
    "repair": "same shared complete-repair allowance, never unchecked patching or an extra retry",
}


def factor(text):
    """Parse a tiny monomial grammar; no eval, execution or symbolic program."""
    if len(text) > 120:
        text = text[:120]
    cursor, nodes = 0, 0

    def space():
        nonlocal cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

    def atom(depth=0):
        nonlocal cursor, nodes
        nodes += 1
        if nodes > 12 or depth > 3:
            raise ValueError("bounded factor required")
        space()
        if text.startswith("sqrt", cursor) or text.startswith("√", cursor):
            cursor += 4 if text.startswith("sqrt", cursor) else 1
            coefficient, powers = atom(depth + 1)
            n, d = math.isqrt(coefficient.numerator), math.isqrt(coefficient.denominator)
            if n * n != coefficient.numerator or d * d != coefficient.denominator:
                raise ValueError("irrational numeric coefficient unsupported")
            return Fraction(n, d), {k: v / 2 for k, v in powers.items()}
        if cursor < len(text) and text[cursor] == "(":
            cursor += 1
            value = expression(depth + 1)
            space()
            if cursor >= len(text) or text[cursor] != ")":
                raise ValueError("closed factor required")
            cursor += 1
            return value
        match = ATOM.match(text, cursor)
        if not match:
            raise ValueError("unsupported factor")
        cursor = match.end()
        value = match.group()
        if value[0].isdigit():
            if len(value) > 12:
                raise ValueError("bounded numeric operand")
            return Fraction(value), {}
        # Common PDF flattening loses a subscript separator, never case-folds.
        value = re.sub(r"^([A-Za-z])_([A-Za-z0-9])$", r"\1\2", value)
        return Fraction(1), {value: Fraction(1)}

    def expression(depth=0):
        nonlocal cursor
        coefficient, powers = atom(depth)
        while True:
            space()
            if cursor >= len(text) or text[cursor] not in "*/×":
                return coefficient, powers
            op = text[cursor]
            cursor += 1
            other, terms = atom(depth)
            sign = -1 if op == "/" else 1
            coefficient = coefficient / other if sign == -1 else coefficient * other
            for key, value in terms.items():
                powers[key] = powers.get(key, Fraction(0)) + sign * value

    try:
        coefficient, powers = expression()
        tail = text[cursor:].lstrip()
        if (
            tail.startswith(tuple("+-^%=_([\"'"))
            or (
                tail.startswith(".")
                and len(tail) > 1
                and (tail[1].isascii() and (tail[1].isalnum() or tail[1] == "_"))
            )
            or coefficient <= 0
        ):
            return None
        if (
            tail
            and tail[0].isascii()
            and tail[0].isalnum()
            and not re.match(r"(?:and|to|before|then|for|is|are|was|were)\b", tail, re.I)
        ):
            return None
        return {
            "coefficient": str(coefficient),
            "powers": {k: str(v) for k, v in sorted(powers.items()) if v},
            "literal": text[:cursor].strip(),
        }
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def observations(text):
    output = []
    for match in OPERATION.finditer(text):
        start = match.start()
        prefix = text[max(0, start - 180) : start]
        # Newlines in PDFs are layout, not sentence boundaries.
        clause = re.split(r"[。！？.!?;；]", prefix)[-1]
        if not (
            ENTITY.search(clause) or (not clause.strip() and ENTITY.search(prefix))
        ) or NEGATION.search(clause):
            continue
        parsed = factor(text[match.end() :])
        if parsed is None:
            continue
        sign = -1 if match.group("div") else 1
        coefficient = Fraction(parsed["coefficient"])
        scale = dict(
            coefficient=str(1 / coefficient if sign == -1 else coefficient),
            powers={k: str(sign * Fraction(v)) for k, v in parsed["powers"].items()},
        )
        output.append(
            dict(
                start=start,
                end=match.end()
                + len(text[match.end() :])
                - len(text[match.end() :].lstrip())
                + len(parsed["literal"]),
                operator="divide" if sign == -1 else "multiply",
                factor=parsed,
                scale=scale,
                entity="dot_product",
            )
        )
    return output


def source_operations(evidence):
    return [dict(chunk_id=c.chunk_id, **o) for c in evidence for o in observations(c.text)]


def mismatches(draft, evidence):
    """Veto only a unique comparable source factor; never select an ambiguous gold."""
    source = source_operations(evidence)
    issues = []
    for index, claim in enumerate(draft.claims):
        for actual in observations(claim.text):
            symbols = set(actual["scale"]["powers"])
            expected = [
                s
                for s in source
                if s["chunk_id"] in claim.citations and set(s["scale"]["powers"]) == symbols
            ]
            scales = {repr(s["scale"]) for s in expected}
            if len(scales) == 1 and actual["scale"] != expected[0]["scale"]:
                issues.append(
                    dict(
                        claim_index=index,
                        code="literal_operation_mismatch",
                        actual=actual,
                        source_operations=expected,
                        semantic_gold_certified=False,
                    )
                )
    return issues
