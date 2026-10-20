"""Conservative literal quantity checks and condition attachment navigation.

No inferred variable meanings, equation execution, truth labels or text rewriting.
Only explicitly defined count variables in simple products can block a draft.
Other syntax is left to the existing independent model checks.
"""

import re

_IDENT = r"[A-Za-z][A-Za-z0-9_]*"
_ROLES = {
    "layer": (r"层数|number of layers|layer count", r"每层|单层|per[- ]layer|single layer"),
    "request": (r"batch\s*size|batch|批大小|批次大小|请求数", r"单请求|每个请求|per[- ]request|single request"),
    "head": (r"头数|number of heads|head count", r"每个头|单头|per[- ]head|single head"),
    "device": (r"设备数|设备数量|number of devices|device count", r"每台设备|单设备|per[- ]device|single device"),
}
_PRODUCT = re.compile(rf"(?<![\w])(?:\d+(?:\.\d+)?|{_IDENT})(?:\s*[*×·]\s*(?:\d+(?:\.\d+)?|{_IDENT}))+", re.ASCII)
_UNSUPPORTED = re.compile(r"[+−/÷^]|(?<![A-Za-z0-9_])(?:sum|exp|log)\b|[Σ∑]", re.I)
_NONASSERTION = re.compile(r"不是|不等于|不能|错误|误称|反例|not\b|incorrect\b|wrong\b|must not|rather than|[\"“”‘’`]", re.I)


def _count_definitions(text):
    definitions = {}
    for role, (meaning, _) in _ROLES.items():
        # Explicit equality of a variable's meaning, not title/keyword inference.
        pattern = re.compile(rf"(?<![A-Za-z0-9_])(?P<symbol>{_IDENT})\s*(?:为|表示|代表|是|denotes\s+|is\s+)(?:the\s+)?(?P<meaning>{meaning})(?![A-Za-z])", re.I)
        for match in pattern.finditer(text):
            symbol = match["symbol"]
            definitions.setdefault(symbol, []).append((role, match.group(), match.span()))
    return {symbol: values[0] for symbol, values in definitions.items()
            if len({value[0] for value in values}) == 1}


def quantity_scope_conflicts(text):
    """Find only explicit per-unit labels retaining that unit's count multiplier.

    Definitions and product must occur in the SAME claim. Sums, quotients,
    functions, quoted corrections, ambiguous meanings and undeclared counts are
    not checked. Empty findings mean unexamined/no matched conflict, never valid.
    """
    definitions = _count_definitions(text)
    findings = []
    for sentence in re.finditer(r"[^。；;\n]+", text):
        value = sentence.group()
        if _NONASSERTION.search(value):
            continue
        for product in _PRODUCT.finditer(value):
            # An arithmetic sub-product inside a larger expression is not a
            # complete expression, so cannot establish the alleged scope clash.
            tail = value[product.end():].lstrip()
            if _UNSUPPORTED.search(value[:product.start()] + tail) or tail.startswith(("*", "×", "·", "(", ")")):
                continue
            prefix = value[:product.start()]
            # Restrict the label to this formula's immediately preceding clause.
            label = re.split(r"[,，:：]", prefix)[-1]
            if not re.search(r"(?:为|约为|等于|=|is\s*|equals\s*)$", label.rstrip(), re.I):
                continue
            operands = re.split(r"\s*[*×·]\s*", product.group())
            for symbol in dict.fromkeys(operands):
                if symbol not in definitions:
                    continue
                role, definition, _ = definitions[symbol]
                if not re.search(_ROLES[role][1], label, re.I):
                    continue
                # Explicit singleton assumption is local and must be exact.
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}\s*=\s*1(?:\.0+)?(?![A-Za-z0-9_.])", text):
                    continue
                findings.append({"kind": "per_unit_product_retains_count",
                    "role": role, "count_symbol": symbol, "definition_quote": definition,
                    "formula_quote": product.group(), "scope_quote": label,
                    "answer_quote": value, "start": sentence.start() + product.start(),
                    "end": sentence.start() + product.end(),
                    "policy": "explicit-count-simple-product-same-claim-v1"})
    return findings


def condition_attachments(text):
    """Literal condition/subject/relation spans; no semantic inference or veto."""
    patterns = [
        r"(?P<condition>[^。；;\n]{1,100}?(?:保持不变|固定))\s*[,，]\s*(?P<object>[^，。；;\n]{1,60}?)(?P<connector>才只依赖|才仅依赖|才依赖|只依赖|仅依赖|只取决于)(?P<dependency>[^，。；;\n]{1,80})",
        r"(?P<condition>[^.;\n]{1,100}?\bfixed)\s*,\s*(?P<object>[^,.;\n]{1,60}?)\s+(?P<connector>only depends on|depends only on)(?P<dependency>[^,.;\n]{1,80})",
    ]
    return [{"answer_quote": match.group(), "condition_quote": match["condition"],
             "object_quote": match["object"], "connector_quote": match["connector"],
             "dependency_quote": match["dependency"], "start": match.start(), "end": match.end(),
             "semantic_support_certified": False}
            for pattern in patterns for match in re.finditer(pattern, text, re.I)]


INSTRUCTION = (
    " Literal quantity_scope_conflicts and condition_attachments are local syntax "
    "observations, never source facts or permission. Bind each condition to its "
    "EXACT subject: a factor, its containing expression and the final output are "
    "different objects. A restriction needed when varying the containing expression "
    "must not be called necessary for the factor itself. Read every opening formula "
    "with its own label; a later correct formula cannot repair an earlier wrong "
    "per-unit label. For a necessary condition, distinguish it from one sufficient "
    "scenario. Empty local findings certify nothing; unmatched syntax still needs "
    "source and whole-answer review."
)
