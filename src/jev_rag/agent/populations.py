"""Finite literal token-population checks; not a general semantic ratio verifier."""

import re
from decimal import Decimal

NUMBER = r"\d{1,3}(?:\.\d{1,3})?"
SELECTION = re.compile(
    rf"\b(?:mask|selects?|chooses?)\s+(?P<percent>{NUMBER})\s*%\s+of\s+"
    r"(?P<population>(?:(?:all|the)\s+)?(?:WordPiece\s+)?(?:input\s+)?"
    r"to(?:-\s*)?kens?(?:\s+positions)?)\b",
    re.I,
)
SUBSET_AS_BASE = re.compile(
    rf"(?:这|该)?\s*(?P<percent>{NUMBER})\s*%\s*(?:是|作为|充当)\s*"
    r"(?:选择|选取|抽样|采样)比例(?:的)?(?:分母|基数)"
)
EXPLICIT_BASE = re.compile(
    rf"(?P<percent>{NUMBER})\s*%\s*(?:的)?(?:分母|基数)\s*"
    r"(?:是|为|指|以)\s*(?P<population>[^。；;\n，,（(]{1,48})"
)
SUBSET = re.compile(r"(?:被|已|所)?选(?:中|取|择)|预测目标|masked\s+tokens?", re.I)
NEGATION = re.compile(r"不|非|否|not\b|rather\b|instead\b", re.I)
PROTOCOL = {
    "version": "literal-token-population-v1.1",
    "source": "actual cited mask/select/choose percentage of input/all/token-position population",
    "claims": "explicit Chinese selection-percentage denominator assertions only",
    "noun_phrase_boundary": "denominator population ends before comma or explanatory parenthesis; comments are not the denominator",
    "ambiguity": "different cited selection rates, unparsed or negated assertions defer to semantic checks",
    "scope": "finite grammar; not a general percentage, entity, semantic or certainty proof",
    "release": "mismatch requires same one complete repair and full verification; no confidence override",
}


def observations(chunk):
    values = []
    for match in SELECTION.finditer(chunk.text):
        percent = Decimal(match["percent"])
        if not 0 < percent < 100:
            continue
        values.append(
            dict(
                chunk_id=chunk.chunk_id,
                start=match.start(),
                end=match.end(),
                source_literal=match.group(),
                percent=str(percent.normalize()),
                population_literal=match["population"],
                population_scope="source_input_tokens",
            )
        )
    return values


def mismatches(draft, evidence):
    by_id = {c.chunk_id: c for c in evidence}
    issues = []
    for index, claim in enumerate(draft.claims):
        cited = [by_id[i] for i in claim.citations if i in by_id]
        source = [item for chunk in cited for item in observations(chunk)]
        rates = {Decimal(item["percent"]) for item in source}
        if len(rates) != 1:
            continue
        rate = next(iter(rates))
        assertions = [
            (m, "selected_fraction_as_selection_denominator")
            for m in SUBSET_AS_BASE.finditer(claim.text)
        ]
        assertions += [
            (m, "selected_population_as_selection_denominator")
            for m in EXPLICIT_BASE.finditer(claim.text)
            if SUBSET.search(m["population"]) and not NEGATION.search(m["population"])
        ]
        for match, kind in assertions:
            clause = re.split(r"[。；;\n]", claim.text[: match.start()])[-1]
            if Decimal(match["percent"]) != rate or NEGATION.search(clause):
                continue
            issues.append(
                dict(
                    claim_index=index,
                    kind=kind,
                    claim_literal=match.group(),
                    claim_start=match.start(),
                    claim_end=match.end(),
                    source_observations=source,
                    semantic_support_certified=False,
                )
            )
    return issues
