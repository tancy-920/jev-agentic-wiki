"""Bounded literal quotient counterexamples; arithmetic, never source truth.

Only explicit 'proves x can differ from a/b' and positive integer assignments
in the same complete answer unit are bound. No code/expression evaluation,
table interpretation, implicit source values, or remote referent resolution.
"""
import re

SYMBOL = r'[A-Za-z][A-Za-z0-9_]{0,23}'


def quotient_counterexample_checks(target,query):
    text=target['text']
    context=target.get('context',text)
    if len(query)>4096 or len(context)>4096:
        return []
    relation=re.compile(rf'(?<![A-Za-z0-9_=])((?:{SYMBOL}\s*=\s*){{1,3}})({SYMBOL})\s*/\s*({SYMBOL})(?![A-Za-z0-9_])(?!\s*[*/+^=-])')
    checks=[]
    for match in relation.finditer(query):
        lefts=re.findall(rf'({SYMBOL})\s*=',match.group(1))
        numerator,denominator=match.group(2,3)
        for left in lefts:
            refutation=re.search(rf'(?<!不)(?<!不能)(?<!无法)(?:说明|证明)\s*{re.escape(left)}\s*(?:可|可以|能)?(?:偏离|违反)\s*{re.escape(numerator)}\s*/\s*{re.escape(denominator)}(?![A-Za-z0-9_])(?!\s*[*/+^=-])',text)
            if not refutation:
                continue
            prefix=text[:refutation.start()]
            boundary=max((prefix.rfind(c) for c in '，,。；;：:\n'),default=-1)
            if re.search(r'不|无|未|没|否',prefix[boundary+1:]):
                continue  # A negated proof claim is not an asserted counterexample.
            values={}
            quotes=[]
            assignment=re.compile(rf'(?<![A-Za-z0-9_=])((?:{SYMBOL}\s*=\s*){{1,4}})([1-9][0-9]{{0,3}})(?![A-Za-z0-9_.])(?!\s*[*/+^=-])')
            matches=list(assignment.finditer(context))
            if len(matches)>16:
                continue
            for item in matches:
                number=int(item.group(2))
                if number>4096:
                    continue
                names=re.findall(rf'({SYMBOL})\s*=',item.group(1))
                for name in names:
                    values.setdefault(name,set()).add(number)
                if any(n in (left,numerator,denominator) for n in names):
                    quotes.append(item.group())
            required=[left,numerator,denominator]
            bound={n:next(iter(values[n])) if len(values.get(n,()))==1 else None for n in required}
            resolved=all(v is not None for v in bound.values())
            differs=(bound[left]*bound[denominator]!=bound[numerator]) if resolved else None
            partial_product=(bound[left]*bound[denominator]
                if bound[left] is not None and bound[denominator] is not None else None)
            checks.append(dict(operation='literal_quotient_counterexample',query_equation_quote=match.group(),
                refutation_quote=refutation.group(),assignment_quotes=quotes,explicit_values=bound,
                inequality_violated=differs,
                left_times_denominator=partial_product,
                equality_would_hold_at_reference=partial_product,
                conditional_reference_is_source_fact=False,
                numerator_value=bound[numerator],
                scope='Only explicit positive integer assignments in this same answer unit. Missing/conflicting reference values unresolved; no table/source inference.',
                semantic_support_certified=False,ground_truth=False,provider_calls=0))
    return checks


def validate_quotient_counterexamples(checks,support):
    if support=='SUPPORTED' and any(c['inequality_violated'] is not True for c in checks):
        raise ValueError('SUPPORTED numerical refutation lacks a resolved violated original quotient equality')


COUNTEREXAMPLE_INSTRUCTION=(
    ' Targets with quotient_counterexamples claim that a numerical example disproves '
    'the written quotient equality. Inspect the locally bound original assignments '
    'before judging that claim. inequality_violated=false means the example still '
    'satisfies that equality; it does not prove the claimed departure. null means '
    'a required numeric reference is missing/conflicting in the same answer unit. '
    'Neither can justify SUPPORTED for the written numerical refutation. Do not '
    'silently insert an omitted reference value, swap a table row, change the '
    'refuted variable or replace the example by a correct shape proof mentally. '
    'A shape argument and this particular numerical example are different claims. '
    'Select actual own original premises, report uncertainty and substantive '
    'feedback; the calculation is operation-only, not a source-support verdict. '
)
