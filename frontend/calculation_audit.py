"""Bounded provider-selected arithmetic receipts, never semantic truth labels."""
import math
import re

from jev_rag.agent.math_tools import dimension_power_moments


def needs_moment_audit(target):
    """Navigation to an explicit variance/scaling/norm discussion only."""
    text = target['text']
    context = target.get('context', text)
    return bool(re.search(r'方差|variance', context, re.I)
                and (re.search(r'范数|尺度|norm', text, re.I)
                     or (re.search(r'方差|variance', text, re.I)
                         and re.search(r'后|after|gives|yields|result', text, re.I)))
                and re.search(r'乘|除|缩放|scal|multipl|divid', text, re.I))


def moment_audit_schema():
    powers = [n / 2 for n in range(-8, 9)]
    properties = {
        'reading': {'type': 'string', 'enum': ['ASSERTED', 'REJECTED', 'UNRESOLVED']},
        **{key: {'type': ['number', 'null'], 'enum': [*powers, None]}
           for key in ('variance_power', 'scale_power', 'claimed_power')},
        'quantity': {'type': 'string', 'enum': ['component_variance', 'expected_squared_norm', 'rms_norm', 'unknown']},
        **{key: {'type': 'string', 'pattern': r'^[\s\S]{0,240}$'}
           for key in ('hypothesis_quote', 'operation_quote', 'result_quote')},
    }
    return dict(type='object', additionalProperties=False, properties=properties, required=list(properties))


MOMENT_INSTRUCTION = (
    ' Targets marked moment_selection require a compact calculation object BEFORE assessment. '
    'Select powers from the ACTUAL written hypothesis and operation, not a corrected statement: '
    'component variance d^variance_power and multiplier d^scale_power. Select the claimed '
    'quantity and its power of d (O(1)=0, sqrt(d)=0.5, d=1). Bind hypothesis_quote, '
    'operation_quote and result_quote to exact nonempty substrings of that target or its '
    'complete context. reading ASSERTED means the answer asserts that consequence; REJECTED '
    'means it explicitly denies that consequence. A teaching/conditional label still ASSERTS '
    'its conditional consequence. Code executes the selected bounded operation and rejects '
    'an internally inconsistent SUPPORTED finding; it never changes your verdict. When '
    'premises, quantity or reading cannot be resolved use UNRESOLVED, null powers, unknown '
    'quantity and empty quotes, and mark this target NOT_SUPPORTED. This is a fallible '
    'semantic selection, not proof of source support or an actual initialization claim.'
)


def validate_moment_selection(value, target, support):
    schema = moment_audit_schema()
    if not isinstance(value, dict) or set(value) != set(schema['properties']):
        raise ValueError('invalid moment selection fields')
    reading = value['reading']
    powers = [value[k] for k in ('variance_power', 'scale_power', 'claimed_power')]
    quotes = [value[k] for k in ('hypothesis_quote', 'operation_quote', 'result_quote')]
    if reading == 'UNRESOLVED':
        if powers != [None, None, None] or value['quantity'] != 'unknown' or quotes != ['', '', ''] or support == 'SUPPORTED':
            raise ValueError('unresolved moment selection cannot certify a supported target')
        return dict(selection=value.copy(), calculation=None, arithmetic_consistent=None,
                    semantic_support_certified=False, ground_truth=False)
    if not isinstance(reading, str) or reading not in {'ASSERTED', 'REJECTED'}:
        raise ValueError('invalid moment reading')
    if any(type(p) not in (int, float) or abs(p) > 4 or not math.isfinite(p) or p * 2 != int(p * 2) for p in powers):
        raise ValueError('invalid bounded moment powers')
    context = target.get('context', target['text'])
    if any(not isinstance(q, str) or not q.strip() or len(q) > 240
           or (q not in context and q not in target['text']) for q in quotes):
        raise ValueError('moment quote does not bind to actual answer context')
    law = dimension_power_moments(powers[0], powers[1])
    quantities = {'component_variance': 'output_component_variance_power',
                  'expected_squared_norm': 'expected_squared_norm_power',
                  'rms_norm': 'root_mean_squared_norm_power'}
    if not isinstance(value['quantity'], str) or value['quantity'] not in quantities:
        raise ValueError('invalid moment quantity')
    equal = law[quantities[value['quantity']]] == powers[2]
    consistent = equal if reading == 'ASSERTED' else not equal
    receipt = dict(selection=value.copy(), calculation=law, arithmetic_consistent=consistent,
                   semantic_support_certified=False, ground_truth=False)
    if support == 'SUPPORTED' and not consistent:
        raise ValueError('SUPPORTED moment finding conflicts with its selected arithmetic')
    return receipt
