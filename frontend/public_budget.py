"""Public budget derivations; rendered verbatim, never executed or certified."""
import re
from copy import deepcopy

FIELDS = ('definitions', 'parameter_terms', 'parameter_total', 'operation_terms', 'operation_total', 'exclusions')
LABELS = ('记号', '参数分项', '参数合计', '运算分项', '运算合计', '范围')
# 448 model characters + 32 literal label/separator characters = 480.
FIELD_LIMITS = dict(zip(FIELDS, (80, 90, 45, 145, 55, 33), strict=True))


def requested(query):
    return bool(re.search(r'参数|parameter', query, re.I)
                and re.search(r'计算|运算|comput', query, re.I)
                and re.search(r'预算|budget|推导|derive', query, re.I))


def prepare(schema, example, *, unit_type):
    """U02 holds the requested public derivation rather than an opaque paragraph."""
    schema, example = deepcopy(schema), deepcopy(example)
    base = deepcopy(schema['$defs'][unit_type])
    props = base['properties']
    base['properties'] = {k: v for k, v in props.items() if k != 'text'}
    work = dict(type='object', additionalProperties=False,
                properties={key: dict(type='string', minLength=1, maxLength=FIELD_LIMITS[key]) for key in FIELDS},
                required=list(FIELDS))
    base['properties']['work'] = work
    base['required'] = [k for k in base['required'] if k != 'text'] + ['work']
    schema['properties']['units']['properties']['U02'] = base
    for key in schema['properties']['units']['properties']:
        if key not in {'U01','U02'}:
            schema['properties']['units']['properties'][key] = dict(type='null')
    unit = example['units']['U02']
    unit.pop('text')
    unit['work'] = {key: '...' for key in FIELDS}
    return schema, example


def bind(data, *, char_cap):
    """Labels/punctuation only; all model-selected terms still face both judges."""
    if not isinstance(data, dict) or not isinstance(data.get('units'), dict):
        raise ValueError('public budget requires named units')
    if any(v is not None for k,v in data['units'].items() if k not in {'U01','U02'}):
        raise ValueError('public budget uses U01 and U02 only')
    unit = data['units'].get('U02')
    if not isinstance(unit, dict) or set(unit) != {'source_excerpt_ids', 'addresses', 'work'}:
        raise ValueError('U02 requires closed public budget work, sources and addresses')
    work = unit['work']
    if not isinstance(work, dict) or set(work) != set(FIELDS):
        raise ValueError('all public budget derivation fields required')
    for key,value in work.items():
        if not isinstance(value,str) or not value.strip() or len(value)>FIELD_LIMITS[key]:
            raise ValueError(f'public budget {key} requires 1..{FIELD_LIMITS[key]} characters')
    text = '；'.join(label+'：'+work[key] for label, key in zip(LABELS, FIELDS, strict=True))+'。'
    if len(text) > char_cap:
        raise ValueError(f'public budget rendered length {len(text)} exceeds {char_cap}')
    cleaned = deepcopy(data)
    cleaned['units']['U02'] = {k: v for k, v in unit.items() if k != 'work'} | {'text': text}
    return cleaned, dict(public_work=deepcopy(work), rendered_text=text,
                         factual_text_rewritten=False, generated_expressions_executed=False,
                         semantic_support_certified=False, additional_provider_calls=0)


INSTRUCTION = (
    ' PUBLIC BUDGET OUTPUT: For this explicit parameter/computation budget request, U02 is REQUIRED '
    'and uses work instead of text. work has definitions,parameter_terms,parameter_total,'
    'operation_terms,operation_total,exclusions: write the public derivation requested by the user, '
    'including each contributing operation and the summed products before simplification. '
    'U01 uses normal text for the requested conceptual conditions and necessity question; U03/U04 MUST be null. Avoid unrelated examples. All work fields are public answer text, '
    'not private deliberation. Code adds labels only and runs both original judges on the '
    'entire rendered U02. The six fields combined with labels must fit480 characters; '
    'Define short symbols once; do not repeat long dimension names. Field character maxima: definitions80, parameter_terms90, parameter_total45, operation_terms145, operation_total55, exclusions33. Never substitute a qualitative similarity claim for the '
    'requested count or operation sum. No generated expression is executed or approved. '
)
