"""Public conditional moment calculation from bounded model-selected inputs."""
import re
from copy import deepcopy

from jev_rag.agent.math_tools import dimension_power_moments


def requested(query):
    return bool(re.search(r'假设|assumption',query,re.I)
                and re.search(r'乘|缩放|scal|multipl',query,re.I)
                and re.search(r'sqrt|√',query,re.I))


def prepare(schema, example, *, unit_type):
    schema,example=deepcopy(schema),deepcopy(example)
    unit=deepcopy(schema['$defs'][unit_type])
    unit['properties']={k:v for k,v in unit['properties'].items() if k!='text'}
    powers=[i/2 for i in range(-8,9)]
    unit['properties']['moments']=dict(type='object',additionalProperties=False,
        properties=dict(dimension=dict(type='string',enum=['d','d_model','d_k']),
                        variance_power=dict(type='number',enum=powers),scale_power=dict(type='number',enum=powers)),
        required=['dimension','variance_power','scale_power'])
    unit['required']=[k for k in unit['required'] if k!='text']+['moments']
    slots=schema['properties']['units']['properties']
    slots['U02']=unit
    for key in slots:
        if key not in {'U01','U02'}:
            slots[key]=dict(type='null')
    example['units']['U02'].pop('text')
    example['units']['U02']['moments']=dict(dimension='d',variance_power=0,scale_power=0)
    return schema,example


def bind(data, *, char_cap):
    if not isinstance(data,dict) or not isinstance(data.get('units'),dict):
        raise ValueError('public moment calculation requires named units')
    slots=data['units']
    unit=slots.get('U02')
    if any(v is not None for k,v in slots.items() if k not in {'U01','U02'}):
        raise ValueError('public moment calculation uses U01 and U02 only')
    if not isinstance(unit,dict) or set(unit)!={'source_excerpt_ids','addresses','moments'}:
        raise ValueError('closed public moment unit required')
    choice=unit['moments']
    if not isinstance(choice,dict) or set(choice)!={'dimension','variance_power','scale_power'}:
        raise ValueError('closed public moment inputs required')
    if not isinstance(choice['dimension'],str) or choice['dimension'] not in {'d','d_model','d_k'}:
        raise ValueError('bounded dimension symbol required')
    for key in ('variance_power','scale_power'):
        value=choice[key]
        if type(value) not in (int,float) or value not in [i/2 for i in range(-8,9)]:
            raise ValueError('bounded half-integer moment powers required')
    receipt=dimension_power_moments(choice['variance_power'],choice['scale_power'])
    def power(p):
        return f"{choice['dimension']}^({p:g})"
    text=(f"条件解释（本地有界算术，并非论文的初始化设定或作者动机）：令维数为{choice['dimension']}，"
          f"假设各分量均值0、方差为{power(choice['variance_power'])}；每分量乘{power(choice['scale_power'])}后，"
          f"每分量方差为{power(receipt['output_component_variance_power'])}，"
          f"E‖向量‖²={power(receipt['expected_squared_norm_power'])}，"
          f"向量均方根范数√E‖向量‖²={power(receipt['root_mean_squared_norm_power'])}；"
          f"每分量均方根为{power(receipt['output_component_variance_power']/2)}。"
          "这是所选假设下缩放的量纲后果，本身不证明与另一信号匹配或有训练收益。")
    if len(text)>char_cap:
        raise ValueError('public moment rendering exceeded capacity')
    cleaned=deepcopy(data)
    cleaned['units']['U02']={k:v for k,v in unit.items() if k!='moments'}|dict(text=text)
    return cleaned,dict(selected_inputs=choice,calculation=receipt,rendered_text=text,
        semantic_support_certified=False,additional_provider_calls=0,actual_initialization_certified=False)


INSTRUCTION=(
    ' PUBLIC CONDITIONAL CALCULATION: U01 gives the requested original-source facts and defines '
    'the dimension, separating them from hypotheses. U02 uses moments instead of text: select '
    'dimension (d,d_model,d_k), variance_power and scale_power from bounded half-integers[-4,4]. '
    'Choose scale_power for the ACTUAL requested operation and explicitly hypothetical input '
    'component variance d^variance_power. Code calculates and displays component variance, '
    'expected squared vector norm, vector RMS norm and component RMS; it does not infer '
    'actual initialization, source intent, signal matching or training effects. Both original '
    'judges see the entire rendered answer. U03/U04 if present MUST be null. Do not add '
    'an unrelated calculation for a different variable. Example powers are syntax only, '
    'not selected assumptions or an answer. These public fields are not private deliberation. '
)
