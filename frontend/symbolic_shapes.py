"""Bounded literal matrix declarations and fixed right-multiplication arithmetic.

No expression evaluation, source-support judgment or generated code execution.
"""
import re

from jev_rag.agent.math_tools import matrix_product_shape


def literal_matrix_axis_checks(request):
    claims = getattr(request, 'answer_claims', None) or [request.claim]
    symbol = r'[A-Za-z][A-Za-z0-9_]{0,23}(?:\^[A-Za-z0-9_]{1,12})?'
    atom = r'(?:[1-9][0-9]{0,3}|[A-Za-z][A-Za-z0-9_]{0,23})'
    dimension = rf'{atom}(?:\s*[·*]\s*{atom}){{0,3}}'
    declaration = re.compile(rf'({symbol})\s*∈\s*R\^\{{({dimension})\s*×\s*({dimension})\}}')
    result = []
    for index, claim in enumerate(claims, 1):
        if len(claim.text) > 4096:
            continue
        matches = list(declaration.finditer(claim.text))
        if len(matches)>16:
            continue  # Abstain, do not crop excess declarations.
        seen, conflict = {}, False
        for match in matches:
            name, rows, columns = match.groups()
            if any(int(n)>4096 for n in re.findall(r'(?<![A-Za-z0-9_])[0-9]+', rows+' '+columns)):
                conflict=True
                break
            shape = (re.sub(r'\s+', '', rows), re.sub(r'\s+', '', columns))
            if name in seen and seen[name] != shape:
                conflict=True
                break
            seen[name]=shape
        if conflict:
            continue
        for match in matches:
            name, rows, columns = match.groups()
            result.append(dict(operation='literal_right_matrix_axes', answer_unit=f'A{index:02d}',
                declaration_quote=match.group(), matrix_symbol=name,
                declared_rows=rows, declared_columns=columns,
                contracted_input_axis=rows, resulting_output_axis=columns,
                rule='For right multiplication X M, X last dimension equals M rows; output last dimension equals M columns. Matrix multiplication alone does not additionally require M rows=M columns. Extra model equalities must come from actual assumptions.',
                conditional_rectangular_example=dict(left_shape=[1,2],right_shape=[2,3],
                    calculation=matrix_product_shape([[1,2],[2,3]]),
                    scope='Generic legal rectangular arithmetic only, not a claimed configuration of the cited model.'),
                limitation='Literal declaration may itself be wrong or discussed to reject it. Check original source, multiplication direction and actual assumptions. No support verdict or semantic certificate.',
                provider_calls=0, semantic_support_certified=False, ground_truth=False))
    return result


def matrix_audit_declarations(target):
    """Select only explicit finite equalities involving this matrix's axes.

    A generic requirement sentence does not acquire an algebra obligation merely
    because its surrounding answer unit contains a matrix declaration. Other
    prose still receives the ordinary original-source audit.
    """
    from types import SimpleNamespace

    context = target.get('context', target['text'])
    if not re.search(r'要求|必须|不必|才能|不要求|无需|need|require|necessary', target['text'], re.I):
        return []
    equations = literal_axis_equalities(target['text'])
    if not equations:
        return []
    checks = literal_matrix_axis_checks(SimpleNamespace(claim=SimpleNamespace(text=context)))
    bindings = _axis_equation_bindings(target['text'])
    pairs = [sorted(_canonical_factors(eq[k]) for k in ('left','right')) for eq in bindings]
    paired = [c for c in checks if sorted(_canonical_factors(c[k])
              for k in ('declared_rows','declared_columns')) in pairs]
    if paired:
        return list(dict.fromkeys(c['declaration_quote'] for c in paired))
    sides = {_canonical_factors(side) for eq in bindings for side in (eq['left'], eq['right'])}
    return list(dict.fromkeys(c['declaration_quote'] for c in checks
        if any(_canonical_factors(c[k]) in sides for k in ('declared_rows', 'declared_columns'))))


def matrix_audit_schema(target):
    # An adjacent verbal equality already includes its explicit necessity and
    # negation. Offer complete literal spans instead of asking for a second,
    # freely abbreviated quotation that could omit its written left operand.
    verbal = [b['quote'] for b in _axis_equation_bindings(target['text']) if b['reading']]
    relation = {'type': 'string', 'pattern': r'^[\s\S]{0,160}$'}
    if verbal:
        offered = list(verbal)
        text = target['text']
        if len(text) <= 160:
            offered.append(text)
        for quote in verbal:
            start = text.find(quote)
            left = max((m.end() for m in re.finditer(r'[。；;\n]', text[:start])), default=0)
            end = start+len(quote)
            boundary = re.search(r'[。；;\n]', text[end:])
            right = end+boundary.start() if boundary else len(text)
            span = text[left:right].strip()
            if len(span) <= 160:
                offered.append(span)
        relation = {'type': 'string', 'enum': ['', *dict.fromkeys(offered)]}
    properties = {
        'declaration_quote': {'type': 'string', 'enum': ['', *matrix_audit_declarations(target)]},
        'basis': {'type': 'string', 'enum': ['MULTIPLICATION', 'EXTRA_ASSUMPTION', 'UNRESOLVED']},
        'relation_quote': relation,
        'assumption_quote': {'type': 'string', 'pattern': r'^[\s\S]{0,160}$'},
        'equation_quote': {'type': 'string', 'enum': ['', *literal_axis_equalities(target['text'])]},
    }
    return dict(type='object', additionalProperties=False, properties=properties, required=list(properties))


def literal_axis_equalities(text):
    """Finite literal dimension equalities; no evaluation or inferred propositions."""
    return [item['quote'] for item in _axis_equation_bindings(text)]


def matrix_wire_schema(target):
    """Do not spend a provider choice on one unambiguous written equality."""
    schema=matrix_audit_schema(target)
    if len(literal_axis_equalities(target['text'])) == 1:
        del schema['properties']['equation_quote']
        schema['required'].remove('equation_quote')
    return schema


def validate_matrix_wire_selection(value,target,support):
    schema=matrix_wire_schema(target)
    if not isinstance(value,dict) or set(value)!=set(schema['properties']):
        raise ValueError('invalid matrix wire selection fields')
    equations=literal_axis_equalities(target['text'])
    bound=dict(value)
    if len(equations)==1:
        bound['equation_quote']='' if value.get('basis')=='UNRESOLVED' else equations[0]
    result=validate_matrix_selection(bound,target,support)
    result.update(wire_selection=dict(value),
        equation_binding='unique-target-equality-v1' if len(equations)==1 else 'provider-selected-v1')
    return result


def _axis_equation_bindings(text):
    """Bind written symbols, including one adjacent explicit verbal equality.

    No pronoun resolution, distant antecedents, inferred '=' quotation or algebra
    evaluation. Unsupported prose remains in the ordinary source audit.
    """
    atom = r'(?:[1-9][0-9]{0,3}|[A-Za-z][A-Za-z0-9_]{0,23})'
    dimension = rf'{atom}(?:\s*[·*]\s*{atom}){{0,3}}'
    end = r'(?![A-Za-z0-9_])(?!\s*[+−/*^=·×-])'
    result = [dict(quote=m.group(),left=m.group(1),right=m.group(2),reading=None)
              for m in re.finditer(
                  rf'(?<![A-Za-z0-9_*/^=+−-])({dimension})\s*=\s*({dimension}){end}', text)
              if not re.search(r'[+−/*^=·×-]\s*$', text[:m.start()])]
    for match in re.finditer(
        rf'(?:维度|宽度)(?:是|为)\s*({dimension})\s*[，,]\s*(必须|不必|无需)等于\s*({dimension}){end}', text):
        result.append(dict(quote=match.group(),left=match.group(1),right=match.group(3),
            reading='ASSERTED' if match.group(2)=='必须' else 'REJECTED'))
    return list({item['quote']:item for item in result}.values())


def _canonical_factors(expression):
    return tuple(sorted(re.split(r'[·*]', re.sub(r'\s+', '', expression))))


MATRIX_INSTRUCTION = (
    ' For targets marked matrix_selection, fill matrix_check BEFORE assessment. '
    'Bind declaration_quote to an offered literal declaration and relation_quote to the '
    'actual necessary-condition phrase in THIS target, including its negation. '
    'Do not output reading, direction or constraint categories: code binds the literal '
    'equation operands and parses only the explicitly supported necessity forms. A right product '
    '(n,r)(r,c) yields (n,c); a left product (r,c)(c,n) yields (r,n). Neither product '
    'alone needs r=c. Do not confuse output width with input contraction. '
    'basis MULTIPLICATION means the necessity is claimed from that operation alone. '
    'EXTRA_ASSUMPTION requires an exact additional condition written BEFORE the '
    'relation in the same complete answer-unit context; quote it in assumption_quote. '
    'The additional condition must explicitly contain the SAME literal equation. '
    'A later chosen example cannot justify an earlier operation-level necessity. '
    'Otherwise assumption_quote is empty. basis UNRESOLVED uses all empty quotes '
    'and cannot be SUPPORTED. Code checks consistency of '
    'the selected finite algebra and rejects inconsistent SUPPORTED, never rewrites '
    'a verdict. Selections and assumption interpretation remain fallible; verify '
    'against own cited originals. If this target has exactly one offered equality, '
    'code binds that exact original equality; equation_quote is absent from its '
    'wire schema and MUST NOT be emitted. With multiple offered equalities, '
    'equation_quote must be an offered exact equality '
    'INSIDE relation_quote. Its operands are bound to the selected declaration axes '
    'by code: an equality between matrix rows and columns is not an input-to-row '
    'contraction equality. Do not relabel the written '
    'equality into a corrected input-to-row equality. If no literal equality is '
    'written use empty equation_quote; unresolved keeps it empty too. '
    'Other targets keep their existing schema. '
    'A later configuration where the equality happens to hold proves existence, '
    'not the written operational necessity. If your feedback identifies that '
    'difference, the corresponding audit support and overall choice must reflect '
    'the unsupported implication. SCOPE_OR_WORDING feedback and lower confidence '
    'do not make a false necessary condition SUPPORTED. '
    'Use JSON numbers for numeric fields (syntax only: "confidence":0.5). '
    'Type declarations, attribute notation and markup are not JSON values. '
    'For a single flat prediction there is NO answers/q0 wrapper. Close its '
    'feedback object and prediction root once each, then stop; there is no '
    'additional envelope to close. '
    'An offered equality can be written as adjacent "dimension is r, must equal c". '
    'Quote that complete offered verbal equality verbatim; code binds its actual operands. '
    'For verbal equalities relation_quote selects a complete offered enum span; '
    'never shorten it to only the must-equal suffix. '
    'Do not invent an equals-sign quotation or resolve a distant '
    'or ambiguous antecedent. '
)


def validate_matrix_selection(value, target, support):
    from types import SimpleNamespace

    schema = matrix_audit_schema(target)
    if not isinstance(value, dict) or set(value) != set(schema['properties']):
        raise ValueError('invalid matrix selection fields')
    for key, spec in schema['properties'].items():
        if not isinstance(value[key], str) or ('enum' in spec and value[key] not in spec['enum']):
            raise ValueError('invalid matrix selection value')
    if value['basis'] == 'UNRESOLVED':
        expected = dict(declaration_quote='', basis='UNRESOLVED',
            relation_quote='', assumption_quote='',equation_quote='')
        if value != expected or support == 'SUPPORTED':
            raise ValueError('unresolved matrix selection cannot support target')
        return dict(selection=value.copy(), algebra_consistent=None, semantic_support_certified=False)
    quote = value['relation_quote']
    if not quote.strip() or len(quote)>160 or quote not in target['text']:
        raise ValueError('matrix relation quote does not bind to target')
    context = target.get('context', target['text'])
    declaration = value['declaration_quote']
    if not declaration:
        raise ValueError('matrix declaration required')
    checks = literal_matrix_axis_checks(SimpleNamespace(claim=SimpleNamespace(text=context)))
    shape = next(c for c in checks if c['declaration_quote'] == declaration)
    equation = value['equation_quote']
    equations = literal_axis_equalities(target['text'])
    if equations and (not equation or equation not in quote):
        raise ValueError('matrix equality must bind inside actual relation quote')
    row_column_equality = False
    reading = None
    if equation:
        binding = next(item for item in _axis_equation_bindings(target['text']) if item['quote']==equation)
        sides = [_canonical_factors(binding[k]) for k in ('left','right')]
        axes = [_canonical_factors(shape[k]) for k in ('declared_rows','declared_columns')]
        row_column_equality = sorted(sides) == sorted(axes)
        # Bounded literal relation grammar, not free-text semantic interpretation.
        negative = re.search(r'(?:不要求|不需要|无需|不必)(?:满足)?\s*'+re.escape(equation), quote)
        positive = re.search(r'(?<!不)(?:要求|需要|必须)(?:满足)?\s*'+re.escape(equation), quote)
        if binding['reading']:
            reading = binding['reading']
        elif negative:
            reading = 'REJECTED'
        elif positive and not re.search(r'不|无|未|避免|否定', quote[:positive.start()][-6:]):
            reading = 'ASSERTED'
    assumption = value['assumption_quote']
    if value['basis'] == 'EXTRA_ASSUMPTION':
        prefix = context[:context.find(quote)]
        if (not assumption.strip() or len(assumption)>160 or assumption not in prefix
            or not equation or equation not in assumption):
            raise ValueError('matrix additional assumption must precede relation in own unit')
        consistent = None  # Quote binding does not certify the selected condition's meaning.
    else:
        if assumption:
            raise ValueError('multiplication alone has no additional assumption quote')
        if row_column_equality and reading:
            required = axes[0] == axes[1]
            consistent = required if reading == 'ASSERTED' else not required
        else:
            consistent = None  # No claim to understand an unsupported relation grammar.
            if row_column_equality and support == 'SUPPORTED':
                raise ValueError('unresolved literal matrix necessity cannot support target')
    if support == 'SUPPORTED' and consistent is False:
        raise ValueError('SUPPORTED matrix finding conflicts with selected algebra')
    return dict(selection=value.copy(), algebra_consistent=consistent,
        literal_reading=reading, literal_row_column_equality=row_column_equality,
        declared_rows=shape['declared_rows'], declared_columns=shape['declared_columns'],
        semantic_support_certified=False, ground_truth=False)


def literal_matrix_relation_calculations(target):
    """Expose operation-only algebra, never an original-source support verdict."""
    results = []
    for declaration in matrix_audit_declarations(target):
        for equation in literal_axis_equalities(target['text']):
            selection = dict(declaration_quote=declaration, equation_quote=equation,
                relation_quote=target['text'],basis='MULTIPLICATION',assumption_quote='')
            binding=next(b for b in _axis_equation_bindings(target['text']) if b['quote']==equation)
            if binding['reading']:
                selection['relation_quote']=equation
            elif len(target['text'])>160:
                # Keep the exact necessary-condition phrase, not a truncated quote.
                start=target['text'].find(equation)
                preceding=target['text'][:start]
                boundaries=[preceding.rfind(s) for s in ('，','。','；',';','\n')]
                selection['relation_quote']=target['text'][max(boundaries)+1:start+len(equation)]
            try:
                receipt=validate_matrix_selection(selection,target,'NOT_SUPPORTED')
            except ValueError:
                continue
            if receipt['literal_row_column_equality'] and receipt['literal_reading']:
                results.append(dict(equation_quote=equation,declaration_quote=declaration,
                    literal_reading=receipt['literal_reading'],
                    operation_only_necessity_consistent=receipt['algebra_consistent'],
                    scope='Only finite literal matrix multiplication algebra under the written declaration; additional equalities, actual model configuration and source support are not certified.',
                    provider_calls=0,semantic_support_certified=False))
    return results
