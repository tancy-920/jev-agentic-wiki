"""Complete, explicitly non-evidential JSON examples for nested audit syntax."""


def premise_prediction_example(requests, targets, clauses, requirements, *, flat, bind_choice_type=False):
    from frontend.calculation_audit import needs_moment_audit
    from frontend.symbolic_shapes import matrix_audit_declarations, matrix_wire_schema

    predictions = {}
    for index, request in enumerate(requests):
        key = f'q{index}'
        findings = {}
        for name, target in targets.get(key, {}).items():
            finding = dict(premise_span_ids=[])
            if matrix_audit_declarations(target):
                finding['matrix_check'] = dict(declaration_quote='', basis='UNRESOLVED',
                    relation_quote='', assumption_quote='', equation_quote='')
                if 'equation_quote' not in matrix_wire_schema(target)['properties']:
                    del finding['matrix_check']['equation_quote']
            if needs_moment_audit(target):
                finding['calculation'] = dict(reading='UNRESOLVED', variance_power=None,
                    scale_power=None, claimed_power=None, quantity='unknown',
                    hypothesis_quote='', operation_quote='', result_quote='')
            prefix = target['text'] if target.get('kind') == 'literal_formula' else ''
            finding.update(assessment=prefix+' Syntax only.', support='NOT_SUPPORTED')
            findings[name] = finding
        item = dict(type='choice', evidence_audit=findings)
        if key in clauses:
            item['clause_support'] = {f'c{i}': False for i in range(len(clauses[key]))}
        if key in requirements:
            item['requirement_matches'] = {f'r{i}': dict(assessment='Syntax only.',
                addressed=False, answer_unit_ids=[]) for i in range(len(requirements[key][1]))}
        item.update(choice='NOT_SUPPORTED', confidence=1/len(request.candidates),
            probabilities={c: 1/len(request.candidates) for c in request.candidates},
            feedback=dict(kind='NONE', answer_quote='', source_chunk_id='', source_quote='', note=''))
        if bind_choice_type:
            item.pop('type')
        predictions[key] = item
    return predictions['q0'] if flat else dict(answers=predictions)
