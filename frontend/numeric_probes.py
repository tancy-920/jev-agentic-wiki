"""Optional lexical navigation to a bounded calculation, never an acceptance rule."""
import re

from jev_rag.agent.math_tools import (
    calculation_probe,
    fixed_bilinear_content_probe,
    matrix_product_shape,
)

from .symbolic_shapes import literal_matrix_axis_checks


def literal_norm_dimension_checks(request):
    """Exact norm products under explicit numeric, column-vector declarations.

    Incomplete grammar, conflicting declarations and mixed conventions abstain.
    A formula may be mentioned to reject it: undefined does NOT judge its parent.
    No evaluation of arbitrary expressions, symbolic inference or code execution.
    """
    claims = getattr(request, "answer_claims", None) or [request.claim]
    checks = []
    symbol = r"[A-Za-z](?:_[A-Za-z0-9]+)?"
    declaration = re.compile(rf"({symbol})\s*∈\s*R(?:\^\{{([1-9][0-9]{{0,3}})(?:×([1-9][0-9]{{0,3}}))?\}}|([¹²³⁴⁵⁶⁷⁸⁹]))")
    superscripts = {c: i for i, c in enumerate("¹²³⁴⁵⁶⁷⁸⁹", 1)}
    factor = re.compile(rf"({symbol})(ᵀ|\^T)?")
    for unit, claim in enumerate(claims, 1):
        text = claim.text
        if len(text) > 4096 or "列向量约定" not in text or "行向量" in text:
            continue
        dimensions, quotes, conflict = {}, {}, False
        for match in declaration.finditer(text):
            name, rows, columns, sup = match.groups()
            shape = [superscripts[sup], 1] if sup else [int(rows), int(columns or 1)]
            if max(shape) > 4096 or (name in dimensions and dimensions[name] != shape):
                conflict = True
                break
            dimensions[name], quotes[name] = shape, match.group()
        if conflict:
            continue
        for match in re.finditer(r"‖([^‖\n]{1,120})‖(?:²|\^2)?|\|\|([^|\n]{1,120})\|\|(?:²|\^2)?", text):
            expression = match.group(1) or match.group(2)
            atom = rf"{symbol}(?:ᵀ|\^T)?"
            if not re.fullmatch(rf" *{atom}(?: *[*·]? *{atom})* *", expression):
                continue
            tokens, shapes, position = [], [], 0
            while position < len(expression):
                if expression[position] in " *·":
                    position += 1
                    continue
                token = factor.match(expression, position)
                if not token or token.group(1) not in dimensions or len(tokens) >= 8:
                    break
                name, transpose = token.groups()
                shape = dimensions[name][::-1] if transpose else list(dimensions[name])
                tokens.append(dict(literal=token.group(), shape=shape, declaration=quotes[name]))
                shapes.append(shape)
                position = token.end()
            else:
                if shapes:
                    checks.append(dict(operation="literal_norm_product_dimensions", answer_unit=f"A{unit:02d}",
                        literal_formula=match.group(), factors=tokens, calculation=matrix_product_shape(shapes),
                        scope="Only this unit's explicit numeric column-vector declarations; no inferred model widths.",
                        limitation="Definedness of the written product only. A parent can correctly discuss an undefined expression; this is not a support verdict.",
                        semantic_support_certified=False, ground_truth=False, provider_calls=0))
    return checks


def verification_numeric_probes(request):
    text = request.query + "\n" + request.claim.text
    # Navigation eligibility only. This does not assert what the author meant,
    # that a source supports the claim, or that the computed instance applies.
    checks = [*literal_norm_dimension_checks(request), *literal_matrix_axis_checks(request)]
    if (re.search(r'多头|multi.head|MHA',text,re.I)
        and re.search(r'预算|计算|参数|维度|budget|compute|dimension|parameters',text,re.I)):
        checks.append(calculation_probe('projected_attention_work'))
    if (re.search(r'相对|relative', text, re.I)
        and re.search(r'RoPE|rotary|双线性|bilinear|内积', text, re.I)):
        checks.append(fixed_bilinear_content_probe())
    if (re.search(r"embedding|嵌入|方差|variance", text, re.IGNORECASE)
        and re.search(r"缩放|scal|sqrt|√", text, re.IGNORECASE)):
        checks.append(calculation_probe("centered_vector_scaling"))
    if (re.search(r"加性|additive", text, re.IGNORECASE)
        and re.search(r"计算量|复杂度|complexity|FLOPs?", text, re.IGNORECASE)):
        checks.append(calculation_probe("dense_attention_work"))
    if not re.search(r"对称|symmetr", text, re.IGNORECASE):
        return checks
    if not any("softmax" in c.text.lower() for c in request.evidence):
        return checks
    return [calculation_probe("row_softmax_symmetric_input"),
            calculation_probe("rectangular_linear_projection"), *checks]
