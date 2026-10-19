"""Bounded local numeric operations; no generated programs or semantic verdicts."""
import math


def _validate_matrix(matrix):
    if (not isinstance(matrix, list) or not 1 <= len(matrix) <= 4
        or any(not isinstance(row, list) or not 1 <= len(row) <= 4 for row in matrix)
        or len({len(row) for row in matrix}) != 1
        or any(type(x) not in (int, float) or abs(x) > 10 or not math.isfinite(x)
               for row in matrix for x in row)):
        raise ValueError("bounded finite rectangular numeric matrix required")


def row_softmax(matrix):
    """A small finite matrix only; stable softmax independently on each row."""
    _validate_matrix(matrix)
    result = []
    for row in matrix:
        weights = [math.exp(x - max(row)) for x in row]
        total = sum(weights)
        result.append([x / total for x in weights])
    return result


def matrix_multiply(left, right):
    """Bounded typed operands, never a parsed/generated program."""
    _validate_matrix(left)
    _validate_matrix(right)
    if len(left[0]) != len(right):
        raise ValueError("incompatible matrix dimensions")
    return [[sum(a*b for a, b in zip(row, column, strict=True))
             for column in zip(*right, strict=True)] for row in left]


def matrix_transpose(matrix):
    _validate_matrix(matrix)
    return [list(column) for column in zip(*matrix, strict=True)]


def matrix_product_shape(shapes):
    """Bounded declared shapes only; undefined products have no output shape."""
    if (not isinstance(shapes, list) or not 1 <= len(shapes) <= 8
        or any(not isinstance(s, list) or len(s) != 2
               or any(type(n) is not int or not 1 <= n <= 4096 for n in s) for s in shapes)):
        raise ValueError("bounded positive matrix shapes required")
    result = list(shapes[0])
    for index, shape in enumerate(shapes[1:], 1):
        if result[1] != shape[0]:
            return dict(conformable=False, output_shape=None, incompatible_factor_index=index,
                        left_shape=result, right_shape=list(shape))
        result = [result[0], shape[1]]
    return dict(conformable=True, output_shape=result, incompatible_factor_index=None)


def fixed_bilinear_content_probe():
    """Two finite evaluations of x^T A x with the SAME identity matrix.

    Pure arithmetic. Matching this instance to a claim or source is a separate,
    fallible interpretation. No author intent or learned weights are inferred.
    """
    fixed = [[1, 0], [0, 1]]
    cases = []
    for vector in ([[1], [0]], [[2], [0]]):
        score = matrix_multiply(matrix_multiply(matrix_transpose(vector), fixed), vector)[0][0]
        cases.append(dict(column_vector=vector, fixed_matrix=fixed, score=score))
    return dict(operation='fixed_bilinear_content_probe', operation_version='bounded-bilinear-v1',
        formula='x^T A x', cases=cases, same_matrix=True, same_vector_for_both_operands=True,
        scores_equal=cases[0]['score']==cases[1]['score'],
        assumptions='Two-dimensional real column vectors, A=I fixed in BOTH cases; vary only vector content. '
            'If applying to a position-indexed rotary equation, zero relative offset gives identity rotation; '
            'identity projections are an illustrative permitted choice, not actual model weights. Same-position '
            'self comparison uses the SAME content in both operands within each case.',
        limitation='This computation tests content independence at fixed matrix only. It does NOT refute '
            'that positional dependence is relative, nor prove that an ambiguous sentence asserts content independence. '
            'Check the exact subject/qualifiers and source equation before using it. No semantic verdict is computed.',
        semantic_support_certified=False, ground_truth=False, provider_calls=0)


def dense_matmul_work(left_shape, right_shape):
    """Classical dense scalar work, excluding hardware, sparsity and fusion."""
    product = matrix_product_shape([left_shape, right_shape])
    if not product["conformable"]:
        raise ValueError("incompatible matrix dimensions")
    rows, inner, columns = left_shape[0], left_shape[1], right_shape[1]
    return dict(output_shape=product["output_shape"], scalar_multiplications=rows*inner*columns,
                scalar_additions=rows*columns*(inner-1),
                assumptions="Classical dense multiplication only; scalar products counted separately, not elapsed time, tensor-core work or an empirical benchmark.")


def scaled_vector_moments(dimension, component_variance, scale):
    """Centered coordinates; E(sum x_j²) needs no independence assumption."""
    if (type(dimension) is not int or not 1 <= dimension <= 4096
        or type(component_variance) not in (int, float)
        or not 0 <= component_variance <= 16 or not math.isfinite(component_variance)
        or type(scale) not in (int, float) or abs(scale) > 64 or not math.isfinite(scale)):
        raise ValueError("bounded centered-vector moments required")
    output_variance = component_variance*scale*scale
    norm_squared = dimension*output_variance
    return dict(dimension=dimension, input_component_variance=component_variance, scale=scale,
                output_component_variance=output_variance, expected_squared_norm=norm_squared,
                root_mean_squared_norm=math.sqrt(norm_squared),
                assumptions="Zero-mean coordinates of equal declared variance; RMS norm sqrt(E||x||²), not E||x|| or a measured typical norm. No actual initialization or author intent inferred.")


def dimension_power_moments(variance_power, scale_power):
    """Symbolic powers of d>0, using only bounded half-integer arithmetic."""
    if any(type(p) not in (int, float) or abs(p) > 4 or not math.isfinite(p)
           or 2*p != int(2*p) for p in (variance_power, scale_power)):
        raise ValueError("bounded half-integer dimension powers required")
    variance = variance_power + 2*scale_power
    return dict(input_component_variance_power=variance_power, scale_power=scale_power,
                output_component_variance_power=variance,
                expected_squared_norm_power=1+variance,
                root_mean_squared_norm_power=(1+variance)/2,
                assumptions="Zero-mean coordinates with component variance d^v, multiplied by d^s. Then output variance d^(v+2s), E||x||²=d^(1+v+2s). Powers describe this conditional law, not source support, initialization or author intent.")


def projected_attention_work(nq, nk, width, heads, key_width, value_width):
    """Bounded dense matrix-chain arithmetic, not a model/configuration claim."""
    values=(nq,nk,width,heads,key_width,value_width)
    if any(type(v) is not int or not 1<=v<=64 for v in values):
        raise ValueError('bounded positive integer dimensions required')
    shapes=[('query_projection',[nq,width],[width,key_width],heads),
            ('key_projection',[nk,width],[width,key_width],heads),
            ('value_projection',[nk,width],[width,value_width],heads),
            ('scores',[nq,key_width],[key_width,nk],heads),
            ('weighted_values',[nq,nk],[nk,value_width],heads),
            ('output_projection',[nq,heads*value_width],[heads*value_width,width],1)]
    terms=[dict(name=name,left=left,right=right,repeats=repeats,
                multiplications=left[0]*left[1]*right[1]*repeats,
                shape_check=matrix_product_shape([left,right]))
           for name,left,right,repeats in shapes]
    return dict(dimensions=dict(nq=nq,nk=nk,d=width,h=heads,k=key_width,v=value_width),
        terms=terms,total_multiplications=sum(t['multiplications'] for t in terms),
        projection_weight_parameters=heads*(2*width*key_width+width*value_width)+heads*value_width*width,
        concat_shape=[nq,heads*value_width],output_shape=[nq,width],
        symbolic_multiplications='h*(nq*d*k+nk*d*k+nk*d*v+nq*nk*k+nq*nk*v)+nq*h*v*d',
        equal_width_substitution='k=v=d/h: (2*nq+2*nk)*d^2+2*nq*nk*d',
        exclusions='Dense scalar multiplications only, not wall-clock speed or FLOPs; exclude bias, scaling, softmax, normalization, nonlinearities, memory and cached-reuse effects.',
        assumption='Independent row-wise linear projections with equal Q/K output width k, V output width v and an output projection from h*v to d. These dimensions must be supported by the cited architecture before application.',
        semantic_support_certified=False,ground_truth=False,provider_calls=0)


def calculation_probe(operation):
    """Allowlisted fixture, usable by either decision provider; not an answer key."""
    if operation == 'projected_attention_work':
        return dict(operation=operation,operation_version='bounded-projected-chain-v1',
            rectangular_case=projected_attention_work(2,3,12,3,2,4),
            equal_head_width_case=projected_attention_work(2,3,12,3,4,4),
            interpretation='The rectangular case is algebraically conformable while k differs from d/h. It is a hypothetical numeric instance, not evidence that a named library supports it or that changing KV head count changes k.',
            semantic_support_certified=False,ground_truth=False,provider_calls=0)
    if operation == "centered_vector_scaling":
        dimension = 4
        return dict(operation="centered_vector_scaling", operation_version="bounded-moments-v1",
            cases=[dict(before=scaled_vector_moments(dimension, variance, 1),
                        multiplied_by_sqrt_dimension=scaled_vector_moments(dimension, variance, math.sqrt(dimension)),
                        divided_by_sqrt_dimension=scaled_vector_moments(dimension, variance, 1/math.sqrt(dimension)),
                        dimension_power_law=dict(before=dimension_power_moments(power, 0),
                            multiplied_by_sqrt_dimension=dimension_power_moments(power, .5),
                            divided_by_sqrt_dimension=dimension_power_moments(power, -.5)))
                   for variance, power in ((1, 0), (1/dimension, -1), (1/(dimension*dimension), -2))],
            limitation="Hypothetical numeric moments, not an embedding initialization claim, source quotation, intention or quality result. A claimed conditional consequence must match its own variance and scaling direction.",
            semantic_support_certified=False, ground_truth=False, provider_calls=0)
    if operation == "dense_attention_work":
        return dict(operation="dense_attention_work", operation_version="bounded-dense-work-v1",
            dimensions=dict(query_count=2, key_count=3, query_key_width=4, value_width=5, additive_hidden_width=6),
            dot_scores=dense_matmul_work([2, 4], [4, 3]),
            weighted_values=dense_matmul_work([2, 3], [3, 5]),
            additive_query_projection=dense_matmul_work([2, 4], [4, 6]),
            additive_key_projection=dense_matmul_work([3, 4], [4, 6]),
            additive_hidden_score=dense_matmul_work([2*3, 6], [6, 1]),
            additive_pairwise_hidden_additions=2*3*6, additive_nonlinearity_evaluations=2*3*6,
            assumptions="Illustrative standard dense dot and one-hidden-layer additive scoring, with query/key projections precomputed once. Additive projections cost nq*dq*da + nk*dk*da multiplications; hidden scores cost nq*nk*da; pointwise adds/nonlinearities counted separately. Softmax and value aggregation separate. Counts depend on these algorithm/width assumptions; no equal-width or hardware-speed conclusion is automatic.",
            semantic_support_certified=False, ground_truth=False, provider_calls=0)
    if operation == "rectangular_linear_projection":
        inputs = [[1, 0, 1], [0, 1, 0]]
        projection = [[1, 0, 1], [0, 1, 1]]
        projected = matrix_multiply(inputs, matrix_transpose(projection))
        gram = matrix_multiply(projected, matrix_transpose(projected))
        return dict(operation="linear_projection_and_gram", operation_version="bounded-matrix-v1",
            input_matrix=inputs, input_shape=[2, 3], column_projection=projection,
            projection_shape=[2, 3], batch_projection_formula="Q=X W_column^T",
            column_vector_formula="q_i=W_column x_i", projected_matrix=projected,
            gram_formula="S=Q Q^T", gram_matrix=gram,
            diagonal_squared_norms=[sum(x*x for x in row) for row in projected],
            invalid_product_shapes=dict(left=[1, len(inputs[0])], right=[len(projection), len(projection[0])],
                conformable=len(inputs[0]) == len(projection)),
            interpretation="For this column-vector convention the diagonal is ||W_column x_i||^2. The product x_i^T W_column is not conformable here. A row-vector convention changes all transposes consistently. This illustrative computation does not set any actual model width.",
            semantic_support_certified=False, ground_truth=False, provider_calls=0)
    if operation != "row_softmax_symmetric_input":
        raise ValueError("unknown bounded calculation operation")
    scores = [[1, 2], [2, 4]]
    weights = row_softmax(scores)
    return dict(operation="row_softmax", operation_version="bounded-row-softmax-v1",
        input_kind="unnormalized_score_matrix", output_kind="row_normalized_weights",
        input_matrix=scores, output_matrix=weights,
        input_is_symmetric=all(scores[i][j] == scores[j][i] for i in range(2) for j in range(2)),
        output_is_symmetric=abs(weights[0][1] - weights[1][0]) <= 1e-12,
        assumption="Finite two-position row-wise softmax, no mask or extra position term; scores can be X*X^T for X=[1,2]^T and scalar shared projection1.",
        interpretation="One computed instance. A failing instance refutes a universal matching claim; a passing instance never proves universality. Applicability must be checked against the actual claim and cited definitions.",
        semantic_support_certified=False, ground_truth=False, provider_calls=0)
