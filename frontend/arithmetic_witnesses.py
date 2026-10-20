"""Small fixed arithmetic illustrations; never source evidence or acceptance labels."""

import math
import re

INSTRUCTION = (
    " local_arithmetic_witnesses are deterministic illustrations, not source facts "
    "or claims about a measured model. Use only when the cited operation and stated "
    "conditions match. A counterexample must preserve the allegedly determining "
    "quantity and actually violate the alleged conclusion; unchanged ratios among "
    "other outputs are not proof that the requested output is unchanged. A first-order "
    "zero derivative is not an exact finite-change counterexample. Cite the actual "
    "operation, identify the fixed and changing operands and verify the resulting "
    "aggregate before using a numerical illustration. Do not cite a witness as a "
    "paper or infer empirical performance from it."
)


def arithmetic_witnesses(evidence):
    text = "\n".join(c.text for c in evidence).lower()
    result = []
    if "softmax" in text:
        before, after = [0.0, 0.0, math.log(3)], [0.0, math.log(2), math.log(2)]
        def weight(z):
            return math.exp(z[0]) / math.fsum(math.exp(x) for x in z)
        a, b = weight(before), weight(after)
        result.append(dict(operation="standard_row_softmax", conditions="fixed mask with all three positions visible; fixed temperature1",
                           fixed_target_logit=0, before=before, after=after,
                           target_weight_before=a, target_weight_after=b,
                           target_weights_equal=math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)))
    if re.search(r"[\w)]\s*[*×·]\s*[\w(]", text):
        before, after = [1, 4], [2, 2]
        result.append(dict(operation="product_of_two_factors", conditions="same remaining multiplicative factors",
                           before=before, after=after, product_before=math.prod(before),
                           product_after=math.prod(after), products_equal=math.prod(before) == math.prod(after)))
    return [dict(**item, role="ILLUSTRATIVE_ARITHMETIC_NOT_SOURCE_EVIDENCE",
                 provider_calls=0, acceptance_certified=False) for item in result]
