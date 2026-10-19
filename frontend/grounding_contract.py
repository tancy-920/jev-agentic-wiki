"""Shared semantic contract for prose and comparison-cell product generation.

No topic answers or source facts. A schema change must not remove these rules.
"""

SEMANTIC_CONTRACT = (
    ' Scope every factual sentence to its own cited originals. Preserve every equation '
    'argument, operator, negation, quantifier, parameter-sharing axis and execution stage. '
    'Describe a conditional implementation only under its branch condition; it is not '
    'automatically the default architecture. A dependence on one variable does not '
    'eliminate other variables. Distinguish per-head parameters from shared-across-layer '
    'parameters without contradicting either scope. Attribute empirical conclusions '
    'to their actual model, dataset and settings; omit unrequested measurements. '
    'Do not infer intent, performance or a universal guarantee from algebra alone. '
    'For a conditional explanation state the assumed input distribution or other premise, '
    'apply the ACTUAL requested operation, and give its short checkable consequence. '
    'A calculation for a different operation or variable does not explain this one. '
    'For a budget/derivation enumerate ALL contributing operations and sum their products '
    'before simplification; distinguish parameter count from computation and state exclusions. '
    'For statistics name input layout, reduced axes and feature cardinality separately; '
    'distinguish training from evaluation and expectation from variance. '
    'For execution dependencies name current and stored operands, their producers '
    'and update order; separate independent instances from successive dependent stages. '
    'For a counterexample violate the actual requested equality under permitted shapes, '
    'not an unrelated setting. '
    'A general conclusion must remain true for every included case: carry each exception '
    'through all later sentences, or explicitly restrict the conclusion to the affected '
    'subset. A correct list of proportions does not license an unconditional summary '
    'that contradicts one component. Distinguish changes to an input, a selected score '
    'row or column, and a residual path. Never infer that a whole representation is '
    'removed from blocking only one path. Remove unsupported extras, not requested content. '
    'These instructions provide no source facts and do not replace independent checks. '
)
