"""Finite, fallible semantic checks inside the existing paid verification call."""

from frontend.quantity_scope import INSTRUCTION as QUANTITY_SCOPE_INSTRUCTION

CHECKS = ("change_is_justified", "aggregation_scope_matches", "conditions_are_consistent")

INSTRUCTION = (
    " Before choice, fill semantic_scope_checks with exactly three booleans. "
    "change_is_justified: a claim that varying inputs MUST change an output needs "
    "more than dependence. Several changed operands may compensate, preserving a "
    "sum, product or normalized ratio. One-factor monotonicity needs other operands "
    "fixed and a stated nondegenerate domain. aggregation_scope_matches: verbal "
    "per-item/per-layer/per-request scope must match sums and multiplicative counts "
    "in the equation; do not call a total formula a per-unit amount unless counts "
    "are explicitly set to one or removed. conditions_are_consistent: read the "
    "opening conclusion and every later exception together; an unconditional "
    "opening is not repaired by a contradicting qualification later. Set a check "
    "false for a real error, true when satisfied or not applicable. An actual "
    "error requires NOT_SUPPORTED or CONTRADICTED plus existing bounded feedback "
    "quoting the exact problematic text; no private reasoning. SUPPORTED requires "
    "all three true but true checks alone do not prove source support or coverage."
    " Any local_arithmetic_witnesses are finite illustrative calculations, not "
    "evidence about a model or truth labels. Check applicability to the cited "
    "operation and mask/domain. A counterexample must preserve the fixed operand "
    "and actually violate the claimed necessity; approximate first-order "
    "cancellation is not an exact finite-change witness."
    + QUANTITY_SCOPE_INSTRUCTION
)


def check_schema():
    return {"type": "object", "additionalProperties": False,
            "properties": {key: {"type": "boolean"} for key in CHECKS},
            "required": list(CHECKS)}


def validate_checks(value, choice):
    valid = (isinstance(value, dict) and set(value) == set(CHECKS)
             and all(type(item) is bool for item in value.values()))
    if not valid or (choice == "SUPPORTED" and not all(value.values())):
        raise ValueError("invalid_semantic_scope_checks")
    return value
