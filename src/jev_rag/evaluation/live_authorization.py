"""Explicit local grants and aggregate ledger; context receipts do not grant calls."""

import json
from contextlib import contextmanager
from uuid import UUID

from jev_rag.io import sha256_file


@contextmanager
def authorization_lock(workspace, authorization_id):
    """Refuse concurrent spending; an interrupted run's stale lock is explicit."""
    identifier = str(UUID(authorization_id))
    path = workspace / "results/private/live_authorizations" / f"{identifier}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8"):
        pass
    try:
        yield
    finally:
        path.unlink()


def remaining_authorization(
    workspace, authorization_id, scope, run_cap, max_output_tokens, *, plan_binding_sha256=None
):
    identifier = str(UUID(authorization_id))
    grant_path = workspace / "results/private/live_authorizations" / f"{identifier}.json"
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    # Only the new formal main scope may use a larger explicitly human-approved grant.
    # Historical <=80-call development grants retain their original boundary.
    large_scope = grant.get("scopes") == [scope]
    limit = 10000 if scope == "m5-main" and large_scope else 80
    token_limit = 2048
    if scope == "m6-transformer-interview-development" and grant.get("max_calls", 0) > 80:
        if (not large_scope or grant.get("large_study_explicitly_approved") is not True
            or not isinstance(plan_binding_sha256, str) or len(plan_binding_sha256) != 64
            or any(c not in "0123456789abcdef" for c in plan_binding_sha256)
            or grant.get("plan_binding_sha256") != plan_binding_sha256
            or grant.get("transport_retries") != 0):
            raise ValueError("interview development above80 requires explicit approval, exact plan and zero retries")
        limit = 200
    if scope in {"m6-chinese-relevance", "m6-source-verifier", "m6-basis-diverse"}:
        if (
            not large_scope
            or not isinstance(plan_binding_sha256, str)
            or len(plan_binding_sha256) != 64
            or any(c not in "0123456789abcdef" for c in plan_binding_sha256)
            or grant.get("plan_binding_sha256") != plan_binding_sha256
            or grant.get("transport_retries") != 0
        ):
            raise ValueError(
                "Chinese relevance grant requires one exact plan binding and zero retries"
            )
        limit = 120
        if scope == "m6-basis-diverse":
            limit = 200
            if grant.get("large_study_explicitly_approved") is not True:
                raise ValueError(
                    "broad basis study requires explicit approval above80; standing small-call approval is insufficient"
                )
        if scope == "m6-source-verifier":
            limit, token_limit = 80, 8192
            if (
                grant.get("fast_max_output_tokens") != 2048
                or grant.get("strong_max_output_tokens") != 8192
            ):
                raise ValueError("source verifier requires separately bounded fixed tier caps")
    if scope == "m6-ablation" and large_scope:
        limit, token_limit = 600, 8192
        if (
            grant.get("fast_max_output_tokens") != 2048
            or grant.get("strong_max_output_tokens") != 8192
        ):
            raise ValueError("M6 requires separately approved fixed tier output caps")
    if (
        grant.get("type") != "HUMAN_APPROVED_LIVE_AUTHORIZATION"
        or grant.get("human_attested") is not True
        or grant.get("authorization_id") != identifier
        or not isinstance(grant.get("approval_text"), str)
        or not grant["approval_text"].strip()
        or scope not in grant.get("scopes", [])
        or type(grant.get("max_calls")) is not int
        or not 1 <= grant["max_calls"] <= limit
        or type(run_cap) is not int
        or not 1 <= run_cap <= grant["max_calls"]
        or type(grant.get("max_output_tokens")) is not int
        or not 1 <= max_output_tokens <= grant["max_output_tokens"] <= token_limit
    ):
        raise ValueError("explicit human grant with matching finite scope and limits required")
    used, receipts = 0, []
    for path in sorted((workspace / "results/reports/live_study").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("authorization_id") != identifier:
            continue
        count = receipt.get("used_provider_calls")
        cap = receipt.get("run_cap")
        if (
            receipt.get("mode") != "LIVE"
            or receipt.get("authorization_sha256") != sha256_file(grant_path)
            or type(count) is not int
            or type(cap) is not int
            or not 0 <= count <= cap <= grant["max_calls"]
        ):
            raise ValueError("invalid authorization receipt; budget must not be reset")
        used += count
        receipts.append(path.stem)
    if used + run_cap > grant["max_calls"]:
        raise ValueError("aggregate authorized budget exceeded")
    return {
        "authorization_id": identifier,
        "authorization_sha256": sha256_file(grant_path),
        "authorization_cap": grant["max_calls"],
        "prior_used_provider_calls": used,
        "prior_study_ids": receipts,
    }
