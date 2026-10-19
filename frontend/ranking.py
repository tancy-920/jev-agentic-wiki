"""Bounded app-only rank fusion. Predictions never certify evidence or authorize actions."""

import json

from pydantic import ValidationError

from jev_rag.agent.contracts import Decision, ProviderError
from jev_rag.evaluation.preflight import canonical_hash
from jev_rag.io import sha256_file

RANKING_POLICY = {
    "version": "application-soft-relevance-rrf-v2",
    "weights": [0.0, 0.1, 0.2],
    "rrf_k": 1,
    "auxiliary_ties": "binary labels are tied; relevant rank1, irrelevant rank1+positive count",
    "scope": "short trusted candidate list; does not change dense/BM25 RRF",
    "uncertainty": "keep local position; no second provider",
    "failure": "keep complete local ordering",
    "candidate_removal": False,
    "prediction_is_ground_truth": False,
}


def load_selection(workspace):
    """Source-bound development selection, never final answer acceptance."""
    directory = workspace / "results/processed/rag_revision"
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    review_path = directory / "relevance_review.json"
    delegation_path = directory / "relevance_delegation.json"
    delegation = json.loads(delegation_path.read_text(encoding="utf-8"))
    confirmation_path = directory / "relevance_confirmation.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    try:
        labels = [
            {
                "proposal_id": p["id"],
                "chunk_id": p["chunk"]["chunk_id"],
                "choice": p["proposed_choice"],
            }
            for p in review["proposals"]
        ]
    except (KeyError, TypeError):
        raise ValueError("invalid development review schema") from None
    if (
        selection.get("status") != "reviewed_development_selected"
        or selection.get("selected_weight") not in {0.1, 0.2}
        or selection.get("ranking_policy_sha256") != canonical_hash(RANKING_POLICY)
        or selection.get("chunks_sha256") != sha256_file(workspace / "data/processed/chunks.jsonl")
        or selection.get("review_sha256") != sha256_file(review_path)
        or selection.get("delegation_sha256") != sha256_file(delegation_path)
        or delegation.get("review_sha256") != sha256_file(review_path)
        or delegation.get("user_delegated_ai_review") is not True
        or delegation.get("human_attested") is not False
        or selection.get("confirmation_sha256") != sha256_file(confirmation_path)
        or confirmation.get("type") != "HUMAN_ATTESTED_SOURCE_REVIEW"
        or confirmation.get("review_sha256") != sha256_file(review_path)
        or confirmation.get("human_attested") is not True
        or confirmation.get("ground_truth_release") is not True
        or confirmation.get("split") != "development"
        or not confirmation.get("approval_text", "").strip()
        or confirmation.get("labels") != labels
        or len(labels) != 20
        or len({p["chunk_id"] for p in labels}) != 20
        or selection.get("gate_passed") is not True
    ):
        raise ValueError("development ranking selection missing or source mismatch")
    return selection


def rank_candidates(documents, predictions, *, weight, threshold=0.85):
    if weight not in RANKING_POLICY["weights"] or len(documents) > 20:
        raise ValueError("bounded development weight and <=20 candidates required")
    ids = [d["chunk_id"] for d in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    if len(predictions) != len(documents):
        raise ValueError("prediction/candidate count mismatch")
    parsed = [Decision.model_validate(d) for d in predictions]
    if any(d.choice not in {"RELEVANT", "IRRELEVANT"} for d in parsed):
        raise ValueError("invalid relevance label")
    trusted = [
        i
        for i, d in enumerate(parsed)
        if all(v is not None and v >= threshold for v in (d.probability, d.confidence))
    ]
    # Uncertain entries keep their exact slots. Only trusted entries can move.
    local_rank = {i: n + 1 for n, i in enumerate(trusted)}
    positive_count = sum(parsed[i].choice == "RELEVANT" for i in trusted)
    auxiliary_rank = {
        i: 1 if parsed[i].choice == "RELEVANT" else 1 + positive_count for i in trusted
    }
    k = RANKING_POLICY["rrf_k"]
    ordered = sorted(
        trusted,
        key=lambda i: (
            -((1 - weight) / (k + local_rank[i]) + weight / (k + auxiliary_rank[i])),
            i,
        ),
    )
    result = list(documents)
    for slot, index in zip(trusted, ordered, strict=True):
        result[slot] = documents[index]
    return result, {
        "policy": RANKING_POLICY,
        "weight": weight,
        "threshold": threshold,
        "original_chunk_ids": ids,
        "ranked_chunk_ids": [d["chunk_id"] for d in result],
        "uncertain_positions": [i for i in range(len(ids)) if i not in trusted],
        "predictions": [d.model_dump() for d in parsed],
        "fallback": False,
    }


def predict_and_rank(documents, requests, decisions, *, weight, threshold):
    if not documents:
        return [], {"policy": RANKING_POLICY, "weight": weight, "fallback": False}
    try:
        predictions = (
            decisions.decide_many(requests)
            if hasattr(decisions, "decide_many")
            else [decisions.decide(r) for r in requests]
        )
        if any(Decision.model_validate(d).source != decisions.source for d in predictions):
            raise ValueError("provider source mismatch")
        return rank_candidates(documents, predictions, weight=weight, threshold=threshold)
    except (ProviderError, ValueError, TypeError, ValidationError):
        return list(documents), {
            "policy": RANKING_POLICY,
            "weight": weight,
            "fallback": True,
            "reason": "relevance_provider_invalid_or_failed",
            "original_chunk_ids": [d["chunk_id"] for d in documents],
            "ranked_chunk_ids": [d["chunk_id"] for d in documents],
        }
