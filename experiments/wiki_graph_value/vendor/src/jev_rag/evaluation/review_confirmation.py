"""Human approval binds exact reviewed content; provenance is not review status."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone


def confirmed_review(*, audit_sha256, inputs, labels, approval_text, confirm_human=False):
    if confirm_human is not True or not isinstance(approval_text, str) or not approval_text.strip():
        raise ValueError("explicit human confirmation and approval text required")
    hashes = [audit_sha256, *inputs.values()]
    if not inputs or any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
        for h in hashes
    ):
        raise ValueError("exact reviewed content and input hashes required")
    if not labels or len({label["id"] for label in labels}) != len(labels):
        raise ValueError("unique reviewed labels required")
    label_hash = hashlib.sha256(
        json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "type": "HUMAN_REVIEW_CONFIRMED",
        "status": "approved",
        "reviewer": "human user, explicit chat acceptance",
        "human_attested": True,
        "label_origin": "AI_DRAFT",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_text": approval_text,
        "audit_sha256": audit_sha256,
        "inputs_sha256": deepcopy(inputs),
        "labels_sha256": label_hash,
        "labels": deepcopy(labels),
        "split": "development",
        "independent_annotation": False,
        "held_out_eligible": False,
        "new_live_calls": 0,
    }
