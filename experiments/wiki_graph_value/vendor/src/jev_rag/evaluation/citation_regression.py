"""Source-bound development regression; reviewed paragraphs, unlabeled spans."""

import json
import re
from uuid import UUID

from jev_rag.agent.contracts import ContextualCitationRequest, Draft, TranslatedDecisionRequest
from jev_rag.agent.language import QueryTranslation, digest
from jev_rag.agent.verification import VerificationPolicy, checks_for, validate_draft
from jev_rag.io import read_jsonl, sha256_file
from jev_rag.schema import Chunk


def sentence_partition(claim):
    """Fixed punctuation rule, no paraphrase, labels, model calls or adaptive tuning."""
    spans = re.findall(r".*?[。；]|.+$", claim.text, flags=re.S)
    return {
        **claim.model_dump(),
        "spans": [{"text": text, "citations": claim.citations} for text in spans],
    }


def prepare_regression(workspace, config, study_id):
    identifier = str(UUID(study_id))
    directory = workspace / "results/reports/language_intent" / identifier
    confirmation_path = directory / "human_confirmation.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if (
        confirmation.get("human_attested") is not True
        or confirmation.get("status") != "approved"
        or confirmation.get("split") != "development"
        or confirmation.get("audit_sha256") != sha256_file(directory / "ai_audit.md")
        or confirmation.get("inputs_sha256", {}).get("review")
        != sha256_file(directory / "review.json")
        or confirmation.get("labels_sha256")
        != digest(json.dumps(confirmation["labels"], sort_keys=True, ensure_ascii=False))
    ):
        raise ValueError("exact human-confirmed development review required")
    parent_path = workspace / "results/reports/live_study" / f"{identifier}.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("type") != "LIVE_CHINESE_TRANSLATION_INTENT_DEVELOPMENT_NOT_MAIN_BENCHMARK"
        or confirmation["inputs_sha256"].get("live_study") != sha256_file(parent_path)
        or parent["corpus_sha256"] != sha256_file(workspace / config.paths.manifest)
        or parent["chunks_sha256"]
        != sha256_file(workspace / config.paths.processed / "chunks.jsonl")
    ):
        raise ValueError("review/source receipt binding changed")
    paragraphs = Draft.model_validate(parent["citation_component"]["draft"]).claims
    labels = [r for r in confirmation["labels"] if r["task"] == "citation"]
    if len(labels) != len(paragraphs) or any(
        label["text"] != claim.text
        or label["citations"] != claim.citations
        or label["id"] != f"answer:paragraph{index + 1}"
        or label["label"] not in {"SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"}
        for index, (label, claim) in enumerate(zip(labels, paragraphs))
    ):
        raise ValueError("confirmed labels must exactly match the original paragraphs")
    verification = VerificationPolicy.model_validate(
        {**config.verification.model_dump(), "mode": "paragraph_and_spans"}
    )
    draft = validate_draft({"claims": [sentence_partition(c) for c in paragraphs]}, verification)
    case = parent["cases"][0]
    translation = QueryTranslation.model_validate(case["translation"]).validate_for(
        case["question"], config.translation
    )
    chunks = {
        c["chunk_id"]: Chunk.model_validate(c)
        for c in read_jsonl(workspace / config.paths.processed / "chunks.jsonl")
    }
    requests = []
    for index, paragraph in enumerate(draft.claims):
        if not set(paragraph.citations).issubset(parent["citation_component"]["evidence_ids"]):
            raise ValueError("regression evidence was not originally retrieved")
        for claim, context, span_index in checks_for(paragraph, verification):
            request_type = ContextualCitationRequest if context else TranslatedDecisionRequest
            request = request_type(
                task="citation",
                query=case["question"],
                translated_query=translation.english_query,
                claim=claim,
                evidence=[chunks[c] for c in claim.citations],
                candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"],
                **({"parent_claim": context} if context else {}),
            )
            requests.append(
                {
                    "paragraph_index": index,
                    "scope": "span" if context else "paragraph",
                    "span_index": span_index,
                    "request": request,
                    "request_sha256": digest(request.model_dump_json()),
                }
            )
    return {
        "parent_study_id": identifier,
        "parent_sha256": sha256_file(parent_path),
        "human_confirmation_sha256": sha256_file(confirmation_path),
        "corpus_sha256": parent["corpus_sha256"],
        "chunks_sha256": parent["chunks_sha256"],
        "draft": draft.model_dump(),
        "paragraph_labels": [r["label"] for r in labels],
        "span_labels": None,
        "split": "development",
        "verification_policy": verification.snapshot(),
        "partition_rule": "exact-chinese-sentence-semicolon-v1",
        "requests": requests,
    }


def aggregate_checks(checks):
    """Conservative diagnostic aggregation, not graph/retry or semantic acceptance."""
    if not checks:
        return "WITHHELD_INCOMPLETE"
    for check in checks:
        if check.get("status") != "completed":
            return "WITHHELD_FAILURE"
        if not check.get("accepted"):
            return "WITHHELD_UNCERTAIN"
        if check.get("choice") != "SUPPORTED":
            return check["choice"]
    return "SUPPORTED"


def regression_summary(checks, labels):
    rows = []
    for layer in ("jev", "llm"):
        for index, label in enumerate(labels):
            selected = [r for r in checks if r["layer"] == layer and r["paragraph_index"] == index]
            paragraph = [r for r in selected if r["scope"] == "paragraph"]
            rows.append(
                {
                    "layer": layer,
                    "paragraph_index": index,
                    "reviewed_label": label,
                    "paragraph_outcome": aggregate_checks(paragraph),
                    "paragraph_and_spans_outcome": aggregate_checks(selected),
                    "attempted_checks": len(selected),
                    "accepted_negative": label != "SUPPORTED"
                    and aggregate_checks(selected) == "SUPPORTED",
                    "span_accuracy": None,
                }
            )
    return rows
