"""Explicit benchmark-live controls, isolated from user-visible answers.

AI source proposals are development diagnostics, never released ground truth.
Only typed technical claims and actual source passages reach the provider.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from frontend.answering import ProductBudgetAdapter
from frontend.backend import ApplicationGate, ApplicationLLMDecisions, application_hash
from frontend.session import application_runtime_lock
from jev_rag.agent.contracts import AnswerCoverageRequest, Claim, DecisionRequest
from jev_rag.evaluation.live_authorization import authorization_lock
from jev_rag.evaluation.preflight import canonical_hash
from jev_rag.io import read_jsonl, write_json
from jev_rag.knowledge import load_product_config, nonpeak_allowed
from jev_rag.llm.credentials import load_credentials
from jev_rag.llm.policy import ModelPolicy
from jev_rag.schema import Chunk


def run_controls(workspace, identifier, cap, path, grant_binding):
    expected = Path(path).resolve()
    feedback_controls = expected.name in {"bounded_feedback_validation_v9.json", "bounded_feedback_validation_v10.json", "bounded_feedback_validation_v11.json"}
    reviewed_roles = feedback_controls or expected.name in {"flash_role_validation_v1.json", "flash_role_validation_v2.json"}
    expected_cap = 4 if feedback_controls else 16 if reviewed_roles else 8
    expected_count = 2 if feedback_controls else 8 if reviewed_roles else 4
    allowed = {workspace / "results/processed/transformer_wiki" / name
               for name in ("bounded_feedback_validation_v9.json", "bounded_feedback_validation_v10.json", "bounded_feedback_validation_v11.json", "parallel_validation_v6.json", "parallel_validation_v7.json", "flash_validation_v1.json", "flash_validation_v2.json", "flash_role_validation_v1.json", "flash_role_validation_v2.json")}
    if expected not in {p.resolve() for p in allowed} or cap != expected_cap:
        raise ValueError("exact frozen controls path, eight-call cap and nonpeak required")
    plan = json.loads(expected.read_text(encoding="utf-8"))
    preauthorization = grant_binding(workspace, identifier, cap) if feedback_controls else {}
    if not (nonpeak_allowed(override_date=preauthorization.get("peak_override_date")) if feedback_controls else nonpeak_allowed()):
        raise ValueError("nonpeak or exact date-scoped approval required")
    if feedback_controls and preauthorization.get("verification_feedback_enabled") is not True:
        raise ValueError("explicit bounded feedback control authorization required")
    cfg = load_product_config(workspace)
    cfg.llm.max_output_tokens = 2048
    cfg.agent.jev_mode = "live"
    if (plan["authorization_id"] != identifier or plan.get("controls_receipt")
        or application_hash(workspace) != plan["application_source_sha256"]
        or canonical_hash(cfg.model_dump(mode="json")) != plan["config_sha256"]
        or len(plan["controls"]) != expected_count):
        raise ValueError("control plan drift or already used")
    if reviewed_roles:
        from jev_rag.io import sha256_file

        confirmed_claims = {}
        for name, digest in plan["confirmation_bindings"].items():
            confirmation = (expected.parent / name).resolve()
            if not confirmation.is_relative_to(expected.parent) or sha256_file(confirmation) != digest:
                raise ValueError("reviewed control confirmation drift")
            packet = json.loads(confirmation.read_text(encoding="utf-8"))
            if packet.get("human_attested") is not True:
                raise ValueError("reviewed source controls require human confirmation")
            for item in packet["confirmed"]:
                texts = item.get("reference_facts", [item.get("claim")])
                citations = item.get("evidence_ids", item.get("citations", []))
                if feedback_controls and item.get("ai_proposed_label") not in {"SUPPORTED_COMPLETE_IN_PROJECT_SCOPE", "CONTRADICTED"}:
                    raise ValueError("feedback controls require exact source-level labels, not completeness-only labels")
                negative = item.get("ai_proposed_label") == "CONTRADICTED"
                for text in texts:
                    if text:
                        confirmed_claims[(text, tuple(sorted(citations)))] = "NOT_SUPPORTED" if negative else "SUPPORTED"
        for control in plan["controls"]:
            signature = (control["claim"], tuple(sorted(control["citations"])))
            if confirmed_claims.get(signature) != control["ai_expectation"]:
                raise ValueError("control differs from exact confirmed source claim")
    chunks = {c["chunk_id"]: Chunk.model_validate(c)
              for c in read_jsonl(workspace / cfg.paths.processed / "chunks.jsonl")}
    with authorization_lock(workspace, identifier), application_runtime_lock(workspace):
        authorization = grant_binding(workspace, identifier, cap)
        if authorization["plan_binding_sha256"] != canonical_hash(plan):
            raise ValueError("exact initial plan/grant binding required")
        session = str(uuid4())
        receipt_path = workspace / "results/reports/live_study" / (session + ".json")
        receipt = dict(type="LIVE_M6_APPLICATION_SESSION_RECEIPT", session_id=session,
                       **authorization, run_cap=cap, used_provider_calls=0, status="serving",
                       purpose="isolated source/whole controls; no user-visible answer")
        write_json(receipt_path, receipt)

        def persist(remaining):
            receipt["used_provider_calls"] = cap - remaining
            write_json(receipt_path, receipt)

        gate = ApplicationGate(entrypoint="benchmark-live", max_calls=cap,
                               enforce_nonpeak=True, peak_override_date=authorization.get("peak_override_date"), on_consume=persist)
        cfg.llm.fast_reasoning_effort = cfg.llm.strong_reasoning_effort = "none"
        key = load_credentials(workspace, os.environ, cfg.llm.provider).get_secret_value()
        adapters = [ProductBudgetAdapter(cfg.llm, api_key=key, gate=gate,
                    structured_policy=cfg.structured_output) for _ in range(2)]
        judges = [ApplicationLLMDecisions(a, ModelPolicy(cfg.llm), strong_completion=True,
                                          verification_feedback_enabled=feedback_controls)
                  for a in adapters]
        results = []
        passed = False
        threshold = cfg.agent.decision_threshold if feedback_controls else .85
        try:
            for control in plan["controls"]:
                claim = Claim(text=control["claim"], citations=control["citations"])
                evidence = [chunks[c] for c in claim.citations]
                args = dict(query=control["question"], claim=claim, evidence=evidence,
                            candidates=["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"])
                source = DecisionRequest(task="citation", **args)
                whole = AnswerCoverageRequest(answer_claims=[claim], **args)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = [pool.submit(judges[0].decide, source), pool.submit(judges[1].decide, whole)]
                    predictions = [job.result() for job in jobs]
                accepted = [p.choice == "SUPPORTED" and all(v is not None and v >= threshold
                            for v in (p.probability, p.confidence)) for p in predictions]
                matched = all(accepted) if control["ai_expectation"] == "SUPPORTED" else not any(accepted)
                if reviewed_roles and control["ai_expectation"] != "SUPPORTED":
                    matched = all(p.choice in {"NOT_SUPPORTED", "CONTRADICTED"}
                                  and all(v is not None and v >= threshold for v in (p.probability, p.confidence))
                                  for p in predictions)
                feedback_valid = all(a.records[-1].get("bounded_verifier_feedback") and all(item.get("quote_binding_valid") for item in a.records[-1]["bounded_verifier_feedback"]) for a in adapters) if feedback_controls else None
                if feedback_controls:
                    matched = matched and feedback_valid
                results.append(dict(feedback_valid=feedback_valid,control=control, actual_sources=[c.model_dump(mode="json") for c in evidence],
                                    predictions=[p.model_dump() for p in predictions],
                                    matches_ai_safety_expectation=matched, human_attested=False))
                if not matched:
                    break
            passed = len(results) == expected_count and all(r["matches_ai_safety_expectation"] for r in results)
        finally:
            receipt.update(status="closed", unused_calls_closed=gate.remaining)
            write_json(receipt_path, receipt)
            output = dict(threshold=threshold,feedback_protocol_enabled=feedback_controls,type="LIVE_DEVELOPMENT_VERIFICATION_CONTROLS", controls=results,
                          ai_expectations_met=passed, actual_calls=cap-gate.remaining,
                          llm_calls=sorted([r for a in adapters for r in a.records],key=lambda r:r["timestamp_utc"]),
                          application_source_sha256=plan["application_source_sha256"], ground_truth_release=False,
                          reference_source_labels_human_confirmed=reviewed_roles,
                          estimated_cost_usd=None,
                          limitation=("Exact source reference claims confirmed with AI-assisted review; provider predictions remain separate. Development controls only, not real user answer delivery, held-out quality or speed proof."
                                      if reviewed_roles else "AI proposed controlled examples only; not formal quality or human ground truth"))
            output_path = expected.parent / expected.name.replace("validation", "controls")
            write_json(output_path, output)
            plan.update(controls_receipt=receipt_path.relative_to(workspace).as_posix(),
                        controls_output=output_path.relative_to(workspace).as_posix(), controls_passed_ai=passed)
            if not passed:
                plan["status"] = "CLOSED_CONTROL_FAILURE"
            write_json(expected, plan)
        print(json.dumps(dict(control_expectations_met=passed, actual_calls=cap-gate.remaining)))
