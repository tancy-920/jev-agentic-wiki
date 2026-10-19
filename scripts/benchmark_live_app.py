"""Explicit benchmark-live application session; loopback only, finite grant, no retries."""

import argparse
import json
import os
import secrets
import socket
import sys
from pathlib import Path
from uuid import UUID

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_rag.config import load_config
from jev_rag.evaluation.live_authorization import authorization_lock
from jev_rag.evaluation.preflight import canonical_hash
from jev_rag.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 8765


def make_smoke_plan(workspace, config, query):
    from frontend.backend import application_hash

    if not isinstance(query, str) or not query.strip() or len(query) > 3000:
        raise ValueError("one original bounded application query required")
    return {
        "type": "APPLICATION_TWO_VARIANT_FUNCTIONAL_SMOKE",
        "application_source_sha256": application_hash(workspace),
        "config_sha256": canonical_hash(config.model_dump(mode="json")),
        "query": query,
        "variants": ["A1", "A0"],
        "style": "balanced",
        "complex_multi_hop": False,
        "hard_conflicting_evidence": False,
        "max_calls": 40,
        "maximum_question_calls": 20,
        "transport_retries": 0,
        "human_semantic_review": "pending",
        "ground_truth_release": False,
        "quality_or_speed_advantage_proven": False,
    }


def grant_binding(workspace, identifier, run_cap, acceptance_plan=None, phase=None):
    identifier = str(UUID(identifier))
    path = workspace / "results/private/live_authorizations" / f"{identifier}.json"
    grant = json.loads(path.read_text(encoding="utf-8"))
    scope, limit = "m6-application", 40
    if acceptance_plan is not None:
        from frontend.acceptance import SCOPE

        scope, limit = SCOPE, 192
        if (
            grant.get("plan_binding_sha256") != canonical_hash(acceptance_plan)
            or grant.get("large_study_explicitly_approved") is not True
            or acceptance_plan.get("max_calls") != 192
        ):
            raise ValueError("independent explicit >80-call approval and exact plan required")
        if phase not in {"pilot", "formal"}:
            raise ValueError("one frozen study phase required")
    if (
        grant.get("type") != "HUMAN_APPROVED_LIVE_AUTHORIZATION"
        or grant.get("authorization_id") != identifier
        or grant.get("human_attested") is not True
        or not grant.get("approval_text", "").strip()
        or grant.get("scopes") != [scope]
        or type(grant.get("max_calls")) is not int
        or type(run_cap) is not int
        or not 1 <= run_cap <= grant["max_calls"] <= limit
        or grant.get("fast_max_output_tokens") != 2048
        or grant.get("strong_max_output_tokens") != 8192
        or grant.get("transport_retries") != 0
    ):
        raise ValueError("new human-approved application-only <=40-call grant required")
    used = 0
    product_api_format = grant.get("product_api_format")
    product_timeout = grant.get("product_request_timeout_seconds", 30)
    verification_feedback_enabled = grant.get("verification_feedback_enabled", False)
    if type(verification_feedback_enabled) is not bool or (verification_feedback_enabled and acceptance_plan is not None):
        raise ValueError("explicit product-only feedback protocol required")
    peak_override_date = grant.get("peak_override_date")
    if peak_override_date is not None:
        from datetime import datetime, timedelta, timezone

        today = (datetime.now(timezone.utc) + timedelta(hours=8)).date().isoformat()
        if type(peak_override_date) is not str or peak_override_date != today or acceptance_plan is not None:
            raise ValueError("peak exception requires today's explicit product authorization")
    if type(product_timeout) is not int or product_timeout not in {30, 60} or (acceptance_plan is not None and product_timeout != 30):
        raise ValueError("explicit product timeout30 or60 required")
    if product_api_format is not None and (acceptance_plan is not None or product_api_format not in {"responses", "chat_completions"}):
        raise ValueError("explicit bounded product API format required")
    carry = grant.get("carry_forward")
    if carry is not None:
        session_id = str(UUID(carry["session_id"]))
        previous_path = workspace / "results/reports/live_study" / f"{session_id}.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if (acceptance_plan is not None or previous.get("status") != "closed"
            or previous.get("type") != "LIVE_M6_APPLICATION_SESSION_RECEIPT"
            or sha256_file(previous_path) != carry.get("receipt_sha256")
            or type(carry.get("added_calls")) is not int or carry["added_calls"] < 0
            or grant["max_calls"] != previous["run_cap"] - previous["used_provider_calls"] + carry["added_calls"]):
            raise ValueError("closed exact receipt and explicit additive allowance required")
        if carry["added_calls"] == 0:
            previous_grant_path = workspace / "results/private/live_authorizations" / f"{UUID(previous['authorization_id'])}.json"
            previous_grant = json.loads(previous_grant_path.read_text(encoding="utf-8"))
            if (sha256_file(previous_grant_path) != previous.get("authorization_sha256")
                or grant.get("approval_origin") != "explicit_connection_repair_preserve_allowance"):
                raise ValueError("verified source permission required for zero-addition transfer")
            mutable = {"authorization_id", "max_calls", "approved_at_utc", "approval_origin", "approval_text", "carry_forward"}
            if {k: v for k, v in grant.items() if k not in mutable} != {k: v for k, v in previous_grant.items() if k not in mutable}:
                raise ValueError("permission expansion forbidden during connection repair")
        for candidate in (workspace / "results/private/live_authorizations").glob("*.json"):
            other = json.loads(candidate.read_text(encoding="utf-8"))
            if other.get("authorization_id") != identifier and (other.get("carry_forward") or {}).get("session_id") == session_id:
                raise ValueError("unused allowance already transferred; no duplicate credit")
    for receipt_path in (workspace / "results/reports/live_study").glob("*.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("authorization_id") != identifier:
            continue
        if acceptance_plan and receipt.get("binding", {}).get("phase") == phase:
            raise ValueError("study phase already used; unused balance cannot retry it")
        count = receipt.get("used_provider_calls")
        if (
            receipt.get("type") != "LIVE_M6_APPLICATION_SESSION_RECEIPT"
            or receipt.get("authorization_sha256") != sha256_file(path)
            or receipt.get("status") != "closed"
            or type(count) is not int
            or not 0 <= count <= receipt["run_cap"]
        ):
            raise ValueError(
                "active/interrupted application receipt needs reconciliation; no reset"
            )
        used += count
    if used + run_cap > grant["max_calls"]:
        raise ValueError("aggregate application budget exceeded")
    return {
        "authorization_id": identifier,
        "authorization_sha256": sha256_file(path),
        "authorization_cap": grant["max_calls"],
        "prior_used_provider_calls": used,
        "scope": scope,
        "plan_binding_sha256": grant.get("plan_binding_sha256"),
        "large_study_explicitly_approved": grant.get("large_study_explicitly_approved", False),
        "answer_budget_jev_approved": grant.get("answer_budget_jev_approved") is True,
        "generation_source_refs_enabled": grant.get("generation_source_refs_enabled") is True,
        "product_request_timeout_seconds": product_timeout,
        "peak_override_date": peak_override_date,
        "verification_feedback_enabled": verification_feedback_enabled,
        **({"product_api_format": product_api_format} if product_api_format is not None else {}),
        **({"carry_forward": carry} if carry is not None else {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--max-calls", type=int, required=True, choices=range(1, 193))
    parser.add_argument("--smoke-query")
    parser.add_argument("--smoke-plan", type=Path)
    parser.add_argument("--acceptance-plan", type=Path)
    parser.add_argument("--verification-controls-plan", type=Path)
    parser.add_argument("--verification-audit-plan", type=Path)
    parser.add_argument("--phase", choices=["pilot", "formal"], default="pilot")
    parser.add_argument(
        "--research-ranking",
        action="store_true",
        help="Explicitly enable historical A0/A1 ranking research; default is product A only.",
    )
    args = parser.parse_args(argv)
    if args.verification_audit_plan:
        if args.dry_run or args.verification_controls_plan or args.acceptance_plan or args.smoke_plan or args.smoke_query or args.research_ranking:
            raise ValueError("isolated confirm-live verifier audit only")
        from frontend.verification_audit import run_audit

        run_audit(ROOT,args.authorization_id,args.max_calls,args.verification_audit_plan,grant_binding)
        return 0
    if args.verification_controls_plan:
        if (args.dry_run or args.acceptance_plan or args.smoke_plan or args.smoke_query
            or args.research_ranking):
            raise ValueError("isolated controls require explicit benchmark-live only")
        from frontend.verification_controls import run_controls

        run_controls(ROOT, args.authorization_id, args.max_calls,
                     args.verification_controls_plan, grant_binding)
        return 0
    if bool(args.smoke_query) != bool(args.smoke_plan):
        raise ValueError("smoke query and exact frozen plan are required together")
    from jev_rag.knowledge import load_product_config

    research_requested = bool(args.research_ranking or args.acceptance_plan or args.smoke_plan)
    config = (
        load_config(ROOT / "configs/base.yaml") if research_requested else load_product_config(ROOT)
    )
    config.llm.max_output_tokens = 2048
    config.agent.jev_mode = "live"
    acceptance = None
    if args.acceptance_plan:
        from frontend.acceptance import check_pilot_review, check_references, validate_plan

        if args.smoke_plan or args.smoke_query:
            raise ValueError("acceptance and historical smoke must not be combined")
        acceptance = validate_plan(ROOT, config, args.acceptance_plan)
        check_references(ROOT, acceptance)
        if args.phase == "formal":
            check_pilot_review(ROOT, acceptance)
    authorization = grant_binding(
        ROOT, args.authorization_id, args.max_calls, acceptance, args.phase
    )
    if authorization.get("verification_feedback_enabled") and research_requested:
        raise ValueError("product feedback protocol cannot change research studies")
    if authorization.get("peak_override_date") and research_requested:
        raise ValueError("date-scoped product exception cannot change research studies")
    if authorization.get("product_api_format"):
        if research_requested:
            raise ValueError("product API format override cannot change research studies")
        config.llm.api_format = authorization["product_api_format"]
    smoke = None
    if args.smoke_plan:
        smoke = json.loads(args.smoke_plan.read_text(encoding="utf-8"))
        grant_path = ROOT / "results/private/live_authorizations" / f"{args.authorization_id}.json"
        approved = json.loads(grant_path.read_text(encoding="utf-8"))
        if (
            args.max_calls != 40
            or smoke != make_smoke_plan(ROOT, config, args.smoke_query)
            or approved.get("plan_binding_sha256") != canonical_hash(smoke)
        ):
            raise ValueError("exact original-query/source/config/budget smoke binding required")
    if args.dry_run:
        print(
            f"DRY_RUN application: <= {args.max_calls} calls; 2048/8192 caps; no keys or models loaded."
        )
        return 0
    import uvicorn

    from frontend.acceptance import submission_guard
    from frontend.backend import ChatService, application_hash, create_app, run_question
    from frontend.ranking import load_selection
    from frontend.session import application_runtime_lock
    from jev_rag.llm.credentials import load_credentials
    from jev_rag.retrieval import HybridRetriever

    # Fail before keys/weights if the fixed loopback port is already in use.
    with socket.socket() as probe:
        probe.bind((HOST, PORT))
    with authorization_lock(ROOT, args.authorization_id), application_runtime_lock(ROOT):
        authorization = grant_binding(
            ROOT, args.authorization_id, args.max_calls, acceptance, args.phase
        )
        code_sha = application_hash(ROOT)
        research = bool(args.research_ranking or acceptance or smoke)
        rank_selection = None
        if research:
            try:
                rank_selection = load_selection(ROOT)
            except (OSError, ValueError):
                pass
        if smoke and rank_selection is None:
            raise ValueError("JEV smoke requires development rank selection")
        load_credentials(ROOT, os.environ, config.llm.provider)
        if rank_selection is not None:
            load_credentials(ROOT, os.environ, "jev")
        token = secrets.token_urlsafe(32)
        retriever = None
        knowledge_binding = None
        article_runtime = not research and (ROOT / "knowledge/llm_wiki/manifest.json").exists()
        if article_runtime:
            from frontend.retrieval import build_product_retriever

            retriever = build_product_retriever(ROOT, config)
            knowledge_binding = retriever.store.binding()
        else:
            retriever = HybridRetriever(config)
        if not article_runtime and not research and (ROOT / config.paths.processed / "wiki.json").exists():
            from jev_rag.knowledge import KnowledgeStore, WikiRetriever

            store = KnowledgeStore(ROOT / config.paths.processed)
            knowledge_binding = store.binding()
            retriever = WikiRetriever(retriever, store)
        connection = ROOT / ".local/app-connection.json"
        service = None
        try:
            from frontend.retrieval import warm_local_retriever

            local_warmup = warm_local_retriever(retriever)
            service = ChatService(
                ROOT,
                token=token,
                max_calls=args.max_calls,
                authorization=authorization,
                runner=lambda request, gate: run_question(
                    ROOT, config, request, gate, retriever, os.environ,
                    generation_source_refs=authorization.get("generation_source_refs_enabled") is True,
                    product_request_timeout_seconds=authorization["product_request_timeout_seconds"],
                    verification_feedback_enabled=authorization["verification_feedback_enabled"],
                ),
                source_sha256=code_sha,
                source_check=lambda: application_hash(ROOT),
                jev_ready=rank_selection is not None,
                allowed_variants=("A0", "A1") if research else ("A",),
                enforce_nonpeak=True,
                acceptance_plan=acceptance,
                submission_policy=submission_guard(acceptance[args.phase]) if acceptance else None,
                binding={
                    "local_startup_warmup": local_warmup,
                    "knowledge_binding": knowledge_binding,
                    "knowledge_modes": ["llm_wiki", "llm_wiki_graph", "source"]
                    if article_runtime else ["source", "wiki", "wiki_graph"],
                    "corpus_sha256": sha256_file(ROOT / config.paths.manifest),
                    "chunks_sha256": sha256_file(ROOT / config.paths.processed / "chunks.jsonl"),
                    "config": config.model_dump(mode="json"),
                    "functional_smoke_plan": smoke,
                    "acceptance_plan": acceptance,
                    "phase": args.phase if acceptance else None,
                    "generation_source_refs_enabled": authorization.get("generation_source_refs_enabled") is True,
                },
            )
            if authorization.get("carry_forward"):
                service.restore_conversations(authorization["carry_forward"])
            if acceptance:
                if args.phase == "formal":
                    from uuid import uuid4

                    from frontend.acceptance import SharedPair
                    from frontend.backend import ChatRequest

                    index = {
                        "plan_sha256": canonical_hash(acceptance),
                        "phase": "formal",
                        "session_id": service.session_id,
                        "runs": [],
                        "status": "running",
                    }
                    index_path = ROOT / "results/processed/rag_revision/formal_run_index.json"
                    if index_path.exists():
                        raise ValueError(
                            "formal raw run index already exists; never overwrite failure"
                        )
                    write_json(index_path, index)
                    current_key, pair = None, None
                    for row in acceptance["formal"]:
                        key = (row["case_id"], row["repeat"])
                        if key != current_key:
                            pair, current_key = SharedPair(row["question"]), key
                        request = ChatRequest(
                            request_id=uuid4(),
                            conversation_id=uuid4(),
                            question=row["question"],
                            variant=row["variant"],
                        )
                        request._shared_pair = pair
                        service.submit(request)
                        service.executor.submit(lambda: None).result()
                        response = service.result(str(request.request_id))
                        raw_path = (
                            ROOT
                            / "results/live/application"
                            / service.session_id
                            / f"{request.request_id}.json"
                        )
                        index["runs"].append(
                            {
                                **row,
                                "request_id": str(request.request_id),
                                "raw_sha256": sha256_file(raw_path),
                                "status": response.get("status"),
                            }
                        )
                        write_json(index_path, index)
                        print(
                            f"LIVE formal {row['case_id']} repeat={row['repeat']} {row['variant']}: {response['status']}; calls={response['calls_used']}",
                            flush=True,
                        )
                        if response.get("status") != "answered":
                            index["status"] = "failed_phase_unused_balance_closed"
                            write_json(index_path, index)
                            return 1
                    index["status"] = "completed_source_review_pending"
                    write_json(index_path, index)
                    return 0
            if smoke is not None:
                from uuid import uuid4

                from frontend.backend import ChatRequest

                outcomes = []
                for variant in smoke["variants"]:
                    request = ChatRequest(
                        request_id=uuid4(), question=args.smoke_query, variant=variant
                    )
                    service.submit(request)
                    service.executor.submit(lambda: None).result()
                    response = service.result(str(request.request_id))
                    outcomes.append(response["status"])
                    print(
                        f"LIVE smoke {variant}: {response['status']}; calls={response['calls_used']}",
                        flush=True,
                    )
                return 0 if outcomes == ["answered", "answered"] else 1
            write_json(
                connection,
                {
                    "endpoint": f"http://{HOST}:{PORT}",
                    "token": token,
                    "session_id": service.session_id,
                    "status": "serving",
                },
            )
            print(
                f"Local application listening on {HOST}:{PORT}; finite session={service.session_id}; keys hidden."
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(service, shutdown=lambda: setattr(server, "should_exit", True)),
                    host=HOST,
                    port=PORT,
                    access_log=False,
                    log_level="warning",
                    timeout_graceful_shutdown=150,
                )
            )
            server.run()
        finally:
            if service is not None:
                service.close()
                write_json(connection, {"status": "closed", "session_id": service.session_id})
            retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
