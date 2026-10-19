"""User-clicked finite authorization; launch only the fixed benchmark-live entrypoint."""

import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from jev_rag.io import write_json

ROOT = Path(__file__).resolve().parents[1]
_children = {}


@contextmanager
def application_runtime_lock(workspace):
    path = workspace / ".local/app-runtime.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8"):
        pass
    try:
        yield
    finally:
        path.unlink()


def _alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied means possibly alive; never reset its budget.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def launch_status(workspace=ROOT):
    path = workspace / ".local/app-startup.json"
    if not path.exists():
        return {"state": "idle"}
    data = json.loads(path.read_text(encoding="utf-8"))
    # A historical PID can be inaccessible/reused or its parent still alive
    # after the model session ends. The exact authorization's closed receipts
    # are authoritative; never confuse them with a new startup or reset usage.
    identifier = data.get("authorization_id")
    receipts = []
    for receipt_path in (workspace / "results/reports/live_study").glob("*.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("type") == "LIVE_M6_APPLICATION_SESSION_RECEIPT"
            and isinstance(identifier, str)
            and receipt.get("authorization_id") == identifier
        ):
            receipts.append(receipt)
    if receipts and all(r.get("status") == "closed" for r in receipts):
        return {**data, "state": "stopped", "reason": "session_closed"}
    child = _children.get(str(workspace.resolve()))
    active = child.poll() is None if child else _alive(data.get("pid"))
    return {**data, "state": "starting" if active else "stopped"}


def _unresolved(workspace):
    for path in (workspace / "results/reports/live_study").glob("*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            receipt.get("type") == "LIVE_M6_APPLICATION_SESSION_RECEIPT"
            and receipt.get("status") != "closed"
        ):
            return True
    return False


def _today_peak_override(workspace, now=None):
    try:
        value = json.loads(
            (workspace / "results/private/live_runtime_preferences.json").read_text(
                encoding="utf-8"
            )
        )
        today = ((now or datetime.now(timezone.utc)) + timedelta(hours=8)).date().isoformat()
        if (
            isinstance(value, dict)
            and value.get("approved") is True
            and value.get("origin") == "explicit_user_instruction"
            and value.get("peak_override_date") == today
        ):
            return today
    except (OSError, ValueError, TypeError):
        pass
    return None


def authorize_and_start(
    max_calls, *, approved=False, answer_budget_jev_approved=False, workspace=ROOT
):
    """No activation on render/selection/rerun. An explicit button click is required."""
    if approved is not True or type(max_calls) is not int or max_calls not in (20, 40):
        raise ValueError("explicit user click and a 20/40-request session required")
    if type(answer_budget_jev_approved) is not bool:
        raise ValueError("explicit boolean JEV transmission permission required")
    local = workspace / ".local"
    local.mkdir(parents=True, exist_ok=True)
    lock = local / "app-startup.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if (
            launch_status(workspace)["state"] == "starting"
            or _unresolved(workspace)
            or (local / "app-runtime.lock").exists()
        ):
            raise ValueError(
                "existing starting or unreconciled application session; no quota reset"
            )
        identifier = str(uuid4())
        grant = {
            "type": "HUMAN_APPROVED_LIVE_AUTHORIZATION",
            "authorization_id": identifier,
            "human_attested": True,
            "approval_origin": "explicit_local_demo_authorize_button",
            "approval_text": f"用户点击授权并启动Transformer知识对话，最多{max_calls}次实际API请求；允许向配置的LLM提供方发送问题、最近三轮已完成对话、AI整理的完整Wiki词条，以及用户选择原文RAG时的公开论文/官方文档及本地mygpt技术代码和实验JSON来源片段，不发送密钥、权重或训练数据。",
            "allowed_variants": ["A"],
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "scopes": ["m6-application"],
            "max_calls": max_calls,
            "fast_max_output_tokens": 2048,
            "strong_max_output_tokens": 8192,
            "transport_retries": 0,
            "estimated_cost_usd": None,
            "ground_truth_release": False,
            "answer_budget_jev_approved": answer_budget_jev_approved,
            "generation_source_refs_enabled": True,
            "verification_feedback_enabled": True,
        }
        daily_override = _today_peak_override(workspace)
        if daily_override:
            grant["peak_override_date"] = daily_override
            grant["approval_text"] += (
                f" 用户此前明确允许{daily_override}高峰时段调用，例外只在该北京时间日期有效。"
            )
        if answer_budget_jev_approved:
            grant["approval_text"] += (
                " 用户另勾选可选JEV回答预算试用，允许向配置JEV提供方发送本次技术问题与最多3条检索原文；"
                "仅事实题判断LIGHT/STANDARD/DEEP，非事实题保持标准核验，低置信回退，不传人工标签或审核理由。"
            )
        write_json(workspace / "results/private/live_authorizations" / f"{identifier}.json", grant)
        return _launch_grant(workspace, identifier, max_calls)
    finally:
        lock.unlink()


def _launch_grant(workspace, identifier, max_calls):
    local = workspace / ".local"
    executable = workspace / ".venv/Scripts/python.exe"
    if os.name != "nt":
        executable = workspace / ".venv/bin/python"
    command = [
        str(executable),
        "-X",
        "utf8",
        str(workspace / "scripts/benchmark_live_app.py"),
        "--confirm-live",
        "--authorization-id",
        identifier,
        "--max-calls",
        str(max_calls),
    ]
    # Fixed local executable/entrypoint; no shell or provider-generated command.
    with (local / "app-startup.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    _children[str(workspace.resolve())] = child
    data = {"authorization_id": identifier, "max_calls": max_calls, "pid": child.pid}
    write_json(local / "app-startup.json", data)
    return {**data, "state": "starting"}


def resume_closed_session(session_id, *, approved=False, workspace=ROOT):
    """Transfer only unused credit; never add calls or expand provider permissions."""
    from uuid import UUID

    from jev_rag.io import sha256_file
    from scripts.benchmark_live_app import grant_binding

    if approved is not True:
        raise ValueError("explicit connection-repair request required")
    session_id = str(UUID(session_id))
    local = workspace / ".local"
    local.mkdir(parents=True, exist_ok=True)
    lock = local / "app-startup.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if _unresolved(workspace) or (local / "app-runtime.lock").exists():
            raise ValueError("wait for previous backend to close; no quota reset")
        receipt_path = workspace / "results/reports/live_study" / f"{session_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        previous_id = str(UUID(receipt["authorization_id"]))
        previous_path = workspace / "results/private/live_authorizations" / f"{previous_id}.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if (receipt.get("status") != "closed" or receipt.get("scope") != "m6-application"
            or receipt.get("authorization_sha256") != sha256_file(previous_path)
            or previous.get("allowed_variants") != ["A"]):
            raise ValueError("closed verified product-only authorization required")
        remaining = receipt["run_cap"] - receipt["used_provider_calls"]
        if type(remaining) is not int or not 1 <= remaining <= 40:
            raise ValueError("no transferable unused allowance")
        identifier = str(uuid4())
        grant = {**previous, "authorization_id": identifier, "max_calls": remaining,
                 "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                 "approval_origin": "explicit_connection_repair_preserve_allowance",
                 "approval_text": previous["approval_text"] + " 用户要求修复无法输入的问题；仅结转原会话未用额度，新增0次，原数据发送范围不变。",
                 "carry_forward": {"session_id": session_id, "receipt_sha256": sha256_file(receipt_path), "added_calls": 0}}
        path = workspace / "results/private/live_authorizations" / f"{identifier}.json"
        write_json(path, grant)
        try:
            grant_binding(workspace, identifier, remaining)
        except Exception:
            # Preserve a failed transfer for audit without reserving its credit.
            path.rename(path.with_suffix(".rejected"))
            raise
        return _launch_grant(workspace, identifier, remaining)
    finally:
        lock.unlink()
