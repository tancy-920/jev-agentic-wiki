"""User-clicked finite authorization; launch only the fixed benchmark-live entrypoint."""

import json
import os
import shutil
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from jev_rag.io import write_json

ROOT = Path(__file__).resolve().parents[1]
_children = {}


class SessionStartupError(ValueError):
    """Safe, actionable startup failure text for the UI (never provider errors)."""


def _backend_absent():
    """An exclusive bind proves the fixed local endpoint is available.

    Connecting to an unused port can time out under Windows firewall rules;
    that is not evidence of a running server. Match the entrypoint's bind check.
    """
    try:
        with socket.socket() as probe:
            if os.name == "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", 8765))
        return True
    except OSError:
        return False


def reconcile_interrupted_session(workspace):
    """Close an exactly identified dead session; never restore its unused credit.

    Called only under the startup lock after an explicit new-session approval.
    Preserve original receipts/locks. Unfinished jobs require manual accounting.
    """
    local = workspace / ".local"
    startup = local / "app-startup.json"
    if not startup.exists():
        return
    data = json.loads(startup.read_text(encoding="utf-8"))
    pid = data.get("pid")
    if type(pid) is not int or pid <= 0 or _alive(pid):
        return
    identifier = data.get("authorization_id")
    # Validate before constructing any filesystem path from the record.
    from uuid import UUID

    try:
        identifier = str(UUID(identifier))
    except (ValueError, TypeError, AttributeError):
        raise SessionStartupError("旧会话标识无效，需要核对启动记录；未新增授权。") from None
    matches = []
    for path in (workspace / "results/reports/live_study").glob("*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("type") != "LIVE_M6_APPLICATION_SESSION_RECEIPT":
            continue
        if receipt.get("status") != "closed" and receipt.get("authorization_id") != identifier:
            raise SessionStartupError("另有未结算会话，需要核对调用记录；未新增授权。")
        if receipt.get("authorization_id") == identifier and receipt.get("status") != "closed":
            used, cap = receipt.get("used_provider_calls"), receipt.get("run_cap")
            if (type(used) is not int or type(cap) is not int or not 0 <= used <= cap
                    or receipt.get("in_progress")
                    or any(j.get("state") != "completed" for j in receipt.get("jobs", {}).values())):
                raise SessionStartupError("旧会话存在未完成请求或账目不明确，需要核对；未重置额度。")
            matches.append((path, receipt))
    if not matches and not (local / "app-runtime.lock").exists():
        return
    if not _backend_absent():
        raise SessionStartupError("本地问答端口仍被占用或无法确认状态；请勿重复启动。")
    archive = local / "session-recovery" / str(uuid4())
    archive.mkdir(parents=True)
    for path, receipt in matches:
        shutil.copy2(path, archive / path.name)
        receipt.update(
            status="closed",
            unused_calls_closed=receipt["run_cap"] - receipt["used_provider_calls"],
            interruption_recovery={
                "reason": "owner_process_exited_no_backend_no_pending_jobs",
                "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
                "old_pid": pid,
                "unused_credit_transferred": False,
            },
        )
        write_json(path, receipt)
    for path in [local / "app-runtime.lock",
                 workspace / "results/private/live_authorizations" / f"{identifier}.lock"]:
        if path.exists():
            path.replace(archive / path.name)


@contextmanager
def _startup_lock(local):
    # Kernel lock serializes recovery across Streamlit reruns/processes and is
    # released by the OS even on a hard crash. Keep the guard inode in place.
    with (local / "app-startup.guard").open("a+b") as guard:
        guard.seek(0, 2)
        if guard.tell() == 0:
            guard.write(b"0")
            guard.flush()
        guard.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise SessionStartupError("已有启动任务正在运行，请等待连接完成。") from None
        else:
            import fcntl
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SessionStartupError("已有启动任务正在运行，请等待连接完成。") from None
        try:
            with _startup_marker(local):
                yield
        finally:
            guard.seek(0)
            if os.name == "nt":
                msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(guard, fcntl.LOCK_UN)


@contextmanager
def _startup_marker(local):
    path = local / "app-startup.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
    except FileExistsError:
        try:
            owner = json.loads(path.read_text(encoding="utf-8"))
            pid = owner.get("pid")
        except (OSError, ValueError, AttributeError):
            raise SessionStartupError("启动锁无法确认归属，需要核对；未新增授权。") from None
        if type(pid) is not int or pid <= 0 or _alive(pid):
            raise SessionStartupError("已有启动任务正在运行，请等待连接完成。") from None
        archive = local / "session-recovery" / str(uuid4())
        archive.mkdir(parents=True)
        path.replace(archive / path.name)
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
    try:
        yield
    finally:
        path.unlink()


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
    with _startup_lock(local):
        reconcile_interrupted_session(workspace)
        if (
            launch_status(workspace)["state"] == "starting"
            or _unresolved(workspace)
            or (local / "app-runtime.lock").exists()
        ):
            raise SessionStartupError(
                "已有运行中或未结算的问答会话；请先结束旧会话，未重置调用额度。"
            )
        identifier = str(uuid4())
        grant = {
            "type": "HUMAN_APPROVED_LIVE_AUTHORIZATION",
            "authorization_id": identifier,
            "human_attested": True,
            "approval_origin": "explicit_local_demo_authorize_button",
            "approval_text": f"用户点击授权并启动Transformer知识对话，最多{max_calls}次实际API请求；允许向配置的LLM提供方发送问题、最多6条相关历史消息、当前对话中有消息出处且限长的讨论条件/纠正/未解决问题，以及用户明确保存的讲解偏好；近期全文仅保留最近32轮，任务原话可跨越该窗口但不是技术证据。临时对话不读取或积累这些记忆。还允许发送AI整理的完整Wiki词条，以及用户选择原文RAG时的公开论文/官方文档及本地mygpt技术代码和实验JSON来源片段，不发送密钥、权重或训练数据。",
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
    with _startup_lock(local):
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
                 "approval_text": previous["approval_text"] + " 用户要求修复并重新绑定本地问答服务；仅结转原会话未用额度，新增0次，原数据发送范围不变。",
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
