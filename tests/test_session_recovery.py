import json
from uuid import uuid4

import pytest

from frontend import session
from jev_rag.io import write_json


def old_session(root, monkeypatch, *, state="completed", used=7):
    identifier = str(uuid4())
    write_json(root / ".local/app-startup.json", dict(pid=987654, authorization_id=identifier))
    p = root / "results/reports/live_study/old.json"
    record = dict(
        type="LIVE_M6_APPLICATION_SESSION_RECEIPT",
        authorization_id=identifier,
        status="serving",
        run_cap=40,
        used_provider_calls=used,
        jobs={"MOCK": {"state": state}},
    )
    write_json(p, record)
    locks = [
        root / ".local/app-runtime.lock",
        root / "results/private/live_authorizations" / f"{identifier}.lock",
    ]
    for lock in locks:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch()
    monkeypatch.setattr(session, "_alive", lambda _: False)
    monkeypatch.setattr(session, "_backend_absent", lambda: True)
    return p, record, locks


def test_mock_dead_session_closed_without_refunding_usage(tmp_path, monkeypatch):
    p, original, locks = old_session(tmp_path, monkeypatch)
    monkeypatch.setattr(session, "_launch_grant", lambda *a: {"state": "MOCK"})
    session.authorize_and_start(20, approved=True, workspace=tmp_path)
    closed = json.loads(p.read_text(encoding="utf-8"))
    assert closed["status"] == "closed" and closed["used_provider_calls"] == 7
    assert closed["unused_calls_closed"] == 33
    assert not closed["interruption_recovery"]["unused_credit_transferred"]
    assert all(not lock.exists() for lock in locks)
    archive = next((tmp_path / ".local/session-recovery").iterdir())
    assert json.loads((archive / "old.json").read_text(encoding="utf-8")) == original
    grants = list((tmp_path / "results/private/live_authorizations").glob("*.json"))
    assert len(grants) == 1 and json.loads(grants[0].read_text(encoding="utf-8"))["max_calls"] == 20


@pytest.mark.parametrize(
    "block", ["live_pid", "occupied_port", "pending", "bad_count", "different_session"]
)
def test_mock_uncertain_session_never_reset(tmp_path, monkeypatch, block):
    p, original, locks = old_session(tmp_path, monkeypatch)
    if block == "live_pid":
        monkeypatch.setattr(session, "_alive", lambda _: True)
    if block == "occupied_port":
        monkeypatch.setattr(session, "_backend_absent", lambda: False)
    if block == "pending":
        original["jobs"]["MOCK"]["state"] = "running"
        write_json(p, original)
    if block == "bad_count":
        original["used_provider_calls"] = None
        write_json(p, original)
    if block == "different_session":
        write_json(
            tmp_path / "results/reports/live_study/other.json",
            dict(type=original["type"], status="serving", authorization_id=str(uuid4())),
        )
    monkeypatch.setattr(session, "_launch_grant", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(session.SessionStartupError):
        session.authorize_and_start(20, approved=True, workspace=tmp_path)
    assert json.loads(p.read_text(encoding="utf-8")) == original
    assert all(lock.exists() for lock in locks)
    assert not list((tmp_path / "results/private/live_authorizations").glob("*.json"))


def test_mock_dead_startup_owner_recoverable_and_guard_serializes(tmp_path, monkeypatch):
    local = tmp_path / ".local"
    local.mkdir()
    write_json(local / "app-startup.lock", dict(pid=987654))
    monkeypatch.setattr(session, "_alive", lambda _: False)
    with session._startup_lock(local):
        with pytest.raises(session.SessionStartupError):
            with session._startup_lock(local):
                pytest.fail("concurrent startup")
    assert not (local / "app-startup.lock").exists()
    assert list((local / "session-recovery").glob("*/app-startup.lock"))


def test_mock_backend_absence_requires_exclusive_bind(monkeypatch):
    from unittest.mock import MagicMock

    probe = MagicMock()
    probe.__enter__.return_value = probe
    monkeypatch.setattr(session.socket, "socket", lambda: probe)
    assert session._backend_absent()
    probe.bind.assert_called_once_with(("127.0.0.1", 8765))
    if session.os.name == "nt":
        probe.setsockopt.assert_called_once_with(
            session.socket.SOL_SOCKET, session.socket.SO_EXCLUSIVEADDRUSE, 1
        )
    probe.bind.side_effect = OSError("MOCK occupied or denied")
    assert not session._backend_absent()
