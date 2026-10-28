"""Fixed-loopback API client; never model keys, generated URLs or automatic retries."""

import json
from pathlib import Path
from uuid import UUID

import httpx

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "http://127.0.0.1:8765"


class LocalClient:
    def __init__(self, workspace=ROOT, *, transport=None):
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError("only offline MockTransport injection allowed")
        connection = json.loads(
            (workspace / ".local/app-connection.json").read_text(encoding="utf-8")
        )
        if (
            connection.get("endpoint") != ENDPOINT
            or connection.get("status") != "serving"
            or not isinstance(connection.get("token"), str)
            or len(connection["token"]) < 24
        ):
            raise ValueError("explicit local benchmark-live application session required")
        self.token, self.session_id, self.transport = (
            connection["token"],
            connection["session_id"],
            transport,
        )

    def request(self, method, path, *, payload=None):
        with httpx.Client(
            transport=self.transport, trust_env=False, follow_redirects=False, timeout=10
        ) as client:
            response = client.request(
                method,
                ENDPOINT + path,
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )
        response.raise_for_status()
        result = response.json()
        if path == "/health" and result["session_id"] != self.session_id:
            raise ValueError("local session changed")
        return result

    def health(self):
        return self.request("GET", "/health")

    def submit(self, payload):
        return self.request("POST", "/chat", payload=payload)

    def result(self, request_id):
        return self.request("GET", f"/chat/{UUID(request_id)}")

    def close_session(self):
        return self.request("POST", "/session/close")

    def conversation_memory(self, conversation_id):
        return self.request("GET", f"/conversations/{UUID(str(conversation_id))}/memory")

    def forget_conversation(self, conversation_id):
        return self.request("POST", f"/conversations/{UUID(str(conversation_id))}/forget")

    def preferences(self):
        return self.request("GET", "/preferences")

    def save_preference(self, key, value):
        return self.request("POST", "/preferences", payload={"key": key, "value": value})

    def forget_preferences(self):
        return self.request("POST", "/preferences/forget")

    def cancel(self, request_id):
        return self.request("POST", f"/chat/{UUID(str(request_id))}/cancel")
