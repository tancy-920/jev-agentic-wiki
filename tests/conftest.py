import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    # Tokenizer data is prepared locally, not downloaded by tests. Its default
    # tempfile cache otherwise changes with Windows sandbox/TEMP directories.
    cache = Path(__file__).resolve().parents[1] / ".local/tokenizer-cache"
    if cache.is_dir():
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests are offline; network calls are forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
