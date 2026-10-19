"""Read curated, source-bound recordings. No client, credentials or provider imports."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

FIXTURES = Path(__file__).with_name("fixtures") / "chat_replays.json"
MODES = {
    "source": "常规 RAG（原始来源）",
    "llm_wiki": "LLM-Wiki（完整知识词条）",
    "llm_wiki_graph": "LLM-Wiki＋关联图谱",
}


def load_recordings(path=FIXTURES):
    payload = path.read_bytes()
    if len(payload) > 10_000_000:
        raise ValueError("replay bundle too large")
    expected = path.with_suffix(".sha256").read_text("utf-8").strip()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("replay bundle changed")
    bundle = json.loads(payload)
    if bundle.get("type") != "OFFLINE_REPLAY_FIXTURES" or not bundle.get("recordings"):
        raise ValueError("invalid replay bundle")
    seen = set()
    for item in bundle["recordings"]:
        key = item["question"], item["knowledge_mode"], item["style"]
        if key in seen or item["knowledge_mode"] not in MODES or item["style"] != "balanced":
            raise ValueError("invalid or duplicate recording")
        if item["result"].get("mode") != "OFFLINE_REPLAY" or item["result"].get("calls_used") != 0:
            raise ValueError("replay must never be presented as a new live run")
        seen.add(key)
    return bundle["recordings"]


def replay_question(recordings, question, knowledge_mode, *, conversation_id):
    """Exact recorded question/config only. Never guess a cached answer to a new question."""
    match = next((r for r in recordings if r["question"] == question.strip()
                  and r["knowledge_mode"] == knowledge_mode and r["style"] == "balanced"), None)
    result = deepcopy(match["result"]) if match else dict(
        mode="OFFLINE_REPLAY", state="completed", status="record_not_available",
        answer="此问题暂无对应的历史记录。请选择示例问题，或粘贴示例问题的完整原文。未调用 API，也未生成新答案。",
        evidence=[], timeline=[], citations=[], calls_used=0, iterations=0,
        replay_provenance={"record_found": False},
    )
    result.update(request_id=str(uuid4()), conversation_id=conversation_id, question=question.strip(),
                  knowledge_mode=knowledge_mode, processing_latency_ms=None)
    return result
