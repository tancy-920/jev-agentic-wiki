"""MOCK bounded article repair and version-safe history; no providers."""
import json
from uuid import uuid4

import pytest

from frontend.backend import ChatRequest, ChatService
from jev_rag.llm_wiki import ArticleRetriever, ArticleStore, article_digest


@pytest.fixture
def articles(tmp_path):
    rows = []
    for identifier, title, aliases, related in [
        ("kv-cache", "KV Cache", ["缓存"], ["gqa", "paged"]),
        ("gqa", "GQA", ["分组查询"], ["kv-cache"]),
        ("paged", "分页KV", ["分页"], ["kv-cache"]),
    ]:
        text = f"# {title}\n\n## 定义\nMOCK complete article.\n\n## 机制\n{title} MOCK.\n\n## 边界\nMOCK not ground truth.\n"
        (tmp_path / f"{identifier}.md").write_text(text, "utf8", newline="\n")
        rows.append(dict(article_id=identifier, title=title, aliases=aliases, related=related,
                         markdown_file=f"{identifier}.md", sha256=article_digest(text),
                         source_chunk_ids=["MOCK:original"]))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        type="LLM_WIKI_COMPLETE_ARTICLES_V1", source_binding={"MOCK": "binding"}, articles=rows)), "utf8")
    return ArticleStore(tmp_path)


def test_mock_repair_keeps_original_scope_and_new_evidence_in_one_budget(articles):
    retriever = ArticleRetriever(articles)
    old = [articles.chunks['wiki-article:' + i].model_dump() for i in ('kv-cache', 'gqa')]
    fresh = [articles.chunks['wiki-article:paged'].model_dump()]
    selected, detail = retriever.merge_context(old, fresh, top_k=6)
    assert [c['chunk_id'] for c in selected] == [
        'wiki-article:kv-cache', 'wiki-article:paged', 'wiki-article:gqa']
    assert detail['new_provider_calls'] == 0
    retriever.max_chars = len(old[0]['text']) + len(fresh[0]['text'])
    selected, detail = retriever.merge_context(old, fresh, top_k=6)
    assert len(selected) == 2 and detail['omitted_whole'] == ['wiki-article:gqa']
    assert sum(len(c['text']) for c in selected) <= retriever.max_chars
    assert all(c['text'] == articles.chunks[c['chunk_id']].text for c in selected)
    selected, detail = retriever.merge_context(old, [old[0], *fresh], top_k=2)
    assert len({c['chunk_id'] for c in selected}) == len(selected) == 2


@pytest.mark.parametrize("changed", [False, True])
def test_mock_rebinding_changed_knowledge_preserves_display_without_stale_context(tmp_path, changed):
    from types import SimpleNamespace

    from frontend.backend import canonical_hash
    from jev_rag.io import sha256_file, write_json

    old_id = str(uuid4())
    request = ChatRequest(request_id=uuid4(), conversation_id=uuid4(), question="MOCK question")
    rid = str(request.request_id)
    previous_path = tmp_path / "results/reports/live_study" / (old_id + ".json")
    binding = dict(config={"MOCK": "same"}, knowledge_binding="MOCK old")
    write_json(previous_path, dict(status="closed", binding=binding, jobs={rid: {
        "state": "completed", "input_sha256": canonical_hash(request.model_dump(mode="json"))}}))
    write_json(tmp_path / "results/live/application" / old_id / (rid + ".json"), {
        "request": request.model_dump(mode="json"),
        "response": {"status": "answered", "answer": "MOCK historical answer"}})
    service = SimpleNamespace(workspace=tmp_path, jobs={}, conversations={}, task_memories={}, persist=lambda: None,
        receipt={"binding": {**binding, "knowledge_binding": "MOCK new" if changed else "MOCK old"}})
    ChatService.restore_conversations(service, dict(session_id=old_id, receipt_sha256=sha256_file(previous_path)))
    assert service.jobs[rid]["response"]["answer"] == "MOCK historical answer"
    assert bool(service.conversations) is not changed
    assert service.receipt["conversation_transfer"]["context_restored"] is not changed
    assert service.receipt["conversation_transfer"]["context_reset_reason"] == ("knowledge_changed" if changed else None)
    service.receipt["binding"]["config"] = {"MOCK": "different"}
    with pytest.raises(ValueError, match="unchanged configuration"):
        ChatService.restore_conversations(service, dict(session_id=old_id, receipt_sha256=sha256_file(previous_path)))

