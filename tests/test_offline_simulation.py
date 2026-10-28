"""Offline replay integration. Socket access is forbidden by conftest."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from frontend import chat
from frontend.offline_replay import FIXTURES, load_recordings, replay_question
from frontend.simulation import finish_pending


def app_without_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline replay must not construct a client or authorize a service")

    def idle_loop():
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.is_closed.return_value = False
        loop.is_running.return_value = False
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", idle_loop)
    monkeypatch.setattr(chat, "LocalClient", forbidden)
    monkeypatch.setattr(chat, "authorize_and_start", forbidden)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"),
                            default_timeout=15)
    app.query_params["mode"] = "simulate"
    return app.run()


def complete(app):
    app.session_state["offline_workspace"]["pending"]["ready_at"] = 0
    return app.run()


def test_bundle_retains_success_failure_and_full_evidence():
    records = load_recordings()
    assert len(records) == 26
    assert len({r["question"] for r in records}) == 20
    assert {r["result"]["status"] for r in records} == {"answered", "generation_failure"}
    assert all(r["result"]["calls_used"] == 0 for r in records)
    assert all(r["result"]["replay_provenance"]["source_sha256"] for r in records)
    wiki = next(r for r in records if r["knowledge_mode"] == "llm_wiki")
    assert wiki["result"]["evidence"] and wiki["result"]["citations"]
    assert all(c["text"].startswith("# ") for c in wiki["result"]["evidence"])


def test_bundle_drift_fails_closed(tmp_path):
    path = tmp_path / "recordings.json"
    path.write_bytes(FIXTURES.read_bytes() + b" ")
    path.with_suffix(".sha256").write_bytes(FIXTURES.with_suffix(".sha256").read_bytes())
    with pytest.raises(ValueError, match="changed"):
        load_recordings(path)


def test_exact_question_only_and_independent_result_ids():
    records = load_recordings()
    source = records[0]
    a = replay_question(records, source["question"], source["knowledge_mode"], conversation_id="MOCK-a")
    b = replay_question(records, source["question"], source["knowledge_mode"], conversation_id="MOCK-a")
    assert a["request_id"] != b["request_id"]
    a["evidence"].clear()
    assert b["evidence"]
    missing = replay_question(records, source["question"] + "另外一个问题", source["knowledge_mode"],
                              conversation_id="MOCK-a")
    assert missing["status"] == "record_not_available" and missing["calls_used"] == 0


def test_poll_completion_is_idempotent_and_preserves_wait():
    state = dict(results=[], pending=dict(ready_at=10, result=dict(request_id="MOCK-1")))
    assert not finish_pending(state, now=9)
    assert finish_pending(state, now=10)
    assert not finish_pending(state, now=11)
    assert len(state["results"]) == 1


def test_entry_is_available_without_backend_and_manual_authorization(monkeypatch):
    app = app_without_backend(monkeypatch)
    assert not app.exception
    assert app.sidebar.radio[0].value == "离线模拟"
    assert not app.chat_input
    assert app.text_input(key="offline_wiki_search")
    assert any(m.label == "本次 API 请求" and m.value == "0" for m in app.metric)
    assert not any(b.label == "授权并启动实时对话" for b in app.button)
    assert not any(s.key == "offline_example" for s in app.selectbox)
    assert app.button(key="offline_examples")


def test_added_historical_question_only_offers_its_recorded_mode(monkeypatch):
    app = app_without_backend(monkeypatch)
    app.button(key="offline_examples").click().run()
    app.button(key="offline_card_H01").click().run()
    assert not app.exception
    assert app.sidebar.selectbox(key="offline_knowledge_mode").value == "source"
    assert app.sidebar.selectbox(key="offline_knowledge_mode").options == ["常规 RAG（原始来源）"]
    next(b for b in app.button if b.label == "发送示例问题").click().run()
    complete(app)
    assert not app.exception
    assert app.session_state["offline_workspace"]["results"][-1]["calls_used"] == 0


def test_send_wait_render_citations_export_and_no_duplicate(monkeypatch):
    app = app_without_backend(monkeypatch)
    next(b for b in app.button if b.label == "发送示例问题").click().run()
    assert not app.exception and not app.chat_input
    assert any("仅演示等待动画" in c.value for c in app.caption)
    complete(app)
    assert not app.exception and not app.chat_input
    assert len(app.session_state["offline_workspace"]["results"]) == 1
    assert any("引用与完整词条" in e.label for e in app.expander)
    assert any(e.label == "运行详情与引用核对" for e in app.expander)
    assert app.get("download_button")
    app.run()
    assert len(app.session_state["offline_workspace"]["results"]) == 1


def test_search_and_graph_browsing_stays_offline_and_live_history_is_untouched(monkeypatch):
    app = app_without_backend(monkeypatch)
    app.session_state["chat_results"] = [{"request_id": "MOCK-existing-live"}]
    app.text_input(key="offline_wiki_search").set_value("GQA").run()
    assert not app.exception and not app.chat_input
    assert "GQA 分组查询注意力" in app.selectbox(key="offline_article").options
    app.selectbox(key="offline_article").set_value("gqa").run()
    assert any("完整 Wiki" in e.label for e in app.expander)
    assert app.button(key="offline_graph_mqa")
    app.button(key="offline_graph_mqa").click().run()
    assert not app.exception
    assert app.selectbox(key="offline_article").value == "mqa"
    assert app.session_state["chat_results"] == [{"request_id": "MOCK-existing-live"}]
    assert not app.session_state["offline_workspace"]["results"]


def test_original_rag_failure_can_be_demoed_without_fabricated_answer(monkeypatch):
    app = app_without_backend(monkeypatch)
    app.sidebar.selectbox(key="offline_knowledge_mode").set_value("source").run()
    records = load_recordings()
    assert any(r["result"]["status"] == "generation_failure" for r in records)
    app.button(key="offline_examples").click().run()
    app.button(key="offline_card_D07").click().run()
    next(b for b in app.button if b.label == "发送示例问题").click().run()
    complete(app)
    assert not app.exception
    result = app.session_state["offline_workspace"]["results"][0]
    assert result["status"] == "generation_failure" and result["calls_used"] == 0
    assert "答案生成失败" in result["answer"]


def test_new_conversation_and_history_switch_keep_recordings(monkeypatch):
    app = app_without_backend(monkeypatch)
    next(b for b in app.button if b.label == "发送示例问题").click().run()
    complete(app)
    old = app.session_state["offline_workspace"]["conversation"]
    next(b for b in app.button if b.label == "开始新对话").click().run()
    assert app.session_state["offline_workspace"]["conversation"] != old
    assert len(app.session_state["offline_workspace"]["results"]) == 1
    app.sidebar.button(key="offline_history_" + old).click().run()
    assert app.session_state["offline_workspace"]["conversation"] == old
    assert len(app.chat_message) == 2 and not app.exception


def test_missing_bundle_never_falls_back_to_live(monkeypatch):
    from frontend import simulation

    def missing():
        raise OSError("MOCK missing recordings")

    monkeypatch.setattr(simulation, "load_recordings", missing)
    app = app_without_backend(monkeypatch)
    assert not app.exception and not app.chat_input
    assert any("不会回退到在线 API" in e.value for e in app.error)


def test_ten_cards_select_and_send_without_api(monkeypatch):
    app = app_without_backend(monkeypatch)
    app.button(key="offline_examples").click().run()
    cards = [b for b in app.button if b.key and b.key.startswith("offline_card_")]
    assert len(cards) == 10
    app.button(key="offline_card_H04").click().run()
    assert not app.exception
    assert app.session_state["offline_example"] == "MHA，MQA，GQA的区别？"
    next(b for b in app.button if b.label == "发送选中问题 →").click().run()
    complete(app)
    assert not app.exception
    result = app.session_state["offline_workspace"]["results"][-1]
    assert result["question"] == "MHA，MQA，GQA的区别？"
    assert result["calls_used"] == 0


def test_answer_has_working_wiki_entry(monkeypatch):
    app = app_without_backend(monkeypatch)
    next(b for b in app.button if b.label == "发送示例问题").click().run()
    complete(app)
    link = next(b for b in app.button if b.key and b.key.startswith("wiki_entry_") and b.key.endswith("_transformer"))
    link.click().run()
    assert not app.exception
    assert app.selectbox(key="offline_article").value == "transformer"
    assert any(e.label == "完整 Wiki · Transformer 架构" for e in app.expander)


def test_complete_article_explorer_does_not_build_original_corpus(monkeypatch):
    import frontend.wiki as wiki

    def forbidden(*args, **kwargs):
        pytest.fail('Whole article browsing must not build the original corpus')

    monkeypatch.setattr(wiki, 'verified_store', forbidden)
    app = app_without_backend(monkeypatch)
    app.sidebar.radio[0].set_value('知识 Wiki').run()
    assert not app.exception
    assert app.text_input(key='atlas_query')
    app.text_input(key='atlas_query').set_value('GQA').run()
    app.button(key='atlas_topic_gqa').click().run()
    assert not app.exception
    assert app.session_state['atlas_selected'] == 'gqa'
    assert any('分组查询' in item.value for item in app.markdown)
    app.text_input(key='atlas_query').set_value('MOCK_NO_SUCH_TOPIC_987').run()
    assert any('没有匹配词条' in item.value for item in app.info)


def test_browsing_cache_revalidates_changed_source_and_catalog(tmp_path, monkeypatch):
    import hashlib
    import json

    from frontend import local_catalog
    from jev_rag.io import write_json, write_jsonl
    from jev_rag.knowledge import KnowledgeSource, WikiCatalog
    from jev_rag.schema import Chunk

    text = 'MOCK original source.\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    source = KnowledgeSource(source_id='mock', title='MOCK', kind='code', origin='MOCK', version='MOCK',
        snapshot='mock.txt', normalized='mock.txt', snapshot_sha256=digest, normalized_sha256=digest,
        parser='MOCK', retrieved_at_utc='MOCK')
    chunk = Chunk(source_id='mock', source_kind='code', source_sha256=digest, title='MOCK', authors=[],
        year=2026, chunk_id='mock:1', source_url='', locator='L1', text=text, token_count=4,
        start_char=0, end_char=len(text))
    (tmp_path/'mock.txt').write_text(text, 'utf8', newline='')
    write_jsonl(tmp_path/'sources.jsonl', [source.model_dump()])
    write_jsonl(tmp_path/'chunks.jsonl', [chunk.model_dump()])
    hashes = {n: hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in ('sources.jsonl', 'chunks.jsonl')}
    write_json(tmp_path/'wiki.json', WikiCatalog(sources_sha256=hashes['sources.jsonl'],
        chunks_sha256=hashes['chunks.jsonl'], pages=[], edges=[]).model_dump())
    monkeypatch.setattr(local_catalog, 'product_store_path', lambda _: tmp_path)
    local_catalog._verified_store.cache_clear()
    a = local_catalog.verified_store(tmp_path)
    assert 'term_catalog' not in a.__dict__
    assert local_catalog.verified_store(tmp_path) is a
    # Raw bytes drift must invalidate a warm browsing cache and fail closed.
    (tmp_path/'mock.txt').write_text('MOCK changed source.\n', 'utf8', newline='')
    with pytest.raises(ValueError, match='snapshot changed'):
        local_catalog.verified_store(tmp_path)
    (tmp_path/'mock.txt').write_text(text, 'utf8', newline='')
    row = json.loads((tmp_path/'wiki.json').read_text('utf8'))
    row['chunks_sha256'] = '0'*64
    write_json(tmp_path/'wiki.json', row)
    with pytest.raises(ValueError, match='stale Wiki'):
        local_catalog.verified_store(tmp_path)
    local_catalog._verified_store.cache_clear()
