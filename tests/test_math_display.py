"""Math presentation only. Provider access is prohibited by conftest."""

import asyncio
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from frontend.citation_display import article_text, article_title, compact_citations
from frontend.math_display import answer_markdown
from frontend.offline_replay import load_recordings


@pytest.mark.parametrize("prefix", ["", "LIVE — "])
def test_remove_only_the_known_leading_notice(prefix):
    assert answer_markdown(prefix + "已通过模型引用验证，尚非人工核实的事实\n\n答案") == "答案"
    assert answer_markdown("引用：已通过模型引用验证，尚非人工核实的事实") == "引用：已通过模型引用验证，尚非人工核实的事实"
    assert answer_markdown("LIVE — 答案生成失败，未生成通过验证的答案。") == "LIVE — 答案生成失败，未生成通过验证的答案。"


def test_ffn_formulas_become_math_without_changing_raw_record():
    record = next(r for r in load_recordings() if r["case_id"] == "D03" and r["knowledge_mode"] == "llm_wiki")
    original = deepcopy(record)
    shown = answer_markdown(record["result"]["answer"])
    assert r"$\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$" in shown
    assert r"\operatorname{GELU}(x)=x\Phi(x)" in shown
    assert r"\operatorname{SwiGLU}(x)=[\operatorname{SiLU}(xW)\odot(xV)]W_2" in shown
    assert r"\frac{2}{3}" in shown
    assert r"512\to 2048\to 512" in shown
    assert record == original
    assert answer_markdown(shown) == shown
    for citation in ["mlp.md", "activations.md", "transformer.md"]:
        assert citation not in shown


def test_code_existing_math_links_unknown_and_false_formulas_remain_literal():
    unchanged = r'''`d_model` [d_model](https://example.com/d_model) $x_1+x_2$
```python
FFN(x)=max(0,xW1+b1)W2+b2
```
ReLU(x)=max(1,x) GELU(x)=xPhi(x) a_d_model filename_d_ff.md'''
    assert answer_markdown(unchanged) == unchanged
    assert answer_markdown(r"公式\(x^2\)结束") == r"公式$x^2$结束"
    assert "$$\nx^2\n$$" in answer_markdown(r"\[x^2\]")


def test_inline_sources_removed_but_math_and_code_kept():
    prose = "正文 [题目, full page context, PDF file p. 3] [另一词条 · file.md] 继续。"
    shown = answer_markdown(prose)
    assert "PDF file" not in shown and "file.md" not in shown
    assert "正文" in shown and "继续。" in shown
    assert answer_markdown("[x+y] `source.md` [链接](https://example.com)") == "[x+y] `source.md` [链接](https://example.com)"


def test_kv_and_normalization_presentations_preserve_factors():
    assert answer_markdown("2·L·B·T·H_kv·d_head·s") == r"$2\cdot L\cdot B\cdot T\cdot H_{\mathrm{kv}}\cdot d_{\mathrm{head}}\cdot s$"
    assert answer_markdown("LayerNorm(x + Sublayer(x))") == r"$\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$"


def test_all_curated_answers_render_idempotently():
    for item in load_recordings():
        text = answer_markdown(item["result"]["answer"])
        assert answer_markdown(text) == text
        assert not text.startswith("已通过模型引用验证")
        assert "AI整理词条" not in text
        assert "非原论文逐字原文" not in text
        assert "captured public-source excerpt" not in text
        assert "PDF file p." not in text
        assert "snapshot L" not in text


def test_web_snapshot_citations_removed_without_removing_mask_or_math():
    source = "机制 [GQA, captured public-source excerpt, captured excerpt L50–L62; snapshot L9–L21]，保留[MASK]和[x+y]。"
    assert answer_markdown(source) == "机制 ，保留[MASK]和[x+y]。"


def test_all_article_headers_and_titles_use_clean_display():
    from pathlib import Path

    articles = list(Path("knowledge/llm_wiki").rglob("*.md"))
    assert len(articles) == 36
    for path in articles:
        raw = path.read_text(encoding="utf-8")
        shown = article_text(raw)
        assert "状态：AI整理草稿" not in shown, path
        assert "未获人工逐项确认" not in shown, path
        assert "## 来源追溯" in shown, path
    for record in load_recordings():
        if record["knowledge_mode"] == "source":
            continue
        for evidence in record["result"]["evidence"]:
            assert "AI整理" not in article_title(evidence["title"])
            assert "未获人工逐项确认" not in article_text(evidence["text"])


def test_article_metadata_is_compact_without_changing_evidence_or_caveats():
    record = next(r for r in load_recordings() if r["case_id"] == "D07" and r["knowledge_mode"] == "llm_wiki_graph")
    original = deepcopy(record)
    shown = answer_markdown(record["result"]["answer"])
    assert "kv-cache.md" not in shown
    assert "AI整理词条" not in shown
    assert "非原论文逐字原文" not in shown
    for evidence in record["result"]["evidence"]:
        assert "AI整理词条" not in article_title(evidence["title"])
        assert "状态：" not in article_text(evidence["text"])
        assert article_text(article_text(evidence["text"])) == article_text(evidence["text"])
    assert "不能保证" in shown
    assert record == original
    literal = "`[标题（AI整理词条）, 完整知识词条；非原论文逐字原文, test.md]`"
    assert compact_citations(literal) == literal
    assert article_text("限制：未经人工确认的数据不可作为真值。") == "限制：未经人工确认的数据不可作为真值。"


@pytest.mark.parametrize("response_mode", ["LIVE", "OFFLINE_REPLAY"])
def test_shared_ui_sends_math_to_native_markdown_and_keeps_details(monkeypatch, response_mode):
    def idle_loop():
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.is_closed.return_value = False
        loop.is_running.return_value = False
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", idle_loop)
    app = AppTest.from_string('''
from frontend.chat import render_result
render_result({"mode":"OFFLINE_REPLAY","state":"completed","status":"answered",
 "answer":"LIVE — 已通过模型引用验证，尚非人工核实的事实\\n\\nReLU(x)=max(0,x)",
 "question":"MOCK display test","calls_used":0,"request_id":"MOCK-math"})
'''.replace('"mode":"OFFLINE_REPLAY"', '"mode":"' + response_mode + '"')).run()
    assert not app.exception
    assert any(r"$\operatorname{ReLU}(x)=\max(0,x)$" in m.value for m in app.markdown)
    assert not any("尚非人工核实的事实" in m.value for m in app.markdown)
    assert any(e.label == "运行详情与引用核对" for e in app.expander)


@pytest.mark.parametrize("response_mode", ["LIVE", "OFFLINE_REPLAY"])
@pytest.mark.parametrize("case_id", ["D01", "D03", "D07"])
@pytest.mark.parametrize("knowledge_mode", ["llm_wiki", "llm_wiki_graph"])
def test_article_details_omit_editorial_status_in_both_modes(monkeypatch, response_mode, case_id, knowledge_mode):
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.is_closed.return_value = False
    loop.is_running.return_value = False
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: loop)
    app = AppTest.from_string('''
from frontend.chat import render_result
from frontend.offline_replay import load_recordings
record = next(r for r in load_recordings() if r["case_id"] == "CASE_ID" and r["knowledge_mode"] == "KNOWLEDGE_MODE")
result = record["result"]
result["mode"] = "RESPONSE_MODE"
render_result(result)
'''.replace("RESPONSE_MODE", response_mode).replace("CASE_ID", case_id).replace("KNOWLEDGE_MODE", knowledge_mode)).run()
    assert not app.exception
    assert all("AI整理词条" not in option and "非原论文" not in option
               for box in app.selectbox for option in box.options)
    assert not any("状态：AI整理草稿" in t.value for t in app.text)
    assert not any("回答依据：AI整理" in c.value for c in app.caption)
    assert all("AI整理词条" not in m.value and "非原论文逐字原文" not in m.value for m in app.markdown)
