"""Interactive offline chat using the same result renderer as live chat."""

from time import monotonic
from uuid import uuid4

import streamlit as st

from frontend.chat import render_result
from frontend.demo_catalog import FEATURED_CASES, featured_recordings
from frontend.offline_knowledge import browser, related_entries
from frontend.offline_replay import MODES, load_recordings, replay_question
from frontend.ui import heading, processing_indicator


def workspace():
    return st.session_state.setdefault("offline_workspace", dict(
        results=[], conversation=str(uuid4()), pending=None,
    ))


def finish_pending(state, *, now=None):
    pending = state.get("pending")
    if pending is None or (monotonic() if now is None else now) < pending["ready_at"]:
        return False
    result = pending["result"]
    if not any(r["request_id"] == result["request_id"] for r in state["results"]):
        state["results"].append(result)
    state["pending"] = None
    return True


@st.fragment(run_every=0.5)
def poll_replay():
    if finish_pending(workspace()):
        st.rerun(scope="app")
    processing_indicator()
    st.caption("正在载入本地历史记录 · 仅演示等待动画，未运行检索或调用模型")


def example_gallery(recordings, busy):
    st.subheader("全部示例问题 · 10 题")
    st.caption("选择一个感兴趣的问题，查看完整解答，再继续探索相关 Wiki。")
    for index, (case_id, (title, category)) in enumerate(FEATURED_CASES.items(), 1):
        example = next(r["question"] for r in recordings if r["case_id"] == case_id)
        with st.container(border=True):
            st.caption(f"{index:02d} / {category}")
            st.subheader(title)
            st.write(example)
            if st.button("选择此问题 →", key="offline_card_" + case_id, disabled=busy):
                st.session_state["offline_example"] = example
                st.session_state["offline_gallery"] = False
                st.session_state["offline_knowledge_mode"] = st.session_state.get("offline_saved_mode", "llm_wiki_graph")
                st.rerun()


def main():
    heading("KNOWLEDGE PLAYGROUND", "离线知识探索", "搜索完整 Wiki 与关联图谱，或选择十个主题查看历史解答。")
    st.info("历史回放模式：不连接问答后端、不读取密钥、不消耗额度。仅已有记录的问题可回放；输入新问题不会转为在线生成。")
    try:
        recordings = featured_recordings(load_recordings())
    except (OSError, ValueError, KeyError, TypeError):
        st.error("本地回放资料缺失或校验失败，未启用模拟。不会回退到在线 API。")
        return
    state = workspace()
    finish_pending(state)
    busy = state["pending"] is not None
    if st.session_state.get("offline_example") not in {r["question"] for r in recordings}:
        st.session_state["offline_example"] = recordings[0]["question"]
    visible = [r for r in state["results"] if r["conversation_id"] == state["conversation"]]
    if st.session_state.get("offline_gallery"):
        if st.button("← 返回知识探索", key="offline_gallery_back"):
            st.session_state["offline_gallery"] = False
            st.rerun()
        example_gallery(recordings, busy)
        return
    browser()
    if st.button("浏览全部示例问题 · 10 题 ↗", key="offline_examples", width="stretch"):
        st.session_state["offline_saved_mode"] = st.session_state.get("offline_knowledge_mode", "llm_wiki_graph")
        st.session_state["offline_gallery"] = True
        st.rerun()
    with st.sidebar.expander("模拟设置", expanded=True):
        st.metric("本次 API 请求", 0)
        choices = list(dict.fromkeys(r["question"] for r in recordings))
        question = st.session_state["offline_example"]
        available = [m for m in MODES if any(r["question"] == question and r["knowledge_mode"] == m for r in recordings)]
        if st.session_state.get("offline_knowledge_mode") not in available:
            st.session_state["offline_knowledge_mode"] = available[-1]
        mode = st.selectbox("知识检索方式", available, format_func=MODES.get,
                            key="offline_knowledge_mode", disabled=busy)
        st.selectbox("解释深度", ["标准解释（历史记录）"], disabled=True, key="offline_depth")
        st.caption("仅回放记录实际采用的深度；不会重新生成其他深度或模拟 JEV 概率。")
        st.caption(f"共 {len(choices)} 个历史问题；检索方式仅显示该问题实际保存的配置。")
        st.caption(question)
        send = st.button("发送示例问题", type="primary", disabled=busy)
    if st.button("发送选中问题 →", type="primary", disabled=busy):
        send = True
    if st.sidebar.button("开始新对话", key="offline_new", disabled=busy):
        state["conversation"] = str(uuid4())
        st.rerun()
    if st.sidebar.button("清空模拟记录", key="offline_clear", disabled=busy):
        state.update(results=[], conversation=str(uuid4()), pending=None)
        st.rerun()
    groups = {}
    for result in state["results"]:
        groups.setdefault(result["conversation_id"], result["question"])
    if groups:
        st.sidebar.caption("最近模拟对话 · 与实时会话分开保存")
        for identifier, title in reversed(list(groups.items())):
            if st.sidebar.button(title[:24], key="offline_history_" + identifier, disabled=busy):
                state["conversation"] = identifier
                st.rerun()
    visible = [r for r in state["results"] if r["conversation_id"] == state["conversation"]]
    for result in visible:
        render_result(result)
        case = next((r["case_id"] for r in recordings if r["question"] == result["question"]), None)
        if case:
            related_entries(case, result["request_id"])
    if busy:
        with st.chat_message("user"):
            st.text(state["pending"]["result"]["question"])
        poll_replay()
    if send:
        submitted = question
        result = replay_question(recordings, submitted, mode, conversation_id=state["conversation"])
        state["pending"] = dict(result=result, ready_at=monotonic() + 0.8)
        st.rerun()
