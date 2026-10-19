"""Research workspace with ten selectable offline examples and shared clean prose."""

import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from frontend.ui import sidebar_brand, theme
    from frontend.view import load_report, presentation, questions, replay, reports

    st.set_page_config(page_title="Jev Agentic Wiki", page_icon="✳", layout="wide", initial_sidebar_state="auto")
    st.set_option("client.toolbarMode", "viewer")
    theme()
    sidebar_brand()
    choices = ["实时对话", "离线模拟", "知识 Wiki", "资料库", "研究记录（实验回放）"]
    initial = {"simulate": "离线模拟", "replay": "研究记录（实验回放）", "wiki": "知识 Wiki", "library": "资料库"}.get(st.query_params.get("mode"), "实时对话")
    mode = st.sidebar.radio(
        "工作空间", choices, index=choices.index(initial),
        format_func={"实时对话": "◌   研究对话", "离线模拟": "▷   离线模拟 · 零 API", "知识 Wiki": "⌘   知识探索", "资料库": "▤   资料库", "研究记录（实验回放）": "◷   实验档案"}.get,
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="workspace-footer">本地研究空间<br>TRANSFORMER · PAPERS · CODE</div>', unsafe_allow_html=True)
    if mode == "离线模拟":
        from frontend.simulation import main as simulation_main

        simulation_main()
        return
    if mode == "知识 Wiki":
        from frontend.wiki import main as wiki_main

        wiki_main()
        return
    if mode == "资料库":
        from frontend.library import main as library_main

        library_main()
        return
    if mode == "实时对话":
        from frontend.chat import main as chat_main

        chat_main()
        return
    from frontend.ui import heading

    heading("RESEARCH ARCHIVE", "实验档案", "查看历史实验的原始记录、引用与处理过程。")
    st.caption("M6 实验回放：简体中文问题、原始证据和决策过程。此页面不读取密钥、不调用 API。")
    st.warning(
        "当前为已授权真实实验的回放界面，不是任意问题的在线生成。模型验证结果不等于人工质量评分。"
    )
    paths = reports()
    if not paths:
        st.info("尚无 M6 真实运行。先通过 benchmark-live 执行已授权的实验，再在这里检查。")
        return
    chosen = st.sidebar.selectbox("实验", paths, format_func=lambda p: p.parent.name)
    try:
        report = load_report(chosen)
        question_map = questions(report)
    except (OSError, ValueError, KeyError):
        st.error("本地运行记录或题目版本不匹配，未展示结果。")
        return
    variant = st.sidebar.selectbox("决策配置", report["variants"])
    seed = st.sidebar.selectbox("运行编号", report["seeds"])
    st.sidebar.write(f"进度：{len(report['records'])}/{report['expected_runs']} 次运行")
    st.sidebar.write(f"状态：{report['status']}")
    example = st.selectbox(
        "已冻结的中文问题", list(question_map), format_func=lambda q: question_map[q]
    )
    submitted = st.chat_input("输入已实测问题的原文进行回放；新问题不会被伪装成缓存答案")
    query = submitted if submitted else question_map[example]
    with st.chat_message("user"):
        st.text(query)
    row = replay(report, question_map, query, variant, seed)
    if row is None:
        with st.chat_message("assistant"):
            st.info(
                "这是实验回放，不是实时问答。该问题在此配置与运行编号下尚无真实记录。"
                "未调用 API，也未生成替代答案。要询问新问题，请在左侧切换到“实时对话”。"
            )
        return
    view = presentation(row)
    from frontend.math_display import answer_markdown

    with st.chat_message("assistant"):
        answer = view["answer"] or f"未输出答案。终止原因：{view['status']}"
        st.markdown(re.sub(r"\\*!\[", r"\\!\\[", answer_markdown(answer)))
    cols = st.columns(3)
    cols[0].metric("Agent 路由", view["route"] or "未到达")
    cols[1].metric("检索循环次数", view["iterations"] if view["iterations"] is not None else "未知")
    cols[2].metric("终止状态", view["status"])
    st.subheader("检索到的论文")
    if view["papers"]:
        st.dataframe(view["papers"], hide_index=True, width="stretch")
    else:
        st.write("未检索到论文，或在检索前停止。")
    st.subheader("Jev / LLM 决策时间线")
    st.caption(
        f"此组 Jev 任务：{', '.join(row['jev_tasks']) or '无（全部 LLM）'}。路由由程序确定。"
    )
    st.dataframe(view["timeline"], hide_index=True, width="stretch")
    st.subheader("最终证据")
    for chunk in view["evidence"]:
        with st.expander(f"{chunk['title']} · {chunk['section']} · p. {chunk['page']}"):
            st.text(chunk["text"])
            st.caption(f"chunk_id: {chunk['chunk_id']}")
    with st.expander("引用核对记录（模型预测，不是 ground truth）"):
        st.json(view["citations"])


if __name__ == "__main__":
    main()
