"""Live Chinese chat UI through an explicitly authorized fixed-local backend."""

import math
import re
from uuid import uuid4

import httpx
import streamlit as st

from frontend.citation_display import article_text, article_title
from frontend.client import LocalClient
from frontend.math_display import answer_markdown
from frontend.session import authorize_and_start, launch_status
from frontend.ui import heading, topic_cards, welcome
from jev_rag.schema import source_location

VARIANT_LABELS = {
    "A": "LLM 研究助手（本地知识库）",
    "A0": "常规 Agentic RAG（LLM 决策基线）",
    "A1": "JEV 增强（证据辅助排序）",
    "A2": "A2 · Jev 相关性 + 证据充分性",
    "A3": "A3 · Jev 相关性 + 证据 + 循环控制",
    "A4": "A4 · Jev 相关性、意图与引用（研究配置）",
}

STYLE_LABELS = {"balanced": "标准解释", "beginner": "入门解释", "technical": "技术细节"}

VARIANT_DESCRIPTIONS = {
    "A": "本地混合检索与重排；LLM解析追问、生成和核验答案，不调用辅助排序Judge。",
    "A0": "所有模型决策由 LLM 完成，用于常规 Agentic RAG 对照。",
    "A1": "JEV 辅助证据排序；不确定时保留本地次序。生成与最终核验共用 LLM，整体收益待验收。",
    "A2": "历史实验增加 JEV 证据充分性判断。当前对话有证据时直接生成候选，再核验完整答案，因此这一步不会触发。",
    "A3": "历史实验再增加 JEV 循环判断。当前对话有证据时直接生成候选，因此这两项额外判断不会触发。",
    "A4": "JEV 判断相关性、意图与逐项引用；整题完整性由共同 LLM 核验，本地路由由程序确定。尚未证明整体优势。",
}


@st.fragment(run_every=2)
def startup_progress():
    """Poll readiness only; never create an authorization or submit a question."""
    try:
        health = LocalClient().health()
        if not health.get("closed"):
            st.rerun(scope="app")
            return
    except Exception:
        pass
    if launch_status()["state"] != "starting":
        st.rerun(scope="app")
        return
    st.info("正在连接或加载本地检索模型。每2秒自动检查，完成后显示输入框；不调用模型 API。")
    if st.button("检查启动进度"):
        st.rerun(scope="app")


def activation_panel():
    st.info("实时对话尚未启用。可在这里授权并启动，无需打开终端。")
    try:
        state = launch_status()
    except Exception:
        st.error("启动记录暂不可读取；未创建新授权，也未调用模型。")
        return
    if state["state"] == "starting":
        startup_progress()
        return
    if state["state"] == "stopped":
        if state.get("reason") == "session_closed":
            st.info("上次会话已结束。可在下方明确授权启动新会话；旧调用记录和已关闭余额保留。")
        else:
            st.warning(
                "上次服务已停止或启动失败。请检查本地模型与凭据；未完成结算的会话不会自动重置额度。"
            )
    limit = st.selectbox("本次 API 请求上限", [40, 20])
    st.caption("按实际请求计数，失败请求也计入；到达上限后停止。费用取决于模型用量，当前无法预估。")
    st.caption(
        "授权范围：问题、最近三轮已完成对话、AI整理的完整Wiki词条，以及原文模式下论文/官方文档和mygpt技术代码及实验JSON片段发送给配置的LLM提供方；密钥、权重和训练数据不进入提示词。默认使用完整词条LLM-Wiki。"
    )
    permit_jev = st.checkbox("允许可选JEV回答预算试用（默认关闭）", value=False)
    if permit_jev:
        st.caption("另允许问题与最多3条检索原文发送给配置的JEV提供方。仅直接事实题可能采用一次Flash整答核验；机制解释仍按标准核验。所有LLM阶段仅使用Flash，整体提速与轻量负例质量尚未验收。")
    if st.button("授权并启动实时对话", type="primary"):
        try:
            authorize_and_start(limit, approved=True, answer_budget_jev_approved=permit_jev)
            st.rerun()
        except Exception:
            st.error("未能启动：已有启动任务、未结算会话或本地运行环境不可用。未自动追加额度。")


def timing_label(result):
    value = result.get("processing_latency_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return f"{value / 1000:.2f} 秒"
    return "未记录"


def comparison_rows(results):
    """Show recorded same-input runs only; no automatic rerun or quality inference."""
    eligible = [
        r
        for r in results
        if r.get("variant") in {"A0", "A1"}
        and r.get("presentation_profile") is not None
        and r.get("application_source_sha256")
        and r.get("shared_policy_sha256")
        and r.get("question")
    ]
    if not eligible:
        return []
    latest = eligible[-1]
    keys = (
        "question",
        "presentation_profile",
        "application_source_sha256",
        "shared_policy_sha256",
        "mode",
        "conversation_context_sha256",
    )
    matched = [r for r in eligible if all(r.get(k) == latest.get(k) for k in keys)]
    if {r["variant"] for r in matched} != {"A0", "A1"}:
        return []
    return [
        {
            "配置": VARIANT_LABELS[r["variant"]],
            "状态": r.get("status", r["state"]),
            "处理耗时": timing_label(r),
            "实际请求数": r.get("calls_used"),
        }
        for r in matched
    ]


def render_result(result):
    with st.chat_message("user"):
        st.text(result.get("question", "已提交的问题"))
    with st.chat_message("assistant"):
        offline = result.get("mode") == "OFFLINE_REPLAY"
        term_trace = next((s.get("query_terminology") for s in reversed(result.get("knowledge_navigation", []))
                           if s.get("query_terminology")), None)
        answer = result.get("answer") or f"未输出答案：{result.get('status', result['state'])}"
        display_answer = answer_markdown(answer) if result.get("mode") != "MOCK" else answer
        if result.get("mode") == "MOCK":
            st.text(answer)
        else:
            # Model prose may use Markdown, but must not load remote image URLs.
            st.markdown(re.sub(r"\\*!\[", r"\\!\\[", display_answer))
        diagnostic = result.get("failure_diagnostic")
        if diagnostic:
            st.info(diagnostic["message"] + " " + diagnostic["next_step"])
    if offline:
        st.caption(f"{len(result.get('evidence', []))} 条引用 · {result.get('status')} · 历史模型结果，非本次重新核验")
    else:
        st.caption(f"{timing_label(result)} · {len(result.get('evidence', []))} 条原文 · {result.get('mode', '未知')} · AI 回答，请结合引用判断")
    article_evidence = result.get("verification_scope") == "article_faithfulness_not_original_source_certification"
    if result.get("answer"):
        evidence_text = "\n\n".join(
            (f"[{i}] {article_title(c.get('title', ''))}\n{article_text(c.get('text', ''))}"
             if article_evidence else f"[{i}] {c.get('title', '')} / {c.get('section', '')}\n{c.get('text', '')}")
            for i, c in enumerate(result.get("evidence", []), 1)
        )
        export = f"# {result.get('question', '研究记录')}\n\n{display_answer}\n\n## 本次检索原文\n\n{evidence_text}\n\nAI 生成，未经人工最终验收。"
        if offline:
            export = "离线模拟／历史回放 · 本次 API 请求为 0；未执行新检索、生成或核验。\n\n" + export
            export += "\n\n## 回放来源\n\n" + str(result.get("replay_provenance", {}))
        st.download_button("↓ 导出研究记录", export, file_name="jev-research.md", mime="text/markdown",
                           key="export-" + result["request_id"] if result.get("request_id") else None)
    with st.expander("运行详情与引用核对"):
        if term_trace and term_trace.get("corrections"):
            st.text("本次检索词解释：" + "；".join(
                f"{c['original']} → {c['source_alias']}" for c in term_trace["corrections"][:8]
            ) + "。原问题保留，此解释不作为事实证据。")
        cols = st.columns(4)
        cols[0].metric("处理耗时", "离线回放" if offline else timing_label(result))
        cols[1].metric("实际 API 请求", result.get("calls_used", "未知"))
        cols[2].metric("检索循环", result.get("iterations", "未知"))
        cols[3].metric("终止状态", result.get("status", result["state"]))
        st.caption("下列流程与核对来自历史记录，本次未执行；历史耗时不能作为模拟速度或当前版本性能。"
                   if offline else "模型核对不是人工 ground truth；耗时不含浏览器刷新等待。失败运行保留。")
        if offline:
            st.json(result.get("replay_provenance", {}))
        if result.get("budget_routing"):
            st.json(result["budget_routing"])
        if result.get("provider_timing"):
            st.json(result["provider_timing"])
        st.dataframe(result.get("papers", []), hide_index=True, width="stretch")
        st.dataframe(result.get("timeline", []), hide_index=True, width="stretch")
        st.json(result.get("citations", []))
    evidence = result.get("evidence", [])
    if evidence:
        with st.expander(f"{'引用与完整词条' if article_evidence else '引用与原文'} · {len(evidence)} 条", expanded=False):
            selected = st.selectbox(
                "选择知识词条" if article_evidence else "选择原文片段", range(len(evidence)),
                format_func=lambda i: (f"{i + 1}. {article_title(evidence[i]['title'])}" if article_evidence
                                       else f"{i + 1}. {evidence[i]['title']} · {evidence[i]['section']}"),
                # Refresh serialized option labels in pre-existing browser sessions.
                key="evidence-display-v2-" + result["request_id"] if result.get("request_id") else None,
            )
            chunk = evidence[selected]
            location = source_location(text=chunk["text"], source_kind=chunk.get("source_kind"),
                                       locator=chunk.get("locator"), page=chunk.get("page"))
            st.caption(location)
            st.text(article_text(chunk["text"]) if article_evidence else chunk["text"])
    if result.get("knowledge_navigation"):
        with st.expander("本次 Wiki 与图谱检索路径"):
            for step in result["knowledge_navigation"]:
                st.write("Wiki：" + ", ".join(step.get("wiki_pages", step.get("seeds", []))))
                st.write("关联主题：" + ", ".join(step["graph_neighbors"]))
                st.caption("完整词条直接提供回答知识；图谱用于补充关联词条。"
                           if step.get("complete_articles") else "旧版Wiki提供原文定位；编辑说明和图谱边不充当回答证据。")


def sync_pending(client):
    """Read the same request ID; never resubmit or spend a model call while polling."""
    pending = st.session_state.get("chat_pending")
    if not pending:
        return False
    result = client.result(pending)
    if result["state"] not in {"completed", "failed"}:
        return False
    if not any(r.get("request_id") == pending for r in st.session_state.chat_results):
        st.session_state.chat_results.append(result)
    st.session_state.chat_pending = None
    return True


@st.fragment(run_every=2)
def poll_pending(client):
    if not st.session_state.get("chat_pending"):
        return
    try:
        if sync_pending(client):
            st.rerun()
    except Exception:
        st.error("进度查询暂不可用，原请求仍保留，不会重复提交。")
    from frontend.ui import processing_indicator

    processing_indicator()
    st.caption("检索、生成或核对中；完成后自动显示答案。")
    if st.button("刷新处理进度"):
        st.rerun()


def conversation_history(*, disabled=False):
    """Review existing browser-session messages without authorizing model access."""
    request_conversations = st.session_state.get("chat_request_conversations", {})
    def conversation_of(result):
        return result.get("conversation_id") or request_conversations.get(result.get("request_id"))

    groups = {}
    for result in st.session_state.get("chat_results", []):
        conversation = conversation_of(result)
        if conversation:
            groups.setdefault(conversation, result.get("question", "研究对话"))
    if groups:
        st.sidebar.caption("最近对话 · 当前浏览器会话内保留")
        for conversation, title in reversed(list(groups.items())):
            if st.sidebar.button(title[:24] + ("…" if len(title) > 24 else ""), key="conversation-" + conversation,
                                 disabled=disabled,
                                 use_container_width=True):
                st.session_state.chat_conversation = conversation
                st.rerun()
    visible = [r for r in st.session_state.get("chat_results", [])
               if conversation_of(r) == st.session_state.chat_conversation]
    return visible


def main():
    if st.session_state.get("chat_reconnect"):
        from frontend.session import resume_closed_session
        try:
            resume_closed_session(st.session_state.chat_reconnect, approved=True)
        except ValueError as error:
            st.info(f"正在等待旧连接安全关闭：{error}")
            st.button("刷新恢复状态", on_click=lambda: None)
            return
        st.session_state.chat_reconnect = None
        st.info("正在恢复连接，剩余额度与历史已保留。")
        startup_progress()
        return
    try:
        client = LocalClient()
        health = client.health()
        if health.get("closed"):
            raise ValueError("closed session")
    except Exception:
        if st.session_state.get("chat_results"):
            heading("RESEARCH CHAT", "研究对话", "继续查阅当前浏览器中保留的研究记录。")
            if st.sidebar.button("开始新对话"):
                st.session_state.chat_conversation = str(uuid4())
                st.rerun()
            visible = conversation_history()
            for result in visible:
                render_result(result)
            if not visible:
                welcome()
        else:
            welcome()
        with st.expander("连接研究助手 · 启用在线问答", expanded=False):
            activation_panel()
        topic_cards()
        st.caption("知识探索与资料库可直接浏览。在线问答在你明确授权后开启。")
        return
    if st.session_state.get("chat_session") != client.session_id:
        transferred = (health.get("conversation_transfer") or {}).get("from_session") == st.session_state.get("chat_session")
        if not st.session_state.get("chat_results"):
            restored = []
            for request_id in (health.get("conversation_transfer") or {}).get("completed_request_ids", [])[-50:]:
                result = client.result(request_id)
                if result.get("state") == "completed":
                    restored.append(result)
            if restored:
                st.session_state.chat_results = restored
                st.session_state.chat_conversation = restored[-1].get("conversation_id") or str(uuid4())
                transferred = True
        st.session_state.update(
            chat_session=client.session_id,
            chat_results=st.session_state.get("chat_results", []),
            chat_pending=None,
            chat_conversation=st.session_state.get("chat_conversation") if transferred and st.session_state.get("chat_conversation") else str(uuid4()),
        )
    try:
        sync_pending(client)
    except Exception:
        pass  # The pending fragment keeps the original request and reports query failure.
    heading("RESEARCH CHAT", "研究对话", "提出问题，沿着论文与代码深入。支持连续追问，引用可展开查看。")
    if health.get("source_current") is False:
        st.warning("后台版本需要更新，输入暂不可用。恢复连接会保留未用额度和已完成对话，不会新增 API 额度。")
        if st.button("恢复连接 · 保留剩余额度", disabled=health["busy"] or bool(st.session_state.chat_pending)):
            st.session_state.chat_reconnect = client.session_id
            client.close_session()
            st.rerun()
    with st.sidebar.expander("对话设置与使用额度", expanded=False):
        st.metric("本次剩余 API 请求", health["remaining_calls"])
        source_ready = health.get("source_current") is not False
        if not source_ready:
            st.warning("程序或配置已更新，当前后台仍绑定旧版本。请使用页面上方的恢复连接按钮；保留未用额度，不追加调用。")
        minimum_calls = health.get("minimum_question_calls", 1)
        budget_ready = health["remaining_calls"] >= minimum_calls
        if not budget_ready:
            st.info(
                f"本地授权预算不足：剩余 {health['remaining_calls']} 次，"
                f"新问题需至少预留 {minimum_calls} 次请求；当前不会提交新问答。"
            )
        st.caption(
            "可以连续提问，单题最多 20 次请求，实际受本次剩余额度限制。"
            "额度耗尽时停止，不展示尚未完成核验的答案。"
        )
        variants = health.get("allowed_variants", ["A0", "A1"])
        variant = st.selectbox("回答引擎", variants, format_func=VARIANT_LABELS.get)
        knowledge_mode = "source"
        if variant == "A" and health.get("knowledge_modes"):
            knowledge_mode = st.selectbox(
                "知识检索方式",
                health["knowledge_modes"],
                index=health["knowledge_modes"].index("llm_wiki_graph")
                if "llm_wiki_graph" in health["knowledge_modes"]
                else 0,
                format_func={
                    "source": "常规 RAG（原始来源）",
                    "wiki": "旧版：Wiki 导航增强 RAG",
                    "wiki_graph": "旧版：Wiki＋图谱导航增强 RAG",
                    "llm_wiki": "LLM-Wiki（完整知识词条）",
                    "llm_wiki_graph": "LLM-Wiki＋关联图谱",
                }.get,
            )
            if knowledge_mode.startswith("llm_wiki"):
                st.caption("轻量匹配完整词条直接作答；图谱补充关联词条。不会运行全库 chunk 召回与重排。词条为 AI 整理草稿，核验针对词条内容，非原论文逐字认证。")
            else:
                st.caption("原文 RAG：检索论文、文档和代码片段；旧版 Wiki 仅辅助导航。")
        st.caption(VARIANT_DESCRIPTIONS[variant])
        use_answer_budget = False
        if variant == "A" and health.get("answer_budget_jev_approved") is True:
            use_answer_budget = st.checkbox("JEV回答预算试用", value=False)
            st.caption("事实题可用轻量整答核验；不确定回标准。机制题不额外调用JEV，所有回答及核验仅使用Flash。尚未证明稳定提速。")
        jev_available = health.get("jev_ready", False)
        if variant == "A1" and not jev_available:
            st.info("JEV排序仍待开发筛查，暂未开放调用；可以继续测试常规RAG。")
        if "A1" in variants:
            st.caption(
                "JEV 模式只替换辅助排序的相关性提供方，两种研究模式共用检索、答案生成及质量检查。"
                "这是待验证的研究配置，尚未证明整体优势。"
            )
        style = st.selectbox(
            "解释深度", ["balanced", "beginner", "technical"], format_func=STYLE_LABELS.get
        )
        multi = False
        conflict = False
        st.caption("规划、生成、核验与修复均使用 DeepSeek Flash，不升级模型。")
        threshold = health.get("decision_threshold")
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and math.isfinite(threshold):
            st.caption(f"本会话核验门槛 {threshold:.2f}：来源与整题都须支持；明确矛盾、无支持和无效引用仍拦截。")
    if st.sidebar.button(
        "开始新对话", disabled=health["busy"] or bool(st.session_state.chat_pending)
    ):
        st.session_state.chat_conversation = str(uuid4())
        st.rerun()
    if st.sidebar.button("结束本次 API 授权", disabled=health["busy"]):
        try:
            client.close_session()
            st.rerun()
        except Exception:
            st.error("结束请求未完成。请刷新确认状态；不会自动创建新授权。")
    visible = conversation_history(disabled=health["busy"] or bool(st.session_state.chat_pending))
    if not visible and not st.session_state.chat_pending:
        welcome()
        st.caption("试着问：GQA 如何减少 KV Cache？也可以粘贴你想理解的技术问题。")
    for result in visible:
        render_result(result)
    comparison = comparison_rows(st.session_state.chat_results)
    if comparison:
        with st.expander("同题实验对照"):
            st.dataframe(comparison, hide_index=True, width="stretch")
            st.caption("仅比较相同输入与策略的已有运行；不能据此证明正确率或稳定提速。")
    pending = st.session_state.chat_pending
    if pending:
        if st.session_state.get("chat_pending_question"):
            with st.chat_message("user"):
                st.text(st.session_state.chat_pending_question)
        poll_pending(client)
    query = st.chat_input(
        "输入论文问题或继续追问（最多 3,000 字）",
        disabled=bool(pending)
        or health["busy"]
        or not budget_ready
        or not source_ready
        or (variant == "A1" and not jev_available),
    )
    if query:
        identifier = str(uuid4())
        # Keep identifier before HTTP: a timeout must not silently create a second paid request.
        st.session_state.chat_pending = identifier
        st.session_state.setdefault("chat_request_conversations", {})[identifier] = st.session_state.chat_conversation
        st.session_state.chat_pending_question = query
        try:
            client.submit(
                {
                    "request_id": identifier,
                    "question": query,
                    "conversation_id": st.session_state.chat_conversation,
                    "variant": variant,
                    "knowledge_mode": knowledge_mode,
                    "style": style,
                    "complex_multi_hop": multi,
                    "hard_conflicting_evidence": conflict,
                    "answer_budget_router": "jev" if use_answer_budget else "off",
                    "light_verification": use_answer_budget,
                }
            )
            st.rerun()
        except httpx.HTTPStatusError as error:
            if 400 <= error.response.status_code < 500:
                st.session_state.chat_pending = None
                detail = error.response.json().get("detail")
                st.error(
                    detail
                    if isinstance(detail, str)
                    else "请求未被接受（输入、预算或忙碌状态），未自动调用模型。请检查后重新提交。"
                )
            else:
                st.error("提交状态未知。保留原 request_id，请刷新查询；不会自动换 ID 再扣费。")
        except Exception:
            st.error("提交状态未知或被拒绝。请刷新查询此请求；不会自动换 ID 再扣费。")
