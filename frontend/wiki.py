"""Read-only Wiki/source navigation; browsing never authorizes or spends API calls."""

import json
from pathlib import Path

import streamlit as st

from frontend.ui import heading
from jev_rag.knowledge import KnowledgeStore, inside, product_store_path

ROOT = Path(__file__).resolve().parents[1]


def browse_topics(store, query, path_id="all"):
    """Reading paths filter browsing, never suppress global concept search."""
    paths = {p.path_id: p for p in store.catalog.learning_paths}
    if path_id != "all" and path_id not in paths:
        raise ValueError("unknown reading path")
    if query.strip():
        return store.search(query, limit=8)
    return ([store.pages[i] for i in paths[path_id].page_ids] if path_id != "all"
            else list(store.pages.values()))


def graph_dot(store, page_id):
    pages = [store.pages[page_id], *store.neighbors([page_id], max_hops=1, limit=8)]
    ids = {p.page_id for p in pages}
    dot = ["digraph topics {", 'rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#edf4ef", color="#c7dcd0", fontcolor="#284c3e", fontname="Microsoft YaHei"]; edge [color="#99b9aa", fontcolor="#78998a", fontsize=10];']
    for page in pages:
        dot.append(
            f"{json.dumps(page.page_id)} [label={json.dumps(page.title, ensure_ascii=False)}];"
        )
    for edge in store.catalog.edges:
        if edge["from"] in ids and edge["to"] in ids:
            dot.append(
                f'{json.dumps(edge["from"])} -> {json.dumps(edge["to"])} [label="关联主题"];'
            )
    return "\n".join([*dot, "}"])


def main():
    heading("KNOWLEDGE EXPLORER", "知识探索", "从概念出发，连接原理、实现与实验。选择一条阅读路线，逐步建立你的知识地图。")
    st.caption("Transformer 原理、优化技术与 mygpt：Wiki → 原文 → 实现 → 实验。浏览不调用 API。")
    try:
        store = KnowledgeStore(product_store_path(ROOT))
    except (OSError, ValueError):
        st.info("Wiki 来源尚未准备好，或来源版本已改变；请完成本地构建后再浏览。")
        return
    st.caption(f"{len(store.pages)} 个主题 / {len(store.sources)} 份来源 / {len(store.chunks):,} 条原文")
    search_column, route_column = st.columns([3, 2])
    query = search_column.text_input(
        "搜索 Wiki 概念", placeholder="例如：KV Cache、GQA、FlashAttention、交叉注意力",
        value=st.query_params.get("topic", ""),
    )
    paths = {p.path_id: p for p in store.catalog.learning_paths}
    path_id = route_column.selectbox("阅读路线", ["all", *paths],
        format_func=lambda i: "全部主题" if i == "all" else paths[i].title)
    if path_id != "all":
        st.caption(paths[path_id].guide)
        st.caption("建议阅读顺序：" + " → ".join(store.pages[i].title for i in paths[path_id].page_ids))
    if query.strip():
        # Search remains global: selecting a route must not hide another topic.
        st.caption("搜索覆盖全部主题；清空搜索后按所选路线浏览。")
    found = browse_topics(store, query, path_id)
    options = [p.page_id for p in found]
    if not options:
        st.info("没有对应主题。可清空搜索浏览目录；当前不涵盖全部 Transformer 技术。")
        return
    selected = st.selectbox("主题", options, format_func=lambda i: store.pages[i].title)
    page = store.pages[selected]
    st.subheader(page.title)
    st.caption("别名：" + " / ".join(page.aliases))
    st.write(page.guide)
    article_root = ROOT / "knowledge/llm_wiki"
    if (article_root / "manifest.json").exists():
        from jev_rag.llm_wiki import ArticleStore

        try:
            articles = ArticleStore(article_root, source_store=store)
            if selected in articles.pages:
                st.markdown(articles.texts[selected])
                st.download_button("下载完整 Markdown 词条", articles.texts[selected],
                                   file_name=selected + ".md", mime="text/markdown")
        except (OSError, ValueError):
            st.warning("完整词条与当前来源版本不一致，暂不显示；下方保留原文导航。")
    st.caption("AI 编辑草稿与原文导航；不表示人工真值确认。回答仍须核对实际来源。")
    reading, connections, originals = st.tabs(["主题导读", "关联图谱", "原文来源"])
    with reading:
        for note in page.coverage_notes:
            st.warning(f"{note.concept}：{note.limitation}")
            for identifier in note.evidence_ids:
                chunk = store.chunks[identifier]
                with st.expander("范围说明原文 · " + chunk.citation):
                    st.code(chunk.text, language="text")
        for section in page.reading_sections:
            st.markdown("**" + section.heading + "**")
            st.write(section.guide)
            for identifier in section.evidence_ids:
                chunk = store.chunks[identifier]
                source = store.sources[chunk.identity]
                with st.expander(chunk.citation):
                    st.caption(f"版本：{source.version}；来源片段 ID：{identifier}")
                    if source.measurement_scope:
                        st.caption(source.measurement_scope)
                    st.code(chunk.text, language="python" if source.kind == "code" else "text")
        articles = store.root / "articles.json"
        if articles.exists():
            data = json.loads(articles.read_text("utf-8"))
            if data.get("wiki_sha256") == store.binding()["wiki.json"]:
                article = data.get("pages", {}).get(selected)
                if article:
                    for section in article.get("sections", []):
                        st.markdown("**" + section["heading"] + "**")
                        for statement in section["statements"]:
                            st.text(statement["text"])
                            for identifier in statement["citations"]:
                                chunk = store.chunks.get(identifier)
                                if chunk is None:
                                    st.warning("引用不在当前来源版本中，此草稿不可核对。")
                                    continue
                                with st.expander("引用原文 · " + chunk.citation):
                                    st.code(chunk.text, language="text")
                                    st.caption("来源片段 ID：" + identifier)
                    st.caption("文章状态：AI_DRAFT；结构绑定通过不等于语义审核通过。")
    with connections:
        st.markdown("**关联主题图谱**")
        st.caption("箭头表示阅读关联，不表示因果、实现继承或性能优势。")
        st.graphviz_chart(graph_dot(store, selected))
        st.write("关联阅读：" + "、".join(store.pages[i].title for i in page.related))
    with originals:
        st.markdown("**原理、实现与实验来源**")
        for source_id in page.source_ids:
            source = store.sources[source_id]
            with st.expander(f"{source.title} · {source.kind}"):
                st.write(source.origin)
                st.caption(f"版本：{source.version}；快照 SHA256：{source.snapshot_sha256}")
                if source.measurement_scope:
                    st.info(source.measurement_scope)
                if source.kind == "web_excerpt":
                    st.caption("仅收录官方网页的指定摘录；不是完整论文或完整文档。")
                source_chunks = [
                    store.chunks[i] for i in page.evidence_ids if store.chunks[i].identity == source_id
                ]
                for chunk in source_chunks:
                    st.caption(chunk.citation + f" · {chunk.chunk_id}")
                    st.code(chunk.text, language="python" if source.kind == "code" else "text")
                if st.checkbox("查看该来源全部规范化文本", key=f"full-source-{selected}-{source_id}"):
                    st.code(inside(store.root, source.normalized).read_text("utf-8"), language="text")
