"""Local, read-only source library. Never starts a provider session."""

from html import escape
from pathlib import Path

import streamlit as st

from frontend.local_catalog import verified_store
from frontend.ui import heading, source_badge
from jev_rag.knowledge import inside

ROOT = Path(__file__).resolve().parents[1]


def filter_sources(sources, query, kind="all"):
    terms = query.casefold().split()
    return [s for s in sources if (kind == "all" or s.kind == kind)
            and all(t in " ".join([s.title, s.origin, *s.authors, s.source_id]).casefold() for t in terms)]


def main():
    heading("SOURCE LIBRARY", "资料库", "论文、官方文档与代码实现，都能回到原文。")
    try:
        store = verified_store(ROOT)
    except (OSError, ValueError):
        st.info("本地资料库尚未准备好，或来源版本已改变。请完成本地构建后再浏览。")
        return
    st.caption(f"{len(store.sources)} 份本地来源 · {len(store.chunks):,} 条原文片段 · 浏览不调用 API")
    search, category = st.columns([3, 1])
    query = search.text_input("查找资料", placeholder="搜索标题、作者、文件或来源…")
    kind = category.selectbox("来源类型", ["all", *sorted({s.kind for s in store.sources.values()})],
                              format_func=lambda k: "全部类型" if k == "all" else source_badge(k))
    sources = filter_sources(store.sources.values(), query, kind)
    st.caption(f"找到 {len(sources)} 份资料")
    if not sources:
        st.info("没有匹配资料。试试缩短关键词，或切换来源类型。")
        return
    selected = st.selectbox("选择资料", [s.source_id for s in sources],
                            format_func=lambda i: store.sources[i].title)
    source = store.sources[selected]
    with st.container(key="source_reading_header"):
        st.caption(f"{source_badge(source.kind)} / {source.year} / 本地收录")
        st.subheader(source.title)
        if source.authors:
            st.caption(" · ".join(source.authors))
        st.write(source.origin)
        if source.kind == "web_excerpt":
            st.caption("收录范围为指定网页摘录，不能视为完整文档。")
        if source.measurement_scope:
            st.info(source.measurement_scope)
    preview, provenance = st.tabs(["原文阅读", "来源信息"])
    with preview:
        chunks = [c for c in store.chunks.values() if c.identity == selected]
        if not chunks:
            st.info("当前索引未收录此来源的片段。")
        else:
            chunk_id = st.selectbox("定位原文片段", [c.chunk_id for c in chunks],
                                     format_func=lambda i: store.chunks[i].citation)
            chunk = store.chunks[chunk_id]
            if source.kind == "code":
                st.code(chunk.text, language="python", wrap_lines=True)
            else:
                st.markdown('<div class="source-prose">' + escape(chunk.text) + '</div>', unsafe_allow_html=True)
        with st.expander("查看全部规范化文本"):
            if st.checkbox("加载全文", key="library-full-" + selected):
                st.code(inside(store.root, source.normalized).read_text("utf-8"), language="text", wrap_lines=True)
    with provenance:
        st.caption("用于复核本地来源版本；这些记录不代表内容已获人工最终验收。")
        st.text(f"版本：{source.version}\n收录时间：{source.retrieved_at_utc}\n解析器：{source.parser}")
        st.code(f"snapshot SHA256: {source.snapshot_sha256}\nnormalized SHA256: {source.normalized_sha256}", language="text", wrap_lines=True)
