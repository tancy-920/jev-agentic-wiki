"""Provider-free complete Wiki browsing and actual manifest graph navigation."""

import json
from html import escape

from frontend.demo_catalog import CASE_TOPICS, search_articles, wiki_articles
from frontend.ui_icons import icon


def choose_topic(identifier):
    import streamlit as st

    st.session_state['offline_topic'] = identifier
    st.session_state['offline_wiki_search'] = ''


def browser():
    import streamlit as st

    articles = wiki_articles()
    by_id = {a['id']: a for a in articles}
    st.subheader('探索 Wiki 与关联图谱')
    query = st.text_input('搜索知识库', placeholder='输入关键词，例如 KV Cache、GQA、归一化',
                          key='offline_wiki_search',
                          on_change=lambda: st.session_state.pop('offline_topic', None))
    target = st.session_state.get('offline_topic')
    if not query.strip() and not target:
        st.caption(f'{len(articles)} 篇完整词条 · 关键词搜索 · 关联主题导航')
        return
    found = search_articles(articles, query)
    if not found:
        st.info('没有匹配词条，请尝试其他关键词。')
        return
    ids = [a['id'] for a in found]
    if target in ids:
        st.session_state['offline_article'] = target
        st.session_state.pop('offline_topic', None)
    elif st.session_state.get('offline_article') not in ids:
        st.session_state['offline_article'] = ids[0]
    selected = st.selectbox('匹配的 Wiki 词条', ids, key='offline_article',
                            format_func=lambda i: by_id[i]['title'])
    article = by_id[selected]
    neighbors = article['related']
    with st.expander('关联图谱 · 点击关联主题继续探索', expanded=True):
        dot = ['digraph wiki {', 'rankdir=LR; bgcolor="transparent";',
               'node [shape=box,style="rounded,filled",fillcolor="#edf4ef",color="#9ab9aa"];']
        for identifier in [selected, *neighbors]:
            dot.append(f'{json.dumps(identifier)} [label={json.dumps(by_id[identifier]["title"], ensure_ascii=False)}];')
        for identifier in neighbors:
            dot.append(f'{json.dumps(selected)} -> {json.dumps(identifier)};')
        st.graphviz_chart('\n'.join([*dot, '}']))
        st.caption('连线表示词条间的阅读关联。')
        for identifier in neighbors:
            st.button('↗ ' + by_id[identifier]['title'], key='offline_graph_' + identifier,
                      on_click=choose_topic, args=(identifier,))
    with st.expander('完整 Wiki · ' + article['title'], expanded=True):
        st.markdown(article['text'])


def related_entries(case_id, key):
    import streamlit as st

    by_id = {a['id']: a for a in wiki_articles()}
    st.markdown('**继续探索相关 Wiki →**')
    for identifier in CASE_TOPICS[case_id]:
        st.button(by_id[identifier]['title'] + ' ↗', key=f'wiki_entry_{key}_{identifier}',
                  on_click=choose_topic, args=(identifier,))


def atlas_graph(article, by_id):
    """Actual manifest links, with ordinary keyboard-accessible destinations."""
    nodes = ''.join(
        f'<a href="?mode=wiki&amp;topic={escape(identifier)}" target="_self">'
        f'{escape(by_id[identifier]["title"])}{icon("arrow")}</a>'
        for identifier in article['related']
    )
    return (f'<div class="atlas-map" aria-label="关联主题">'
            f'<div class="atlas-hub">{icon("book")}<strong>{escape(article["title"])}</strong></div>'
            f'<div class="atlas-links">{nodes}</div></div>')


def atlas():
    """Whole articles first; no full source corpus or planning vocabulary."""
    import streamlit as st

    try:
        articles = wiki_articles()
    except (OSError, ValueError, KeyError):
        st.error('本地词条缺失或版本校验失败，请检查知识库文件。')
        return
    by_id = {a['id']: a for a in articles}
    topic = st.query_params.get('topic', '')
    if topic != st.session_state.get('atlas_topic_param'):
        st.session_state['atlas_topic_param'] = topic
        st.session_state['atlas_query'] = '' if topic in by_id else topic
        if topic in by_id:
            st.session_state['atlas_selected'] = topic
    with st.container(key='atlas_search'):
        query = st.text_input('搜索完整词条', placeholder='输入概念、机制或实现，例如 GQA、位置编码', key='atlas_query')
        st.caption(f'{len(articles)} 篇完整词条 · 关联主题导航 · 本地浏览')
    found = search_articles(articles, query)
    if not found:
        st.info('没有匹配词条，请尝试其他关键词。')
        return
    ids = [a['id'] for a in found]
    selected = st.session_state.get('atlas_selected')
    if selected not in ids:
        selected = ids[0]
        st.session_state['atlas_selected'] = selected
    index, reading = st.columns([1, 2.6], gap='large')
    with index:
        st.markdown('**词条目录**')
        st.caption(f'{len(found)} 个匹配主题')
        with st.container(height=520, border=False, key='atlas_index'):
            for article in found:
                st.button(article['title'], key='atlas_topic_' + article['id'], width='stretch',
                          type='primary' if selected == article['id'] else 'secondary',
                          on_click=lambda value: st.session_state.update(atlas_selected=value), args=(article['id'],))
    with reading:
        article = by_id[selected]
        with st.container(key='atlas_connections'):
            st.markdown('**关联图谱**')
            st.caption('点击关联主题，沿知识脉络继续阅读。')
            st.markdown(atlas_graph(article, by_id), unsafe_allow_html=True)
        with st.container(key='atlas_article'):
            st.markdown(article['text'])
            st.download_button('下载完整词条', article['text'], file_name=selected+'.md', mime='text/markdown')
