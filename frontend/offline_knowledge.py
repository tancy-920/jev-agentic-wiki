"""Provider-free complete Wiki browsing and actual manifest graph navigation."""

import json

from frontend.demo_catalog import CASE_TOPICS, search_articles, wiki_articles


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
