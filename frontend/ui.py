"""Product presentation only. No provider access or runtime policy changes."""

from html import escape
from pathlib import Path

import streamlit as st


def theme():
    st.markdown("<style>" + Path(__file__).with_name("styles.css").read_text("utf-8") + "</style>", unsafe_allow_html=True)


def heading(eyebrow, title, description):
    st.markdown(
        f'<div class="page-heading"><div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def welcome():
    st.markdown('''<div class="welcome">
      <div class="hero-symbol">✳</div><div class="eyebrow">YOUR RESEARCH, CONNECTED</div>
      <h1>从一个问题，<br>走近技术的本质。</h1>
      <p>连接论文、实现与实验记录。沿着原文理解 Transformer，<br class="desktop-break">让每一次深入，都有据可循。</p>
      <div class="hero-tags"><span>论文原文</span><span>代码实现</span><span>连续追问</span></div>
    </div>''', unsafe_allow_html=True)


def topic_cards():
    st.markdown('''<div class="section-label">从熟悉的概念开始 <span>EXPLORE THE KNOWLEDGE</span></div>
    <div class="topic-grid">
      <a class="topic-card" href="?mode=wiki&topic=attention" target="_self"><span class="card-icon">↗</span><small>01 / 基础原理</small><div class="card-title">理解注意力机制</div><p>从计算过程到结构设计，建立完整认识。</p><b>探索相关主题 →</b></a>
      <a class="topic-card" href="?mode=wiki&topic=KV%20Cache" target="_self"><span class="card-icon">⌘</span><small>02 / 推理优化</small><div class="card-title">深入 KV Cache</div><p>连接缓存、分组查询与推理效率。</p><b>探索相关主题 →</b></a>
      <a class="topic-card" href="?mode=wiki&topic=RMSNorm" target="_self"><span class="card-icon">◎</span><small>03 / 模型实现</small><div class="card-title">读懂归一化</div><p>对照公式与实现，理解设计的边界。</p><b>探索相关主题 →</b></a>
    </div>''', unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown('''<div class="brand"><div class="brand-mark"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M6 9c5-2 10-1 14 2 4-3 9-4 14-2v23c-5-2-10-1-14 2-4-3-9-4-14-2Z" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M20 11v23" stroke="currentColor" stroke-width="1.5"/><text x="20" y="27" text-anchor="middle" fill="currentColor" font-size="15" font-family="Georgia,serif">W</text></svg></div><div><strong>Wiki</strong><small>JEV · KNOWLEDGE ATLAS</small></div></div>''', unsafe_allow_html=True)


def source_badge(kind):
    return {"pdf": "论文", "code": "代码", "web_excerpt": "官方文档", "experiment": "实验记录"}.get(kind, kind)


def processing_indicator():
    """Activity only; does not invent stages, reasoning text or progress percentages."""
    st.markdown('''<div class="processing-indicator" role="status" aria-live="polite">
      <span class="activity-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <span>正在处理你的问题</span><small>请稍候</small>
    </div>''', unsafe_allow_html=True)
