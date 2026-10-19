"""Presentation of knowledge articles; original evidence and audit data stay intact."""

import re


def article_title(title):
    return re.sub(r"[（(]AI整理词条[）)]", "", title).strip()


def article_text(text):
    # Remove only the known editorial header, never substantive caveats.
    text = re.sub(
        r"(?m)^>\s*状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。\s*\n?",
        "", text, count=1,
    )
    return text.replace("以下是整理依据与核查入口，不是逐句正确性认证。", "")


def compact_citations(text):
    # Inline citations belong in the separate evidence panel, not answer prose.
    # Protect code and links; mathematical square brackets remain untouched.
    pattern = re.compile(
        r"\[(?:[^\[\]\n]*(?:\.md|PDF file p\.\s*\d+|,\s*p\.\s*\d+|,\s*L\d+[–-]L\d+))\]"
        r"|\[[^\[\]\n]*,\s*captured public-source excerpt,[^\[\]\n]*\]"
    )
    protected = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|\[[^\]\n]*\]\([^\n)]*\))")
    pieces = protected.split(text)
    for i in range(0, len(pieces), 2):
        pieces[i] = pattern.sub("", pieces[i])
        pieces[i] = re.sub(r"[ \t]{2,}", " ", pieces[i])
    return "".join(pieces)
