"""Shared evidence-anchored explanation policy for both decision providers."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AnswerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    style: Literal["beginner", "balanced", "technical"] = "balanced"
    language: Literal["auto", "zh", "zh-CN", "en"] = "auto"

    def snapshot(self, *, version="grounded-explanation-v3"):
        if version not in {
            "grounded-explanation-v1",
            "grounded-explanation-v2",
            "grounded-explanation-v3",
        }:
            raise ValueError("unknown answer policy version")
        values = {"version": version, **self.model_dump()}
        values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values

    def presentation_instructions(self):
        """Style/language only, for contracts that already state grounding rules."""
        style = {
            "beginner": "Assume the reader is learning this topic for the first time. Define unfamiliar terms, explain what happens and why, and connect the steps in plain language. Use a short explicitly marked illustrative analogy when it genuinely helps. Explain where an analogy stops matching the mechanism.",
            "balanced": "Give a clear self-contained explanation. Adapt detail to learning needs explicitly stated in the query. Define terms needed for the question, connect the main steps, and preserve essential conditions and limitations.",
            "technical": "Use precise terminology and equations where the supplied evidence establishes them. State assumptions and limitations; avoid unnecessary introductory material.",
        }[self.style]
        language = {
            "auto": "Follow the user's requested language; otherwise use the query's language.",
            "zh": "Write the explanation in Chinese.",
            "zh-CN": "Write every explanatory claim in Simplified Chinese, even when the evidence is English. Preserve the original meaning, negations, conditions and uncertainty across languages. Retain original paper titles, model names, mathematical notation and necessary English technical terms. Do not translate citation IDs or invent translated quotations. This language requirement applies before claim verification; never append a translation after verification.",
            "en": "Write the explanation in English.",
        }[self.language]
        return style + " " + language

    def instructions(self):
        return (
            "Explain in your own words, synthesizing the supplied evidence; quotations are optional, not required. "
            "Organize the claims as coherent explanatory paragraphs, each with citations covering every factual assertion in that paragraph. "
            "You may simplify presentation but preserve the evidence's meaning, scope and conditions. "
            "Accuracy takes precedence over ease of reading. Never drop a negation, qualifier, assumption, experimental setting, comparison dimension or uncertainty to make a simpler story. Distinguish what the papers measured from what you infer; explain essential caveats plainly instead of omitting them. Do not upgrade 'may' to 'always', correlation to causation, or a task-specific result to a general guarantee. "
            "Preserve the exact identities and variants of compared methods: a result about an unmodified method does not automatically describe a modified variant. Name both comparison targets explicitly instead of ambiguous phrases such as 'the two'. Do not transfer performance or complexity comparisons across pairs, variants or settings. Attribute hypotheses as hypotheses in every sentence where used, not just in a later caveat. Omit incidental comparisons that are unnecessary for the original question. "
            "Do not add unsupported numbers, universal guarantees, empirical results, sources or mechanisms. "
            "Mark illustrative analogies explicitly; they are teaching devices, not quotes or experiments reported by the paper. "
            "Ground an analogy's underlying factual relationships in its cited evidence. "
            "When explaining a supported consequence across passages, make that connection explicit; uncertain speculation must not be stated as a fact. "
            + self.presentation_instructions()
        )

    def heading(self, mode):
        if self.language == "zh-CN":
            return (
                "MOCK — 离线流程演示，不代表模型效果"
                if mode == "MOCK"
                else "LIVE — 已通过模型引用验证，尚非人工核实的事实"
            )
        return (
            "MOCK — offline workflow demonstration"
            if mode == "MOCK"
            else "LIVE — decision-verified answer (not human ground truth)"
        )

    def withheld(self, mode, reason):
        if self.language != "zh-CN":
            return f"{mode} — no verified answer ({reason})."
        messages = {
            "low_confidence": "证据或引用判断不够确定，暂不输出未经验证的结论。",
            "citation_failure": "引用未能支持答案，暂不输出未经验证的结论。",
            "conflicting_evidence": "证据存在尚未解决的冲突，暂时无法给出可靠结论。",
            "provider_failure": "模型服务调用失败，未生成通过验证的答案。",
            "retrieval_failure": "本地检索失败，未生成通过验证的答案。",
            "generation_failure": "答案生成失败，未生成通过验证的答案。",
            "invalid_decision": "模型决策未通过校验，已停止执行。",
            "translation_failure": "问题翻译未通过校验，已停止检索；请保留原问题并检查语言转换。",
        }
        return f"{mode} — " + messages.get(reason, "当前未能得到通过证据和引用验证的答案。")
