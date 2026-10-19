import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from jev_rag.config import LLMConfig


class ModelTier(StrEnum):
    FAST = "fast"
    STRONG = "strong"


class QueryProfile(BaseModel):
    """Predeclared input features, shared across paired systems; never gold answers."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    complex_multi_hop: bool = False
    hard_conflicting_evidence: bool = False


class ModelPolicy:
    version = "fixed-input-tier-v1"

    def __init__(self, config: LLMConfig, profile: QueryProfile | None = None):
        self.config = config.model_copy(deep=True)
        self.profile = profile or QueryProfile()

    def tier(
        self,
        task: Literal["rewrite", "summary", "decision", "reasoning", "generation", "translation"],
    ):
        if task not in {"rewrite", "summary", "decision", "reasoning", "generation", "translation"}:
            raise ValueError("unknown model task")
        if self.config.model == self.config.strong_model == "deepseek-flash":
            return ModelTier.FAST
        if task in {"generation", "reasoning"} and (
            self.profile.complex_multi_hop or self.profile.hard_conflicting_evidence
        ):
            return ModelTier.STRONG
        return ModelTier.FAST

    def model(self, task):
        return (
            self.config.strong_model if self.tier(task) == ModelTier.STRONG else self.config.model
        )

    def snapshot(self):
        values = {
            "version": self.version,
            "config": self.config.model_dump(),
            "profile": self.profile.model_dump(),
            "rule": ("all tasks=Flash fast; complexity does not upgrade model"
                     if self.config.model == self.config.strong_model == "deepseek-flash"
                     else "rewrite/summary/decision=fast; complex generation/reasoning=strong"),
        }
        values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values
