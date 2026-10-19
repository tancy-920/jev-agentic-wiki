"""Bounded source-cited Wiki drafts; no automatic human attestation/public truth."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WikiStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=300)
    citations: list[str] = Field(min_length=1, max_length=3)


class WikiSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heading: Literal["概念与机制", "mygpt 实现与实验", "适用条件与限制"]
    statements: list[WikiStatement] = Field(min_length=1, max_length=2)


class WikiArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: list[WikiSection] = Field(min_length=1, max_length=3)

    def validate_sources(self, ids):
        headings = [s.heading for s in self.sections]
        if len(set(headings)) != len(headings):
            raise ValueError("duplicate Wiki headings")
        for section in self.sections:
            for statement in section.statements:
                if not set(statement.citations) <= set(ids):
                    raise ValueError("Wiki citation outside supplied source spans")
        return self
