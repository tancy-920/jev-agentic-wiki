"""Bound original questions to a fallible, language-only retrieval view."""

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def has_chinese(text):
    return any("\u3400" <= c <= "\u9fff" for c in text)


# Product language normalization only, never aliases for models or operations.
# Inflection preserves the noun; numerical/negative constraints remain separate.
TECHNICAL_NOUNS = frozenset({
    "token", "tensor", "embedding", "layer", "head", "position", "parameter",
    "weight", "logit", "encoder", "decoder", "key", "query",
})


def protected_spans(query):
    """Conservative literal clauses to audit; not a semantic equivalence proof."""
    return [
        span.strip()
        for span in re.split(r"[，。；！？\n;!?]+", query)
        if re.search(
            r"不要|不得|不能|不应|不把|不将|而非|不是|并非|没有|避免|仅|只|至少|至多|最多|最少|假设|可能|保证|必然",
            span,
        )
    ]


class TranslationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    target_language: Literal["en"] = "en"

    def snapshot(self):
        values = {"version": "faithful-retrieval-translation-v1", **self.model_dump()}
        values["hash"] = digest(json.dumps(values, sort_keys=True))
        return values


class TranslationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    english_query: str = Field(min_length=1, max_length=6000)
    constraint_translations: list[str] = Field(default_factory=list, max_length=30)

    def validate_for(self, original, *, allow_token_plural=False, allow_name_spacing=False, term_corrections=(), allow_technical_inflections=False):
        if not self.english_query.strip() or has_chinese(self.english_query):
            raise ValueError("English retrieval view required")
        # Literal names/numbers cannot disappear. Semantic omissions still require review.
        token_source = original
        for correction in term_corrections:
            before, after = correction["original"], correction["source_alias"]
            if re.findall(r"\d+", before) != re.findall(r"\d+", after):
                raise ValueError("term correction changed a number")
            canonical_pattern = r"[\s_-]*".join(re.escape(c) for c in re.sub(r"[\s_-]+", "", after))
            if re.search(r"(?<![A-Za-z0-9_])" + canonical_pattern + r"(?![A-Za-z0-9_])", self.english_query, re.I):
                # The whole explicitly licensed term is accounted for. Other
                # names, values and every original protected clause remain checked.
                token_source = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(before) + r"(?![A-Za-z0-9_])", " ", token_source, flags=re.I)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*|\d+(?:\.\d+)?", token_source)
        replacements = {c["original"].casefold(): c["source_alias"] for c in term_corrections}
        for token in tokens:
            pattern = (r"tokens?" if allow_token_plural and token.casefold() == "token" else
                       r"\s*".join(re.escape(c) for c in token)
                       if allow_name_spacing and re.search(r"[A-Za-z]", token) else re.escape(token))
            # These attention-symbol bundles are ordered lists, unlike opaque
            # model names. License punctuation between symbols only, keeping
            # all members/order and the usual outer token boundaries. This
            # product opt-in does not relax frozen research translations.
            if allow_name_spacing and token.upper() in {"QKV", "QK", "KV"}:
                pattern = r"[\s,/、]*".join(re.escape(c) for c in token)
            noun = token.casefold()
            singular = ("query" if noun == "queries" else
                        noun[:-1] if noun.endswith("s") else noun)
            if allow_technical_inflections and singular in TECHNICAL_NOUNS:
                pattern = r"quer(?:y|ies)" if singular == "query" else re.escape(singular) + "s?"
            replacement = replacements.get(token.casefold())
            if replacement is not None:
                if re.findall(r"\d+", token) != re.findall(r"\d+", replacement):
                    raise ValueError("term correction changed a number")
                pattern = "(?:" + pattern + "|" + re.escape(replacement) + ")"
            if not re.search(r"(?<![A-Za-z0-9_])" + pattern + r"(?![A-Za-z0-9_])", self.english_query, re.I):
                raise ValueError("translation omitted literal name or number")
        spans = protected_spans(original)
        if len(spans) != len(self.constraint_translations):
            raise ValueError("translation omitted protected clause")
        for source, translated in zip(spans, self.constraint_translations, strict=True):
            if not translated.strip() or translated not in self.english_query:
                raise ValueError("constraint must be a literal translated-query span")
            # A-not-A questions ask about polarity; they do not assert a
            # negative. Product opt-in leaves frozen research behavior intact.
            polarity_source = re.sub(r"是不是|能不能|会不会|有没有|应不应该|要不要|可不可以", "", source) if allow_name_spacing else source
            if re.search(r"不要|不得|不能|不应|不把|不将|而非|不是|并非|没有|避免", polarity_source) and not re.search(
                r"\b(?:not|no|never|cannot|without|avoid|rather than|instead of)\b|n't", translated, re.I
            ):
                raise ValueError("negative clause lost explicit polarity")
        return self


class QueryTranslation(TranslationDraft):
    original_sha256: str = Field(pattern=r"[a-f0-9]{64}")
    policy_sha256: str = Field(pattern=r"[a-f0-9]{64}")
    source: Literal["LIVE", "MOCK", "IDENTITY"]

    def validate_for(self, original, policy):
        if (
            self.original_sha256 != digest(original)
            or self.policy_sha256 != policy.snapshot()["hash"]
        ):
            raise ValueError("translation source/policy binding mismatch")
        if self.source == "IDENTITY":
            if (
                has_chinese(original)
                or self.english_query != original
                or self.constraint_translations
            ):
                raise ValueError("invalid identity translation")
        else:
            if not policy.enabled:
                raise ValueError("translation disabled")
            TranslationDraft.validate_for(
                self, original, allow_token_plural=getattr(policy, "allow_token_plural", False)
            )
        return self

    def summarize(self):
        return {
            "original_sha256": self.original_sha256,
            "english_sha256": digest(self.english_query),
            "translation_sha256": digest(self.model_dump_json()),
            "policy_sha256": self.policy_sha256,
            "source": self.source,
            "protected_clause_count": len(self.constraint_translations),
            "semantic_validity": "model_proposal_pending_review"
            if self.source != "IDENTITY"
            else "identity",
        }


class TokenInflectionPolicy(TranslationPolicy):
    """Opt-in ordinary noun inflection only; model names/numbers stay literal."""

    allow_token_plural: Literal[True] = True

    def snapshot(self):
        values = {
            "version": "faithful-retrieval-translation-v2-token-inflection",
            **self.model_dump(),
            "allowed_inflections": {"token": ["token", "tokens"]},
        }
        return {**values, "hash": digest(json.dumps(values, sort_keys=True))}
