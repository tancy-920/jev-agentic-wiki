"""Bounded source-attested technical vocabulary; suggestions never change intent."""

import re

from pydantic import BaseModel, ConfigDict, Field


def term_key(value):
    return re.sub(r"[\s_-]+", "", value).casefold()


def one_edit(left, right):
    if any(c.isdigit() for c in left + right):
        return False  # Never guess versions, dimensions or numeric identifiers.
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        positions = [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
        return len(positions) == 1 or (len(positions) == 2 and positions[1] == positions[0] + 1
            and left[positions[0]] == right[positions[1]] and left[positions[1]] == right[positions[0]])
    short, long = sorted([left, right], key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


class TermChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original: str = Field(min_length=1, max_length=80)
    canonical: str | None = Field(default=None, max_length=80)


class TermCatalog:
    def __init__(self, pages, chunks):
        self.entries = {}
        # Attestation only licenses a spelling, not a factual answer or synonym.
        texts = [(getattr(c, "chunk_id", None), c.text) for c in chunks]
        for page in pages:
            for alias in page.aliases:
                if not (re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{1,79}", alias)
                        or re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{2,40}", alias)):
                    continue
                key = term_key(alias)
                pattern = re.compile(r"(?<![A-Za-z0-9_])" + r"[\s_-]*".join(re.escape(c) for c in key) + r"(?![A-Za-z0-9_])", re.I)
                attested = [identifier for identifier, text in texts if pattern.search(text)]
                if not attested:
                    continue
                entry = self.entries.setdefault(key, {"canonical": alias, "wiki_pages": [],
                    "attested_chunk_ids": [identifier for identifier in attested if identifier][:2]})
                if page.page_id not in entry["wiki_pages"]:
                    entry["wiki_pages"].append(page.page_id)

    def suggest(self, query):
        mentions = []
        # Longest known spaced/hyphenated phrases first; short acronyms only exact.
        spans = []
        for entry in self.entries.values():
            phrase = entry["canonical"]
            pattern = r"[\s_-]*".join(re.escape(c) for c in term_key(phrase))
            for match in re.finditer(r"(?<![A-Za-z0-9_])" + pattern + r"(?![A-Za-z0-9_])", query, re.I):
                spans.append((match.start(), match.end(), entry))
        used = []
        for start, end, entry in sorted(spans, key=lambda s: (-(s[1]-s[0]), s[0])):
            if end - start > 80 or any(start < b and end > a for a, b in used):
                continue
            used.append((start, end))
            mentions.append({"original": query[start:end], "candidates": [entry], "kind": "exact_or_format"})
        for match in re.finditer(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]{4,}(?![A-Za-z0-9_])", query):
            if any(match.start() < b and match.end() > a for a, b in used):
                continue
            key = term_key(match.group())
            options = [e for k, e in self.entries.items() if k.isascii() and len(k) >= 5 and one_edit(key, k)]
            if 1 <= len(options) <= 3:
                mentions.append({"original": match.group(), "candidates": sorted(options, key=lambda e: e['canonical']), "kind": "spelling_candidate"})
        # Surface text is the identifier; duplicate mentions share one decision.
        mentions.sort(key=lambda m: (m['kind'] != 'spelling_candidate', query.find(m['original'])))
        return list({m['original']: m for m in mentions}.values())[:8]

    @staticmethod
    def authorize(query, suggestions, choices):
        offered = {m["original"]: {e["canonical"] for e in m["candidates"]} for m in suggestions}
        duplicates = {c.original for c in choices if sum(v.original == c.original for v in choices) > 1}
        seen, corrections, rejected = set(), [], []
        for choice in choices:
            original, canonical = choice.original, choice.canonical
            if original in duplicates or original in seen or original not in offered or (canonical is not None and canonical not in offered[original]):
                rejected.append(original)
                continue
            seen.add(original)
            if canonical is not None and canonical != original:
                corrections.append({"original": original, "source_alias": canonical,
                    "role": "retrieval_suggestion_not_ground_truth"})
        expanded = query
        # Simultaneous literal substitutions; never recursively rewrite a replacement.
        mapping = {c["original"]: c["source_alias"] for c in corrections}
        if mapping:
            pattern = r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?![A-Za-z0-9_])"
            expanded = re.sub(pattern, lambda m: mapping[m.group()], query)
        return {"original_query": query, "expanded_query": expanded, "corrections": corrections,
                "rejected_choices": rejected, "suggestions": suggestions,
                "policy": "source-vocabulary-one-edit-v1", "new_provider_calls": 0}
