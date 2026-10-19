"""Finite exact passage/abstract locators; no semantic support or generated ranges."""

import hashlib
import re

from .contracts import SourceAnchor

VIEW_VERSION = "literal-passage-views-v1"
PASSAGE_SUFFIX = "::view:passage-v1"
ABSTRACT_SUFFIX = "::view:abstract-v1"


def actual_abstract_range(text):
    heading = re.search(r"^\s*(?:abstract|摘要)\s*$", text, re.I | re.M)
    if heading is None:
        return None
    tail = text[heading.end() :]
    ending = re.search(
        r"^\s*(?:\d+\s+)?(?:I\s*NTRODUCTION|BACKGROUND|引言|绪论)\s*$", tail, re.I | re.M
    )
    stop = heading.end() + ending.start() if ending else len(text)
    return heading.start(), stop


def source_view_catalog(evidence):
    catalog, seen = {}, {}
    for chunk in evidence:
        data = chunk.model_dump()
        if chunk.chunk_id in seen:
            if seen[chunk.chunk_id] != data:
                raise ValueError("conflicting source chunk")
            continue
        seen[chunk.chunk_id] = data
        spans = [(PASSAGE_SUFFIX, 0, len(chunk.text))]
        abstract = actual_abstract_range(chunk.text)
        if abstract is not None:
            spans.append((ABSTRACT_SUFFIX, *abstract))
        for suffix, start, end in spans:
            uid = chunk.chunk_id + suffix
            catalog[uid] = SourceAnchor(
                unit_id=uid,
                chunk_id=chunk.chunk_id,
                start=start,
                end=end,
                text=chunk.text[start:end],
                source_sha256=hashlib.sha256(chunk.text.encode()).hexdigest(),
            )
    return catalog


def clause_view_scopes(points, evidence):
    catalog = source_view_catalog(evidence)
    full = sorted(uid for uid in catalog if uid.endswith(PASSAGE_SUFFIX))
    abstract = sorted(uid for uid in catalog if uid.endswith(ABSTRACT_SUFFIX))
    scopes, previous = {}, False
    for point in points:
        explicit = bool(re.search(r"摘要(?!任务)|\babstract\b", point, re.I))
        inherited = previous and bool(re.match(r"\s*(?:这些|上述)(?:发现|结论|结果)", point))
        restricted = explicit or inherited
        scopes[point] = abstract if restricted else full
        previous = restricted
    return scopes
