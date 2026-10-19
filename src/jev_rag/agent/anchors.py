"""Lossless source locators. Boundaries/selection are heuristic, never semantic gold."""

import hashlib
import re

from .contracts import SourceAnchor

ANCHOR_VERSION = "lossless-source-units-v1"
CERTAINTY = re.compile(
    r"\b(?:suspect\w*|hypothes\w*|assum\w*|conjectur\w*|might|may|could|suggest\w*|speculat\w*|guarante\w*|necessar\w*|causal\w*|optimal|prove\w*|unverified)\b"
    r"|推测|猜测|假设|可能|不确定|保证|必然|必要|最优|证明|未经验证|未被验证",
    re.IGNORECASE,
)
ATTRIBUTION = re.compile(
    r"\b(?:section|appendix|figure|fig\.|table|footnote|abstract|introduction|page)\b"
    r"|章节|第.{0,12}[节章页]|图\s*\d|表\s*\d|脚注|摘要|引言",
    re.IGNORECASE,
)


def source_units(chunk):
    """Partition every original byte-equivalent character, grouping dependent 'Then'."""
    text = chunk.text
    cuts = [
        0,
        *[m.end() for m in re.finditer(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\u4e00-\u9fff])", text)],
        len(text),
    ]
    spans = []
    for start, end in zip(cuts, cuts[1:]):
        if start == end:
            continue
        # Keep the illustrative assumption together with its conditional consequence.
        if spans and re.match(
            r"(?:Then\b|Under these\b|In this case\b|因此|在此条件)", text[start:end].lstrip(), re.I
        ):
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    digest = hashlib.sha256(text.encode()).hexdigest()
    units = [
        SourceAnchor(
            unit_id=f"{chunk.chunk_id}::u{i}",
            chunk_id=chunk.chunk_id,
            start=start,
            end=end,
            text=text[start:end],
            source_sha256=digest,
        )
        for i, (start, end) in enumerate(spans)
    ]
    if not units or "".join(u.text for u in units) != text:
        raise ValueError("nonempty lossless source partition required")
    return units


def anchor_catalog(evidence):
    catalog = {}
    chunks = {}
    for chunk in evidence:
        if chunk.chunk_id in chunks:
            if chunk.model_dump() != chunks[chunk.chunk_id]:
                raise ValueError("conflicting source chunk")
            continue
        chunks[chunk.chunk_id] = chunk.model_dump()
        for unit in source_units(chunk):
            catalog[unit.unit_id] = unit
    return catalog


def validate_anchors(request):
    catalog = anchor_catalog(request.evidence)
    from .source_views import ABSTRACT_SUFFIX, PASSAGE_SUFFIX, source_view_catalog

    if any(a.unit_id.endswith((ABSTRACT_SUFFIX, PASSAGE_SUFFIX)) for a in request.anchors):
        catalog.update(source_view_catalog(request.evidence))
    ids = [a.unit_id for a in request.anchors]
    if len(set(ids)) != len(ids):
        raise ValueError("unique source anchors required")
    if any(a.unit_id not in catalog or a != catalog[a.unit_id] for a in request.anchors):
        raise ValueError("source anchor is not an exact supplied unit")
    if request.claim is None or set(request.claim.citations) != {
        a.chunk_id for a in request.anchors
    }:
        raise ValueError("citations must exactly match selected anchor sources")
    return request


def required_dimensions(claim, anchors):
    # Full support always checks all facts AND qualifiers. Optional checks only
    # target extra risks; their absence is not a semantic safety certificate.
    dimensions = ["support"]
    if CERTAINTY.search(claim.text) or any(CERTAINTY.search(a.text) for a in anchors):
        dimensions.append("certainty")
    if ATTRIBUTION.search(claim.text):
        dimensions.append("attribution")
    return dimensions
