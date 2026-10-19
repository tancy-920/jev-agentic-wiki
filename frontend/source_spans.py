"""Bounded original-span navigation; selecting a span never proves a claim."""
import re
from copy import deepcopy

SPAN_CHARS = 240
MAX_SPANS = 2048
MAX_PROPOSED_SPANS = 32
MAX_CITED_SOURCES = 8


def bounded_source_continuations(documents, corpus, *, limit=20):
    """Read up to2 adjacent originals, never jump a gap or certify their relevance."""
    if type(limit) is not int or not 1 <= limit <= 20 or len(documents) > limit:
        raise ValueError("bounded continuation evidence limit required")
    before = [c["chunk_id"] for c in documents]
    if len(set(before)) != len(before):
        raise ValueError("duplicate continuation input")
    detail = dict(policy="top-pdf-next2-contiguous-originals-within20-v1",
                  before_ids=before, added_ids=[], dropped_ids=[], original_steps=[],
                  additional_provider_calls=0, additional_search_queries=0,
                  semantic_support_certified=False)
    anchor = next((corpus.get(c["chunk_id"]) for c in documents
                   if c.get("source_kind") == "pdf"), None)
    if anchor is None:
        return list(documents), detail
    original = next(c for c in documents if c["chunk_id"] == anchor.chunk_id)
    if anchor.model_dump() != original:
        raise ValueError("continuation anchor differs from immutable corpus")
    if (anchor.source_kind != "pdf" or not anchor.source_sha256
        or type(anchor.start_char) is not int or type(anchor.end_char) is not int):
        return list(documents), detail
    following, cursor = [], anchor
    for _ in range(2):
        candidates = [c for c in corpus.values()
            if c.identity == anchor.identity and c.source_sha256 == anchor.source_sha256
            and c.source_version == anchor.source_version and c.source_kind == "pdf"
            and type(c.start_char) is int and type(c.end_char) is int
            and cursor.start_char < c.start_char <= cursor.end_char < c.end_char]
        if not candidates:
            break
        following_chunk = min(candidates, key=lambda c: (c.start_char, c.end_char, c.chunk_id))
        detail["original_steps"].append(dict(from_id=cursor.chunk_id, to_id=following_chunk.chunk_id,
            start_char=following_chunk.start_char, end_char=following_chunk.end_char,
            source_sha256=following_chunk.source_sha256))
        following.append(following_chunk.model_dump())
        cursor = following_chunk
    new = [c for c in following if c["chunk_id"] not in before]
    additions = new[:max(0, limit-1)]
    keep = [c for c in documents if c["chunk_id"] not in {d["chunk_id"] for d in additions}]
    position = next(i for i, c in enumerate(keep) if c["chunk_id"] == anchor.chunk_id) + 1
    # Keep the chosen anchor even when a preceding prefix consumes the budget.
    available = max(0, limit-position)
    additions = additions[:available]
    result = [*keep[:position], *additions, *keep[position:]][:limit]
    after = [c["chunk_id"] for c in result]
    detail.update(added_ids=[c["chunk_id"] for c in additions],
                  dropped_ids=[i for i in before if i not in after], selected_ids=after)
    return result, detail


def bounded_source_precedents(documents, corpus, *, limit=20):
    """Read up to2 predecessors of the second distinct PDF source, within20."""
    if type(limit) is not int or not 1 <= limit <= 20 or len(documents) > limit:
        raise ValueError("bounded precedent evidence limit required")
    before = [c["chunk_id"] for c in documents]
    if len(set(before)) != len(before):
        raise ValueError("duplicate precedent input")
    detail = dict(policy="second-pdf-previous2-contiguous-originals-within20-v1",
        before_ids=before, added_ids=[], dropped_ids=[], original_steps=[],
        additional_provider_calls=0, additional_search_queries=0, semantic_support_certified=False)
    anchors, identities = [], set()
    for document in documents:
        if document.get("source_kind") != "pdf":
            continue
        chunk = corpus.get(document["chunk_id"])
        if chunk is None or chunk.model_dump() != document:
            raise ValueError("precedent anchor differs from immutable corpus")
        if chunk.identity not in identities:
            anchors.append(chunk)
            identities.add(chunk.identity)
        if len(anchors) == 2:
            break
    if len(anchors) < 2:
        return list(documents), detail
    anchor = anchors[1]
    if (not anchor.source_sha256 or type(anchor.start_char) is not int
        or type(anchor.end_char) is not int):
        return list(documents), detail
    previous, cursor = [], anchor
    for _ in range(2):
        candidates = [c for c in corpus.values()
            if c.identity == anchor.identity and c.source_sha256 == anchor.source_sha256
            and c.source_version == anchor.source_version and c.source_kind == "pdf"
            and type(c.start_char) is int and type(c.end_char) is int
            and c.start_char < cursor.start_char <= c.end_char < cursor.end_char]
        if not candidates:
            break
        chosen = min(candidates, key=lambda c: (-c.start_char, -c.end_char, c.chunk_id))
        detail["original_steps"].append(dict(from_id=cursor.chunk_id, to_id=chosen.chunk_id,
            start_char=chosen.start_char, end_char=chosen.end_char, source_sha256=chosen.source_sha256))
        previous.append(chosen.model_dump())
        cursor = chosen
    additions = [c for c in reversed(previous) if c["chunk_id"] not in before]
    position = before.index(anchor.chunk_id)
    available = max(0, limit-position-1)
    additions = additions[-available:] if available else []
    result = [*documents[:position], *additions, *documents[position:]][:limit]
    after = [c["chunk_id"] for c in result]
    detail.update(anchor_id=anchor.chunk_id, added_ids=[c["chunk_id"] for c in additions],
        dropped_ids=[i for i in before if i not in after], selected_ids=after)
    return result, detail


def mechanism_paper_view(query, evidence, *, project_scope=False):
    """Prefer papers without discarding application docs or implementation facts.

    Source kind is a presentation preference, never a relevance certificate.
    Explicit APIs/projects retain the incoming order as well as all originals.
    """
    implementation = re.search(
        r"\b(?:api|pytorch|torch|tensorflow|jax|hugging\s*face|hf|code)\b|代码|源码|实现|[A-Za-z]_[A-Za-z]",
        query, re.IGNORECASE)
    if project_scope or implementation:
        return list(evidence)
    papers = [c for c in evidence if c.source_kind == "pdf"]
    return papers + [c for c in evidence if c.source_kind != "pdf"]


def source_span_view(evidence):
    """Partition complete originals without dropping or normalizing any character."""
    by_source, spans = {}, {}
    for source in evidence:
        if source.chunk_id in by_source:
            raise ValueError("duplicate source for span view")
        fragments = []
        for start in range(0, len(source.text), SPAN_CHARS):
            if len(spans) >= MAX_SPANS:
                raise ValueError("source span view exceeds bounded inventory")
            end = min(start + SPAN_CHARS, len(source.text))
            identifier = f"E{len(spans)+1:04d}"
            item = dict(chunk_id=source.chunk_id, start_char=start, end_char=end,
                        text=source.text[start:end])
            spans[identifier] = item
            fragments.append(dict(excerpt_id=identifier, text=item["text"]))
        by_source[source.chunk_id] = fragments
    return by_source, spans


def bind_source_span_selection(data, spans):
    """Derive citation IDs from exact bounded selections, never guessed metadata."""
    if not isinstance(data, dict):
        raise ValueError("span draft must be an object")
    units = [data["answer"]] if "answer" in data else data.get("claims")
    if not isinstance(units, list) or not 1 <= len(units) <= 4:
        raise ValueError("bounded span units required")
    cleaned, observations = [], []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or "citations" in unit:
            raise ValueError("span units select excerpts rather than separate citations")
        selected = unit.get("source_excerpt_ids")
        if (not isinstance(selected, list) or not 1 <= len(selected) <= MAX_PROPOSED_SPANS
            or not all(isinstance(i, str) and i in spans for i in selected)
            or len(set(selected)) != len(selected)):
            raise ValueError("bounded unique offered excerpt IDs required")
        citations = list(dict.fromkeys(spans[i]["chunk_id"] for i in selected))[:MAX_CITED_SOURCES]
        authorized = [i for i in selected if spans[i]["chunk_id"] in citations]
        items = [dict(excerpt_id=i, **spans[i]) for i in authorized]
        cleaned.append({**{k: v for k, v in unit.items() if k != "source_excerpt_ids"},
                        "citations": citations})
        observations.append(dict(unit=index, selected_original_spans=items,
                                 proposed_excerpt_ids=list(selected), authorized_excerpt_ids=authorized,
                                 omitted_excerpt_ids=[i for i in selected if i not in authorized],
                                 selection_policy="first8-distinct-sources-within32-span-proposals-v3",
                                 provider_quoted_text=False, semantic_support_certified=False,
                                 additional_provider_calls=0))
    result = deepcopy(data)
    result.update({"answer": cleaned[0]} if "answer" in data else {"claims": cleaned})
    return result, observations



def bind_generation_relation_audit(data):
    """Retain short public relation labels separately; never evidence or a verdict."""
    clean = deepcopy(data)
    if not isinstance(clean, dict):
        raise ValueError("invalid generation relation audit")
    units = [clean["answer"]] if "answer" in clean else clean.get("claims")
    if not isinstance(units, list) or not units:
        raise ValueError("invalid generation relation audit")
    records = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise ValueError("invalid generation relation audit")
        fields = unit.pop("relation_audit", None)
        if (not isinstance(fields, dict) or set(fields) != {"inputs", "operation", "dependency", "scope"}
            or any(not isinstance(v, str) or not v.strip() or len(v) > 60 for v in fields.values())):
            raise ValueError("invalid generation relation audit")
        records.append(dict(unit_index=index, fields=fields, semantic_support_certified=False,
            used_as_verification_evidence=False, additional_provider_calls=0))
    return clean, records
def bind_named_generation_units(data, *, cap=2):
    """Map bounded contiguous closed slots without editing their content."""
    from copy import deepcopy

    if type(cap) is not int or not 2 <= cap <= 4:
        raise ValueError("invalid named generation units cap")
    keys = [f"U{i:02d}" for i in range(1, cap + 1)]
    if (not isinstance(data, dict) or "units" not in data
        or set(data) - {"units", "response_kind"}
        or not isinstance(data["units"], dict) or set(data["units"]) != set(keys)
        or not isinstance(data["units"]["U01"], dict)
        or any(data["units"][key] is not None and not isinstance(data["units"][key], dict)
               for key in keys[1:])):
        raise ValueError("invalid named generation units")
    empty = False
    for key in keys:
        if data['units'][key] is None:
            empty = True
        elif empty:
            raise ValueError("invalid named generation units gap")
    cleaned = deepcopy(data)
    slots = cleaned.pop("units")
    populated = [key for key in keys if slots[key] is not None]
    cleaned["claims"] = [slots[key] for key in populated]
    return cleaned, dict(slot_keys=populated,
        number_of_units=len(cleaned["claims"]), text_rewritten=False, semantic_support_certified=False,
        additional_provider_calls=0)

