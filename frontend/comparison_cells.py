"""Optional literal comparison cells, rendered before every factual check.

The renderer supplies query labels and punctuation only, never technical facts.
"""

import math
from itertools import combinations

from frontend.comparison_grid import comparison_grid
from frontend.source_spans import MAX_CITED_SOURCES, MAX_PROPOSED_SPANS


def layout(query, *, unit_cap, char_cap):
    grid = comparison_grid(query)
    if grid is None:
        return None
    if type(unit_cap) is not int or not 1 <= unit_cap <= 4 or char_cap != 480:
        raise ValueError("bounded comparison-cell capacity required")
    size = math.ceil(len(grid["objects"]) / unit_cap)
    groups = [
        list(range(i, min(i + size, len(grid["objects"]))))
        for i in range(0, len(grid["objects"]), size)
    ]

    def render(indices, cells):
        return "\n".join(
            grid["objects"][i]["text"]
            + "："
            + "；".join(axis["text"] + "：" + cells[i][j] for j, axis in enumerate(grid["axes"]))
            + "。"
            for i in indices
        )

    empty = {i: {j: "" for j in range(len(grid["axes"]))} for i in range(len(grid["objects"]))}
    max_chars = min(
        80, min((char_cap - len(render(g, empty))) // (len(g) * len(grid["axes"])) for g in ([i] for i in range(len(grid["objects"]))))
    )
    if max_chars < 16:
        raise ValueError("comparison labels leave insufficient bounded cell capacity")
    return dict(grid=grid, groups=groups, cell_char_limit=max_chars, unit_char_limit=char_cap,
                unit_count_limit=unit_cap)


def schema(plan, span_ids, requirement_refs):
    axes = {f"a{j}": {"$ref": "#/$defs/cell"} for j in range(len(plan["grid"]["axes"]))}
    obj = dict(
        type="object",
        additionalProperties=False,
        properties={
            "source_excerpt_ids": {"$ref": "#/$defs/spans"},
            "cells": dict(
                type="object", additionalProperties=False, properties=axes, required=list(axes)
            ),
        },
        required=["source_excerpt_ids", "cells"],
    )
    objects = {f"o{i}": obj for i in range(len(plan["grid"]["objects"]))}
    props = dict(
        response_kind=dict(type="string", enum=["ANSWER", "INSUFFICIENT_EVIDENCE"]),
        addresses=dict(
            type="array",
            minItems=1,
            maxItems=len(requirement_refs),
            items=dict(type="string", enum=list(requirement_refs)),
        ),
        objects=dict(
            type="object", additionalProperties=False, properties=objects, required=list(objects)
        ),
    )
    return dict(
        type="object",
        additionalProperties=False,
        properties=props,
        required=list(props),
        **{
            "$defs": dict(
                cell=dict(type="string", minLength=1, maxLength=plan["cell_char_limit"]),
                spans=dict(
                    type="array",
                    minItems=1,
                    maxItems=MAX_PROPOSED_SPANS,
                    items=dict(type="string", enum=list(span_ids)),
                ),
            )
        },
    )


def bind(data, plan, span_ids, requirement_refs):
    if not isinstance(data, dict) or set(data) != {"response_kind", "addresses", "objects"}:
        raise ValueError("closed comparison cell root required")
    if not isinstance(data["response_kind"], str) or data["response_kind"] not in {
        "ANSWER",
        "INSUFFICIENT_EVIDENCE",
    }:
        raise ValueError("invalid comparison response kind")
    addresses = data["addresses"]
    if (
        not isinstance(addresses, list)
        or not addresses
        or any(not isinstance(v, str) or v not in requirement_refs for v in addresses)
        or len(set(addresses)) != len(addresses)
    ):
        raise ValueError("invalid selected requirement refs")
    objects = data["objects"]
    grid = plan["grid"]
    if not isinstance(objects, dict) or set(objects) != {
        f"o{i}" for i in range(len(grid["objects"]))
    }:
        raise ValueError("all literal comparison objects required")
    cells = {}
    refs = {}
    for i in range(len(grid["objects"])):
        obj = objects[f"o{i}"]
        if not isinstance(obj, dict) or set(obj) != {"source_excerpt_ids", "cells"}:
            raise ValueError("closed comparison object required")
        refs[i] = obj["source_excerpt_ids"]
        row = obj["cells"]
        if (
            not isinstance(refs[i], list)
            or not 1 <= len(refs[i]) <= MAX_PROPOSED_SPANS
            or any(not isinstance(v, str) or v not in span_ids for v in refs[i])
            or len(set(refs[i])) != len(refs[i])
        ):
            raise ValueError(f"o{i}: select 1..{MAX_PROPOSED_SPANS} unique offered excerpt IDs")
        if not isinstance(row, dict) or set(row) != {f"a{j}" for j in range(len(grid["axes"]))}:
            raise ValueError("all literal comparison axes required")
        for axis_id, value in row.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"o{i}.{axis_id}: nonempty comparison cell text required")
            if len(value) > plan["cell_char_limit"]:
                raise ValueError(f"o{i}.{axis_id}: comparison cell length {len(value)} exceeds {plan['cell_char_limit']}")
        cells[i] = row
    def render(group):
        return "\n".join(
            grid["objects"][i]["text"] + "：" + "；".join(
                axis["text"] + "：" + cells[i][f"a{j}"] for j, axis in enumerate(grid["axes"])
            ) + "。" for i in group
        )

    def fits(group):
        selected = set(v for i in group for v in refs[i])
        return (len(selected) <= MAX_PROPOSED_SPANS
                and len({span_ids[v]['chunk_id'] for v in selected}) <= MAX_CITED_SOURCES
                and len(render(group)) <= plan["unit_char_limit"])

    groups = plan['groups']
    # Keep the original grouping when valid. Otherwise select bounded contiguous
    # partitions; preserve order, text and EVERY excerpt without inventing facts.
    if not all(fits(g) for g in groups):
        found = None
        for count in range(len(groups), plan['unit_count_limit'] + 1):
            for cuts in combinations(range(1, len(cells)), count - 1):
                bounds = [0, *cuts, len(cells)]
                candidate = [list(range(a,b)) for a,b in zip(bounds,bounds[1:])]
                if all(len(g) <= max(map(len,plan['groups'])) and fits(g) for g in candidate):
                    found = candidate
                    break
            if found:
                break
        if found:
            groups = found
    units = []
    for group in groups:
        selected = list(dict.fromkeys(v for i in group for v in refs[i]))
        if len(selected) > MAX_PROPOSED_SPANS:
            raise ValueError("comparison group exceeds 32 complete excerpt selections")
        # Every object selection must survive the existing per-unit source cap.
        # Reject rather than let another object's citations displace premises.
        if len({span_ids[v]["chunk_id"] for v in selected}) > MAX_CITED_SOURCES:
            raise ValueError("comparison group exceeds complete source authorization capacity")
        text = render(group)
        if len(text) > plan["unit_char_limit"]:
            raise ValueError(f"comparison objects {group}: rendered length {len(text)} exceeds {plan['unit_char_limit']}; shorten cells, preserving conditions")
        units.append(
            dict(
                text=text,
                source_excerpt_ids=selected,
                addresses=list(addresses),
            )
        )
    return dict(response_kind=data["response_kind"], claims=units), dict(
        original_objects=data["objects"],
        groups=groups,
        cell_char_limit=plan["cell_char_limit"],
        rendered_text_lengths=[len(u["text"]) for u in units],
        factual_text_rewritten=False,
        only_literal_query_labels_and_separators_added=True,
        semantic_support_certified=False,
        additional_provider_calls=0,
    )


def instruction(plan):
    return (
        " Return one closed comparison-cell JSON instance with response_kind,addresses,objects. "
        "objects contains EVERY literal oN object; each has source_excerpt_ids and cells with EVERY aN axis. "
        f"Each cell has 1..{plan['cell_char_limit']} characters including spaces. "
        f"Each rendered unit including query labels must fit {plan['unit_char_limit']} characters; cells may use unequal lengths. "
        "Answer exactly that object on that axis as subject, retaining necessary conditions; no overview "
        "or unrelated measurements. No units/claims root. Select 1..32 supplied source excerpt IDs for each "
        "object, covering all its cells; each rendered group must have at most32 distinct excerpt IDs "
        "from at most8 distinct chunk IDs. Several excerpts can belong to one chunk. Code derives "
        "citations. addresses selects supplied requirement refs. "
        "The code adds original query labels and partitions adjacent objects within the supplied unit "
        "count limit, preserving every selected source, before ALL original source and "
        "whole checks; labels are not facts or coverage certification. Neither a formula nor an object "
        "name substitutes for a different requested limitation. Distinguish positional dependence from "
        "the complete function and preserve every operand. Attribute empirical conclusions to actual "
        "evaluation settings; omit optional measurements if provenance does not fit. The complete rendered "
        "answer must still be source-supported; no field is approved by merely being nonempty. "
    )
