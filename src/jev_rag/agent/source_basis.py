"""Exact sentence locators within source views; boundaries are heuristic, not gold."""

from .anchors import source_units
from .contracts import SourceAnchor
from .source_views import source_view_catalog


def basis_catalog(evidence):
    chunks = {c.chunk_id: c for c in evidence}
    catalog = {}
    for uid, view in source_view_catalog(evidence).items():
        local = chunks[view.chunk_id].model_copy(update={"text": view.text})
        for index, unit in enumerate(source_units(local)):
            identifier = f"{uid}::basis:{index}"
            catalog[identifier] = SourceAnchor(
                unit_id=identifier,
                chunk_id=view.chunk_id,
                start=view.start + unit.start,
                end=view.start + unit.end,
                text=unit.text,
                source_sha256=view.source_sha256,
            )
    return catalog


def validate_basis(request):
    evidence_ids = [c.chunk_id for c in request.evidence]
    if len(evidence_ids) != len(set(evidence_ids)) or set(evidence_ids) != set(
        request.claim.citations
    ):
        raise ValueError("fact evidence must exactly match unique cited sources")
    catalog = basis_catalog(request.evidence)
    ids = [a.unit_id for a in request.basis_anchors]
    if len(set(ids)) != len(ids) or any(
        a.unit_id not in catalog or a != catalog[a.unit_id] for a in request.basis_anchors
    ):
        raise ValueError("basis must match exact canonical source units")
    if set(request.claim.citations) != {a.chunk_id for a in request.basis_anchors}:
        raise ValueError("fact citations must exactly match selected basis sources")
    return request
