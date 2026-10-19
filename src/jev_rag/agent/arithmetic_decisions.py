"""Same bounded source-derived literal hints for both decision providers."""

from .anchored import AnchoredJev, AnchoredLLM, anchored_payload
from .arithmetic import PROTOCOL, source_operations


def arithmetic_payload(requests):
    payload = anchored_payload(requests)
    by_id = {c.chunk_id: c for r in requests if r.task == "citation" for c in r.evidence}
    if by_id:
        payload["state"]["literal_operation_observations"] = [
            dict(
                **o,
                source_literal=by_id[o["chunk_id"]].text[o["start"] : o["end"]],
                semantic_support_certified=False,
            )
            for o in source_operations(by_id.values())
        ]
        for key, request in payload["state"]["requests"].items():
            if request.get("claim") is not None and "answer_claims" not in request:
                payload["questions"][key]["instructions"]["question"] += (
                    " Source-derived literal_operation_observations are fallible grammar hints, not truth labels. "
                    "Use only observations for this request's actual evidence_ids. Compare division versus "
                    "multiplication and reciprocal operands with the full passage. Do not confuse dividing "
                    "by a factor with multiplying by it. Missing/ambiguous extraction does not certify support; "
                    "all claims still require full source support, qualifiers and scope."
                )
    return payload


class ArithmeticLLM(AnchoredLLM):
    payload_builder = staticmethod(arithmetic_payload)
    protocol_version = "anchored-arithmetic-v1"


class ArithmeticJev(AnchoredJev):
    payload_builder = staticmethod(arithmetic_payload)
    protocol_version = "anchored-arithmetic-v1"


ARITHMETIC_DECISION_PROTOCOL = dict(
    literal_guard=PROTOCOL,
    version="anchored-arithmetic-v1",
    payload="lossless source catalog plus derived literal observations; no reviewed labels",
)
