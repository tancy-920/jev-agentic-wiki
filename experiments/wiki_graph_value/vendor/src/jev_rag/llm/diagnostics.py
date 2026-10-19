"""Bounded failure categories; never serialize error messages, requests or causes."""

import hashlib
import socket
import ssl

import httpx
from pydantic import ValidationError

VERSION = "safe-failure-categories-v1"


def http_failure_details(response):
    """Fingerprint failure responses without retaining bodies or arbitrary headers."""
    content = response.content
    result = {
        "version": "safe-http-failure-v1",
        "body_bytes": len(content),
        "body_sha256": hashlib.sha256(content).hexdigest(),
        "content_kind": "other",
        "fixed_markers": [],
    }
    media = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    result["content_kind"] = {
        "application/json": "json",
        "text/html": "html",
        "text/plain": "text",
    }.get(media, "other")
    # These are observations of literal error phrases, not provider root causes.
    body = content[:8192].lower()
    result["fixed_markers"] = [
        phrase
        for phrase in (
            "service unavailable",
            "upstream connect error",
            "no healthy upstream",
            "overloaded",
            "rate limit",
            "invalid api key",
        )
        if phrase.encode("ascii") in body
    ]
    retry = response.headers.get("retry-after", "")
    if retry.isascii() and retry.isdigit() and len(retry) <= 5 and int(retry) <= 86400:
        result["retry_after_seconds"] = int(retry)
    return result


class AnchorConstraintError(ValueError):
    def __init__(self, code):
        if code not in {
            "duplicate_anchor",
            "too_many_anchors",
            "unknown_anchor",
            "unknown_requirement",
            "missing_requirement",
            "unit_budget",
            "blank_unit",
            "invalid_source_range",
            "source_range_budget",
            "source_range_gap",
        }:
            raise ValueError("unknown diagnostic category")
        self.code = code
        super().__init__(code)


def failure_details(error):
    chain, current, seen = [], error, set()
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    stage = "response_validation"
    if isinstance(error, httpx.HTTPStatusError):
        stage = "http_status"
    elif isinstance(error, httpx.ConnectError):
        stage = "connect_unknown"
    elif isinstance(error, httpx.TimeoutException):
        stage = "transport_timeout"
    elif isinstance(error, httpx.HTTPError):
        stage = "transport_unknown"
    for item in chain:
        if isinstance(item, ssl.SSLCertVerificationError):
            stage = "tls_certificate"
            break
        if isinstance(item, ssl.SSLError):
            stage = "tls"
            break
        if isinstance(item, socket.gaierror):
            stage = "dns"
            break
        if isinstance(item, ConnectionRefusedError):
            stage = "tcp_refused"
            break
    result = {"version": VERSION, "stage": stage}
    numbers = [
        item.errno for item in chain if isinstance(item, OSError) and type(item.errno) is int
    ]
    if numbers:
        result["os_errno"] = numbers[-1]
    return result


def anchor_validation_details(error):
    result = {
        "version": VERSION,
        "stage": "local_anchor_validation",
        "code": "invalid_json_or_constraint",
    }
    if isinstance(error, AnchorConstraintError):
        result["code"] = error.code
    elif isinstance(error, ValidationError):
        result["code"] = "schema_validation"
        safe_fields = {"claims", "text", "addresses", "anchor_ids"}
        result["fields"] = [
            [
                item if type(item) is int or item in safe_fields else "unlisted_field"
                for item in entry["loc"][:6]
            ]
            for entry in error.errors(
                include_input=False, include_context=False, include_url=False
            )[:8]
        ]
    return result
