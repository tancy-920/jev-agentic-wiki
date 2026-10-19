"""Explicit wire format plus strict, non-repairing JSON parsing and safe diagnostics."""

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

PARSER_VERSION = "unique-finite-json-v1"


class StructuredOutputPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    responses_format: Literal["json_schema", "json_object", "strict_function"] = "json_schema"

    def snapshot(self):
        values = {
            "version": "explicit-structured-wire-v1",
            **self.model_dump(),
            "local_parser": PARSER_VERSION,
            "repair_invalid_output": False,
            "automatic_format_fallback": False,
        }
        values["hash"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values


def parse_json(output):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_member")
            result[key] = value
        return result

    def non_finite(_):
        raise ValueError("non_finite_json_number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non_finite_json_number")
        return number

    return json.loads(
        output, object_pairs_hook=unique_object, parse_constant=non_finite, parse_float=finite_float
    )


def json_observation(output):
    """No content/keys/exception text; fixed categories and syntax positions only."""
    result = {"parser_version": PARSER_VERSION, "output_character_count": len(output)}
    try:
        value = parse_json(output)
        result.update(valid_json=True, root_type=type(value).__name__)
    except json.JSONDecodeError as error:
        character = output[error.pos : error.pos + 1]
        allowed = {
            "Expecting value",
            "Extra data",
            "Expecting property name enclosed in double quotes",
            "Expecting ',' delimiter",
            "Expecting ':' delimiter",
            "Unterminated string starting at",
            "Invalid \\escape",
            "Invalid control character at",
        }
        result.update(
            valid_json=False,
            failure="syntax",
            message=error.msg if error.msg in allowed else "invalid_json",
            position=error.pos,
            line=error.lineno,
            column=error.colno,
            at_end_of_output=error.pos >= len(output),
            error_character_class=character
            if character and character in '{}[]:,.+-"'
            else "other_or_end",
        )
    except ValueError as error:
        allowed = {"duplicate_json_member", "non_finite_json_number"}
        result.update(
            valid_json=False, failure=str(error) if str(error) in allowed else "invalid_json"
        )
    return result
