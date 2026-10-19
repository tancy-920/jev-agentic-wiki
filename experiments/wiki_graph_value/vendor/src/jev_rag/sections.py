"""Conservative heading detection; inferred labels never claim human verification."""

import re

NUMBERED = re.compile(r"^(?:[1-9]|1[0-2])(?:\.\d{1,2}){0,3}\.?\s+[A-Z][A-Za-z][^\n]{0,75}$")
NAMED = {"abstract", "references", "acknowledgments", "acknowledgements", "appendix"}


def section_spans(text: str, previous: str = "unknown") -> tuple[list[tuple[str, str]], str]:
    """Split only on short standalone headings. Preserve every source character."""
    spans, offset, start, current = [], 0, 0, previous
    for line in text.splitlines(keepends=True):
        heading = line.strip()
        is_heading = heading.lower() in NAMED or (
            NUMBERED.fullmatch(heading) is not None
            and len(heading.split()) <= 12
            and not re.search(r"[=<>;]|\.$", heading)
        )
        if is_heading:
            if offset > start:
                spans.append((current, text[start:offset]))
            start, current = offset, heading
        offset += len(line)
    if start < len(text):
        spans.append((current, text[start:]))
    return spans, current
