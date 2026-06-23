"""Tolerant parser for the tagged-text answer protocol.

Models answer with one labelled line per part, e.g.::

    ADJUSTMENT_SET: {X, Z}
    ESTIMATE: 0.42
    IDENTIFIABLE: yes

Parsing is deliberately forgiving (case-insensitive labels, markdown bold/list
markers, wrapping backticks, surrounding prose) because format-following is
disclosed in the prompt but should not be a brittle gotcha. Labels are
normalized to upper-case keys; if a label repeats, the last occurrence wins.
"""

from __future__ import annotations

import re

# Optional markdown bullet/quote prefix, then LABEL, then optional markdown
# emphasis, then the colon. The label is a single identifier token.
_FIELD = re.compile(
    r"^\s*(?:[-*>]\s+)?[*_`]*([A-Za-z_][A-Za-z0-9_]*)[*_`]*\s*:\s*(.*)$"
)


def _clean_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^[`*]+", "", value)
    value = re.sub(r"[`*]+$", "", value)
    return value.strip()


def parse_tagged_fields(text: str) -> dict[str, str]:
    """Return a map of upper-cased label -> raw value string."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = _FIELD.match(line)
        if match:
            label = match.group(1).upper()
            fields[label] = _clean_value(match.group(2))
    return fields
