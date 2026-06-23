"""Ordered-sequence scorer (order-sensitive)."""

from __future__ import annotations

import re

from benchpress.core.registry import register_part_scorer
from benchpress.core.types import Part, PartResult

_SEP = re.compile(r"\s*(?:-+>|→|>|,)\s*")


def _parse(raw: str) -> list[str]:
    inner = re.sub(r"^[\[\{\(]|[\]\}\)]$", "", raw.strip()).strip()
    return [t for t in _SEP.split(inner) if t]


@register_part_scorer("sequence_match")
def sequence_match(gold: Part, raw: str | None) -> PartResult:
    expected = [str(x) for x in gold.expected]
    if raw is None:
        return PartResult(gold.part_id, "sequence_match", False, None, expected, "unparseable")
    parsed = _parse(raw)
    correct = [t.casefold() for t in parsed] == [t.casefold() for t in expected]
    note = "ok" if correct else "sequence_mismatch"
    return PartResult(gold.part_id, "sequence_match", correct, parsed, expected, note)
