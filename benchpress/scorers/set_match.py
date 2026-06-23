"""Set-equality scorer (e.g. minimal sufficient adjustment sets)."""

from __future__ import annotations

import re

from benchpress.core.registry import register_part_scorer
from benchpress.core.types import Part, PartResult

_EMPTY_SYNONYMS = {"", "{}", "[]", "()", "none", "empty", "empty set", "emptyset", "∅"}


def _parse(raw: str) -> set[str]:
    stripped = raw.strip()
    if stripped.lower() in _EMPTY_SYNONYMS:
        return set()
    # Drop surrounding brackets/braces, then split on commas or whitespace.
    inner = re.sub(r"^[\[\{\(]|[\]\}\)]$", "", stripped).strip()
    if inner.lower() in _EMPTY_SYNONYMS:
        return set()
    tokens = re.split(r"[,\s]+", inner)
    return {t.strip() for t in tokens if t.strip()}


@register_part_scorer("set_match")
def set_match(gold: Part, raw: str | None) -> PartResult:
    expected = {str(x) for x in gold.expected}
    if raw is None:
        return PartResult("set", "set_match", False, None, sorted(expected), "unparseable")
    parsed = _parse(raw)
    correct = {x.casefold() for x in parsed} == {x.casefold() for x in expected}
    note = "ok" if correct else "set_mismatch"
    return PartResult(gold.part_id, "set_match", correct, sorted(parsed), sorted(expected), note)
