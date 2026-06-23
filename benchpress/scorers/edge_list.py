"""Directed-edge-set scorer (e.g. a DAG's edges)."""

from __future__ import annotations

import re

from benchpress.core.registry import register_part_scorer
from benchpress.core.types import Part, PartResult

# Match A->B, A→B, A-->B, or (A, B).
_EDGE = re.compile(r"\(?\s*([A-Za-z0-9_]+)\s*(?:-+>|→|,)\s*([A-Za-z0-9_]+)\s*\)?")


def _parse(raw: str) -> set[tuple[str, str]]:
    return {(m.group(1), m.group(2)) for m in _EDGE.finditer(raw)}


def _norm(edges) -> set[tuple[str, str]]:
    return {(str(a).casefold(), str(b).casefold()) for a, b in edges}


@register_part_scorer("edge_list")
def edge_list(gold: Part, raw: str | None) -> PartResult:
    expected = {tuple(e) for e in gold.expected}
    exp_sorted = sorted([list(e) for e in expected])
    if raw is None:
        return PartResult(gold.part_id, "edge_list", False, None, exp_sorted, "unparseable")
    parsed = _parse(raw)
    correct = _norm(parsed) == _norm(expected)
    note = "ok" if correct else "edge_mismatch"
    return PartResult(gold.part_id, "edge_list", correct, sorted([list(e) for e in parsed]), exp_sorted, note)
