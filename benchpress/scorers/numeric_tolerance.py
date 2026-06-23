"""Numeric scorer with absolute (or relative) tolerance."""

from __future__ import annotations

import re

from benchpress.core.registry import register_part_scorer
from benchpress.core.types import Part, PartResult

# A fraction a/b, or a (possibly signed, scientific) decimal.
_FRACTION = re.compile(r"[-+]?\d+\s*/\s*\d+")
_DECIMAL = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse(raw: str):
    text = raw.strip().replace(",", "")
    frac = _FRACTION.search(text)
    if frac:
        num, den = re.split(r"/", frac.group())
        try:
            return float(num) / float(den)
        except ZeroDivisionError:
            return None
    m = _DECIMAL.search(text)
    if m:
        return float(m.group())
    return None


@register_part_scorer("numeric_tolerance")
def numeric_tolerance(gold: Part, raw: str | None) -> PartResult:
    expected = float(gold.expected)
    tol = float(gold.params.get("tol", 0.0))
    relative = bool(gold.params.get("rel", False))
    if raw is None:
        return PartResult(gold.part_id, "numeric_tolerance", False, None, expected, "unparseable")
    parsed = _parse(raw)
    if parsed is None:
        return PartResult(gold.part_id, "numeric_tolerance", False, None, expected, "unparseable")
    bound = tol * abs(expected) if relative else tol
    correct = abs(parsed - expected) <= bound
    note = "ok" if correct else f"diff={abs(parsed - expected):.4g}>tol={bound:.4g}"
    return PartResult(gold.part_id, "numeric_tolerance", correct, parsed, expected, note)
