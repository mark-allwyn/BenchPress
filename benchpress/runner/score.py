"""Offline scoring: tagged-text -> per-part scoring -> conjunctive item result.

Never calls a provider. Operates only on stored response text + stop reason.
"""

from __future__ import annotations

from dataclasses import asdict

from benchpress.core.registry import get_part_scorer
from benchpress.core.status import classify_status
from benchpress.core.tagged_text import parse_tagged_fields
from benchpress.core.types import Item, ItemResult
from benchpress.runner import persist


def score_response(item: Item, content: str, stop_reason: str | None, error: str | None = None) -> ItemResult:
    fields = parse_tagged_fields(content or "")
    extraction_ok = any(p.part_id in fields for p in item.parts)
    status = classify_status(stop_reason, extraction_ok=extraction_ok, error=error)
    parts = [get_part_scorer(p.part_type)(p, fields.get(p.part_id)) for p in item.parts]
    item_correct = status == "ok" and all(pr.correct for pr in parts)
    return ItemResult(item.item_id, status, item_correct, parts)


def score_model(items, path) -> None:
    data = persist.load(path)
    for item in items:
        runs = data.get("runs", {}).get(item.item_id)
        if not runs:
            continue
        last = runs[-1]
        result = score_response(item, last.get("content"), last.get("stop_reason"), last.get("error"))
        last["scored"] = {
            "status": result.status,
            "item_correct": result.item_correct,
            "parts": [asdict(pr) for pr in result.parts],
        }
    persist.save(path, data)
