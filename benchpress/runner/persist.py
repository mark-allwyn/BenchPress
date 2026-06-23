"""Per-model results file with multi-run history and atomic saves."""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchpress.core.types import ItemResult, PartResult


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"runs": {}}
    return json.loads(p.read_text())


def save(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    json.loads(tmp.read_text())  # validate before swap
    os.replace(tmp, p)


def _result_from_scored(item_id: str, scored: dict) -> ItemResult:
    parts = [PartResult(**pr) for pr in scored["parts"]]
    return ItemResult(item_id, scored["status"], scored["item_correct"], parts)


def scored_results(path: str | Path) -> list[ItemResult]:
    data = load(path)
    out: list[ItemResult] = []
    for item_id, runs in data.get("runs", {}).items():
        if runs and runs[-1].get("scored"):
            out.append(_result_from_scored(item_id, runs[-1]["scored"]))
    return out


def load_scored(items, path: str | Path) -> list[ItemResult]:
    data = load(path)
    out: list[ItemResult] = []
    for item in items:
        runs = data.get("runs", {}).get(item.item_id)
        if runs and runs[-1].get("scored"):
            out.append(_result_from_scored(item.item_id, runs[-1]["scored"]))
    return out
