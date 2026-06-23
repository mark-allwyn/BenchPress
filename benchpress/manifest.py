"""Frozen benchmark manifest: makes a shipped item set reproducible from seed.

The manifest records the generator seed + version + kept item ids. Regenerating
from the seed reproduces the items exactly. The manifest is held-out (gitignored)
since the seed is what keeps the canonical set contamination-resistant.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_manifest(benchmark: str, seed: int, version: str, item_ids: list[str]) -> dict:
    return {
        "benchmark": benchmark,
        "version": version,
        "seed": seed,
        "n_items": len(item_ids),
        "item_ids": sorted(item_ids),
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
