"""Locate and load a benchmark's frozen manifest.

A frozen tier ships an immutable ``frozen_v<N>.json`` inside its module directory
(``benchpress/modules/<benchmark>/``). It carries the ``official_run_config`` that
every model must run under so scores stay comparable across models and over time.
The highest version present is the current official manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

_MODULES_ROOT = Path(__file__).resolve().parent / "modules"


def manifest_path(benchmark: str) -> Path | None:
    """Newest ``frozen_v*.json`` for a benchmark's module, or None if unfrozen."""
    d = _MODULES_ROOT / benchmark
    candidates = sorted(d.glob("frozen_v*.json"))
    return candidates[-1] if candidates else None


def load_frozen(benchmark: str) -> dict | None:
    """The parsed manifest for a benchmark, or None if it has no frozen tier."""
    p = manifest_path(benchmark)
    return json.loads(p.read_text()) if p else None


def run_params_from_config(cfg: dict) -> dict:
    """Extract the provider generation params from an ``official_run_config``.

    Only keys with a concrete value are returned, so they cleanly overlay a
    model's config.yaml params without clobbering with None.
    """
    mapping = {
        "max_tokens": cfg.get("max_tokens"),
        "thinking": cfg.get("thinking"),
        "effort": cfg.get("effort"),
        "read_timeout": cfg.get("read_timeout_seconds"),
    }
    return {k: v for k, v in mapping.items() if v is not None}
