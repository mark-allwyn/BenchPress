"""Load the model registry from the root config.yaml (same shape as v2)."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_models(path: str | Path = "config.yaml") -> dict[str, dict]:
    """Return the ``models:`` section as {model_name: spec}, or {} if absent."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return data.get("models") or {}
