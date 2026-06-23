"""Core data shapes shared across generation, running, scoring, and stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Part:
    """One machine-checkable component of an item's answer."""

    part_id: str  # also the tagged-text label, e.g. "ADJUSTMENT_SET"
    part_type: str  # registry key into PART_SCORERS
    expected: Any  # gold value (set / number / list / verdict ...)
    params: dict = field(default_factory=dict)  # scorer params, e.g. {"tol": 0.02}
    skill_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Item:
    """A generated benchmark item. Correct only if every part is correct."""

    item_id: str
    module: str
    bundle_id: str
    variant: str
    difficulty: str
    gen_params: dict
    prompt: str  # fully rendered, incl. format spec + worked example
    parts: list[Part]
    skill_tags: list[str] = field(default_factory=list)


@dataclass
class PartResult:
    """The outcome of scoring one part of a response."""

    part_id: str
    part_type: str
    correct: bool
    parsed: Any
    expected: Any
    note: str = "ok"


@dataclass
class ItemResult:
    """The scored outcome for one item: a status plus per-part results."""

    item_id: str
    status: str  # see core.status
    item_correct: bool  # conjunctive: all parts correct
    parts: list[PartResult]


@dataclass(frozen=True)
class ModuleMeta:
    """Metadata describing a benchmark module."""

    name: str
    version: str
    variants: list[str] = field(default_factory=list)
    bundles: list[str] = field(default_factory=list)
    part_types: list[str] = field(default_factory=list)
