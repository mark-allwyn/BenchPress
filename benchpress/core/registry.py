"""The three extension-point registries.

Generalizes the legacy ``CHECKERS`` dict into three: benchmark modules,
part-scorers (keyed by part_type), and metrics. Each is populated by import
side-effect via a registration decorator, so adding a domain, an answer shape,
or a statistic is a one-function change.
"""

from __future__ import annotations

from typing import Callable

MODULES: dict[str, Callable] = {}
PART_SCORERS: dict[str, Callable] = {}
METRICS: dict[str, Callable] = {}


def _register(store: dict[str, Callable], key: str):
    def decorator(fn: Callable) -> Callable:
        if key in store:
            raise ValueError(f"{key!r} is already registered")
        store[key] = fn
        return fn

    return decorator


def register_module(name: str):
    return _register(MODULES, name)


def register_part_scorer(part_type: str):
    return _register(PART_SCORERS, part_type)


def register_metric(name: str):
    return _register(METRICS, name)


def get_module(name: str) -> Callable:
    return MODULES[name]


def get_part_scorer(part_type: str) -> Callable:
    return PART_SCORERS[part_type]


def get_metric(name: str) -> Callable:
    return METRICS[name]
