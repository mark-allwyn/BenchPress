"""Shared metric helpers - the single source of truth for reported numbers.

This slice provides the accuracy helper used by the console board; the full
stats layer (bootstrap CIs, marginals, Pareto) builds on it in a later slice.
Refusals and invalids count as wrong but are surfaced as separate columns.
"""

from __future__ import annotations

import random

from benchpress.core.types import ItemResult


def accuracy(results: list[ItemResult]) -> dict:
    attempted = len(results)
    correct = sum(1 for r in results if r.item_correct)
    return {
        "attempted": attempted,
        "correct": correct,
        "accuracy": correct / attempted if attempted else 0.0,
        "refusals": sum(1 for r in results if r.status == "refusal"),
        "invalids": sum(1 for r in results if r.status == "invalid_answer"),
        "truncated": sum(1 for r in results if r.status == "truncated"),
        "errors": sum(1 for r in results if r.status == "api_error"),
    }


def bootstrap_ci(results: list[ItemResult], confidence: float = 0.95,
                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on the conjunctive item accuracy. Deterministic."""
    n = len(results)
    if n == 0:
        return (0.0, 0.0)
    flags = [1 if r.item_correct else 0 for r in results]
    rng = random.Random(seed)
    means = sorted(
        sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    alpha = (1 - confidence) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return (lo, hi)


def _marginal(counts: dict) -> dict:
    return {
        key: {"attempted": a, "correct": c, "accuracy": c / a if a else 0.0}
        for key, (a, c) in counts.items()
    }


def part_marginals(results: list[ItemResult]) -> dict:
    """Per-part accuracy (a part can be correct even when the item is wrong)."""
    counts: dict[str, list[int]] = {}
    for r in results:
        for p in r.parts:
            c = counts.setdefault(p.part_id, [0, 0])
            c[0] += 1
            c[1] += 1 if p.correct else 0
    return _marginal(counts)


def bundle_marginals(results: list[ItemResult], items) -> dict:
    """Per-bundle conjunctive accuracy, joining results to item metadata."""
    bundle_of = {i.item_id: i.bundle_id for i in items}
    counts: dict[str, list[int]] = {}
    for r in results:
        bundle = bundle_of.get(r.item_id)
        if bundle is None:
            continue
        c = counts.setdefault(bundle, [0, 0])
        c[0] += 1
        c[1] += 1 if r.item_correct else 0
    return _marginal(counts)


def report(results: list[ItemResult], items=None) -> dict:
    """Assemble the full stats payload from the shared helpers (single source)."""
    payload = dict(accuracy(results))
    lo, hi = bootstrap_ci(results)
    payload["ci95"] = [lo, hi]
    payload["part_marginals"] = part_marginals(results)
    if items is not None:
        payload["bundle_marginals"] = bundle_marginals(results, items)
    return payload
