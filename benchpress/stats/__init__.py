"""Shared metric helpers - the single source of truth for reported numbers.

This slice provides the accuracy helper used by the console board; the full
stats layer (bootstrap CIs, marginals, Pareto) builds on it in a later slice.
Refusals and invalids count as wrong but are surfaced as separate columns.
"""

from __future__ import annotations

import random
from statistics import median

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


def pareto_frontier(points: list[dict], *, accuracy_key: str = "accuracy",
                    minimize: str = "cost") -> list[dict]:
    """Non-dominated points: higher accuracy is better, lower `minimize` is better."""
    frontier = []
    for p in points:
        dominated = any(
            q is not p
            and q[accuracy_key] >= p[accuracy_key]
            and q[minimize] <= p[minimize]
            and (q[accuracy_key] > p[accuracy_key] or q[minimize] < p[minimize])
            for q in points
        )
        if not dominated:
            frontier.append(p)
    return frontier


def saturation(entries: list[tuple], top_n: int = 3) -> bool:
    """True if the top-N models' confidence intervals mutually overlap (no clear
    leader) - the cue to re-harden. Entries are (model, accuracy, (lo, hi))."""
    top = sorted(entries, key=lambda e: -e[1])[:top_n]
    if len(top) < 2:
        return False
    los = [lo for _, _, (lo, _) in top]
    his = [hi for _, _, (_, hi) in top]
    return max(los) <= min(his)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx**0.5 * syy**0.5)


def item_stats(model_results: dict[str, list[ItemResult]]) -> dict:
    """Classical item analysis: per-item p-value + item-total discrimination.

    p_value = fraction of models that got the item right (flags dead/broken
    items); discrimination = correlation between item correctness and the
    model's overall accuracy across models.
    """
    totals = {
        m: (sum(1 for r in rs if r.item_correct) / len(rs) if rs else 0.0)
        for m, rs in model_results.items()
    }
    per_item: dict[str, dict[str, int]] = {}
    for m, rs in model_results.items():
        for r in rs:
            per_item.setdefault(r.item_id, {})[m] = 1 if r.item_correct else 0

    out = {}
    for item_id, per in per_item.items():
        models = list(per)
        flags = [per[m] for m in models]
        skill = [totals[m] for m in models]
        out[item_id] = {
            "p_value": sum(flags) / len(flags) if flags else 0.0,
            "discrimination": _pearson([float(f) for f in flags], skill),
            "n_models": len(models),
        }
    return out


def model_efficiency(data: dict) -> dict:
    """Per-model accuracy + median latency/thinking-tokens from a results file."""
    lat, think, correct, attempted = [], [], 0, 0
    for runs in data.get("runs", {}).values():
        if not runs:
            continue
        last = runs[-1]
        scored = last.get("scored")
        if not scored:
            continue
        attempted += 1
        correct += 1 if scored["item_correct"] else 0
        if last.get("latency_s") is not None:
            lat.append(last["latency_s"])
        if last.get("thinking_tokens") is not None:
            think.append(last["thinking_tokens"])
    return {
        "model": data.get("model_name"),
        "attempted": attempted,
        "accuracy": correct / attempted if attempted else 0.0,
        "median_latency": median(lat) if lat else None,
        "median_thinking": median(think) if think else None,
    }


def report(results: list[ItemResult], items=None) -> dict:
    """Assemble the full stats payload from the shared helpers (single source)."""
    payload = dict(accuracy(results))
    lo, hi = bootstrap_ci(results)
    payload["ci95"] = [lo, hi]
    payload["part_marginals"] = part_marginals(results)
    if items is not None:
        payload["bundle_marginals"] = bundle_marginals(results, items)
    return payload
