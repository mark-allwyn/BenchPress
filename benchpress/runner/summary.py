"""Per-task summary - the single source for the numbers reported by the eval
console, the leaderboard export, and the official runner (so they never drift).

Two headline metrics per task:
  - exact_pct : conjunctive item accuracy (every part/row correct). Harsh headline.
  - per_row_pct : per-part marginal accuracy. Graded, non-saturating, primary.

Validity is read from the status taxonomy (core.status), so truncation/refusal/
error are surfaced as their own columns for every provider, never scored as wrong
answers silently.
"""

from __future__ import annotations

from benchpress.core.types import Item, ItemResult


def _blank() -> dict:
    return {"n": 0, "exact": 0, "part_correct": 0, "part_total": 0,
            "truncated": 0, "errors": 0, "refusals": 0, "invalid": 0}


def _pct(num: int, den: int) -> float:
    return round(num / den * 100, 1) if den else 0.0


def _finalize(s: dict) -> dict:
    return {
        "n": s["n"],
        "exact_pct": _pct(s["exact"], s["n"]),
        "per_row_pct": _pct(s["part_correct"], s["part_total"]),
        "truncated": s["truncated"],
        "errors": s["errors"],
        "refusals": s["refusals"],
        "invalid": s["invalid"],
    }


def per_task_summary(items: list[Item], results: list[ItemResult]) -> dict:
    """Join scored results to item metadata and roll up per bundle + overall.

    Task order follows first appearance in ``items`` so the report is stable.
    """
    by_id = {i.item_id: i for i in items}
    order: list[str] = []
    tasks: dict[str, dict] = {}
    for it in items:
        if it.bundle_id not in tasks:
            tasks[it.bundle_id] = _blank()
            order.append(it.bundle_id)

    total = _blank()
    for r in results:
        it = by_id.get(r.item_id)
        if it is None:
            continue
        for s in (tasks[it.bundle_id], total):
            s["n"] += 1
            s["exact"] += 1 if r.item_correct else 0
            s["truncated"] += 1 if r.status == "truncated" else 0
            s["errors"] += 1 if r.status == "api_error" else 0
            s["refusals"] += 1 if r.status == "refusal" else 0
            s["invalid"] += 1 if r.status == "invalid_answer" else 0
            for p in r.parts:
                s["part_total"] += 1
                s["part_correct"] += 1 if p.correct else 0

    return {
        "tasks": {b: _finalize(tasks[b]) for b in order},
        "overall": _finalize(total),
    }


def format_console(summary: dict, *, title: str = "") -> str:
    """Render a per-task summary as an aligned console table."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'task':<10} {'exact':>8}  {'per-row':>8}  {'n':>3}  {'trunc':>5}  {'err':>3}")
    for name, s in summary["tasks"].items():
        lines.append(f"{name:<10} {s['exact_pct']:7.1f}%  {s['per_row_pct']:7.1f}%  "
                     f"{s['n']:>3}  {s['truncated']:>5}  {s['errors']:>3}")
    o = summary["overall"]
    lines.append(f"{'OVERALL':<10} {o['exact_pct']:7.1f}%  {o['per_row_pct']:7.1f}%  "
                 f"{o['n']:>3}  {o['truncated']:>5}  {o['errors']:>3}")
    return "\n".join(lines)
