"""Build the score-only public leaderboard payload consumed by the dashboard.

Contains numbers and model metadata ONLY - never prompts, gold answers, or raw
model text - so it is safe to commit to a public repo. Every model runs the same
frozen questions, so per-task exact-match and per-row accuracy are comparable.

The published Opus 4.8 baseline from the frozen manifest is always included as an
anchor row, so the dashboard has data before any fresh run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from benchpress.frozen import load_frozen
from benchpress.metrics import (
    efficiency_per_1k,
    efficiency_shown,
    generated_tokens,
    output_cap,
    resilience_pct,
)
from benchpress.runner import per_task_summary, persist


def _overall(task_map: dict) -> dict:
    """Macro-average task percentages (used for the manifest baseline row where
    only per-task numbers are published)."""
    if not task_map:
        return {"exact_pct": 0.0, "per_row_pct": 0.0}
    ex = sum(t["exact_pct"] for t in task_map.values()) / len(task_map)
    pr = sum(t["per_row_pct"] for t in task_map.values()) / len(task_map)
    return {"exact_pct": round(ex, 1), "per_row_pct": round(pr, 1)}


def _baseline_entry(frozen: dict) -> dict | None:
    base = frozen.get("baseline")
    if not base:
        return None
    n = base.get("n")
    tasks = {
        b: {"exact_pct": r["exact_pct"], "per_row_pct": r["per_row_pct"],
            "n": n, "truncated": r.get("truncated", 0), "errors": 0}
        for b, r in base["results"].items()
    }
    total_n = sum(t["n"] for t in tasks.values())
    total_trunc = sum(t["truncated"] for t in tasks.values())
    overall = _overall(tasks)
    # The manifest baseline publishes no token counts, so efficiency is unavailable;
    # resilience still follows from its truncation totals. Baseline ran on Bedrock at
    # the frozen budget (no Gemini cap).
    overall["efficiency"] = None
    overall["resilience_pct"] = resilience_pct(total_n, total_trunc)
    return {
        "name": "claude-opus-4.8",
        "company": "Anthropic",
        "type": "closed",
        "launch_date": "2026-05-28",
        "run_date": base.get("measured_date"),
        "baseline": True,
        "output_cap": frozen.get("official_run_config", {}).get("max_tokens"),
        "efficiency_shown": False,   # manifest baseline publishes no token counts
        "gen_tokens_median": None,
        "overall": overall,
        "tasks": tasks,
    }


def _thinking_off(data: dict) -> bool:
    """True if the provider had to run this model without extended thinking
    (recorded per-response as native_config.thinking_downgraded)."""
    for runs in data.get("runs", {}).values():
        if runs and (runs[-1].get("native_config") or {}).get("thinking_downgraded"):
            return True
    return False


def _median(xs: list[int]) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def _efficiency_metrics(data: dict, results, meta: dict, budget_default: int | None,
                        summary: dict) -> dict:
    """Reasoning-efficiency, resilience, token cost, and output cap for one model,
    from raw token counts + provider token semantics. ``gen_tokens_median`` is the
    frontier x-axis (typical tokens/item); ``efficiency`` is the pooled rows-per-1k."""
    provider = meta.get("provider")
    correct_rows = sum(1 for r in results for p in r.parts if p.correct)
    per_item_gen: list[int] = []
    for r in results:
        runs = data.get("runs", {}).get(r.item_id)
        if not runs:
            continue
        last = runs[-1]
        g = generated_tokens(provider, last.get("output_tokens"), last.get("thinking_tokens"))
        if g is not None:
            per_item_gen.append(g)
    total_gen = sum(per_item_gen) if per_item_gen else None
    o = summary["overall"]
    return {
        "efficiency": efficiency_per_1k(correct_rows, total_gen),
        "efficiency_shown": efficiency_shown(o["per_row_pct"]),
        "resilience_pct": resilience_pct(o["n"], o["truncated"]),
        "gen_tokens_median": _median(per_item_gen),
        "output_cap": output_cap(provider, budget_default),
    }


def _model_entry(name: str, summary: dict, meta: dict, run_date: str | None,
                 thinking_off: bool = False, eff: dict | None = None) -> dict:
    eff = eff or {}
    return {
        "name": name,
        "company": meta.get("company"),
        "type": meta.get("type"),
        "launch_date": meta.get("launch_date"),
        "run_date": run_date,
        "baseline": False,
        "thinking_off": thinking_off,
        "output_cap": eff.get("output_cap"),
        "efficiency_shown": eff.get("efficiency_shown", False),
        "gen_tokens_median": eff.get("gen_tokens_median"),
        "overall": {"exact_pct": summary["overall"]["exact_pct"],
                    "per_row_pct": summary["overall"]["per_row_pct"],
                    "efficiency": eff.get("efficiency"),
                    "resilience_pct": eff.get("resilience_pct")},
        "tasks": {b: {"exact_pct": s["exact_pct"], "per_row_pct": s["per_row_pct"],
                      "n": s["n"], "truncated": s["truncated"], "errors": s["errors"]}
                  for b, s in summary["tasks"].items()},
    }


def _last_run_date(data: dict) -> str | None:
    dates = [runs[-1].get("timestamp") for runs in data.get("runs", {}).values() if runs]
    dates = [d for d in dates if d]
    return max(dates)[:10] if dates else None


def build_leaderboard(benchmark: str, items, paths, models_meta: dict,
                      *, include_baseline: bool = True, now: str | None = None,
                      require_complete: bool = True) -> dict:
    """Assemble the leaderboard dict from scored result files + config metadata.

    With ``require_complete`` (default), only runs that scored every item are
    included - partial/in-progress runs are skipped so the board never shows a
    misleading half-finished score.
    """
    frozen = load_frozen(benchmark) or {}
    bundles = list(dict.fromkeys(i.bundle_id for i in items))

    entries: list[dict] = []
    seen = set()
    if include_baseline and frozen:
        base = _baseline_entry(frozen)
        if base:
            entries.append(base)
            seen.add(base["name"])

    for path in paths:
        data = persist.load(path)
        name = data.get("model_name", Path(path).stem)
        results = persist.load_scored(items, path)
        if not results or (require_complete and len(results) < len(items)):
            continue
        summary = per_task_summary(items, results)
        # Skip runs marred by API/infra errors (e.g. a credit-blocked partial):
        # those errored items are not real attempts and would understate the model.
        if require_complete and summary["overall"]["errors"] > 5:
            continue
        meta = models_meta.get(name, {})
        budget = frozen.get("official_run_config", {}).get("max_tokens")
        eff = _efficiency_metrics(data, results, meta, budget, summary)
        entry = _model_entry(name, summary, meta, _last_run_date(data),
                             thinking_off=_thinking_off(data), eff=eff)
        # A real run supersedes the static baseline row for the same model.
        entries = [e for e in entries if e["name"] != name]
        seen.discard(name)
        entries.append(entry)
        seen.add(name)

    entries.sort(key=lambda e: -e["overall"]["per_row_pct"])
    # The ranked board is thinking-on models only (like-for-like). Models with no
    # extended-thinking mode are demoted to a separate "floor" list - kept as a
    # control (they show the task needs reasoning) but never ranked.
    ranked = [e for e in entries if not e.get("thinking_off")]
    floor = [e for e in entries if e.get("thinking_off")]

    cfg = frozen.get("official_run_config", {})
    return {
        "benchmark": frozen.get("benchmark", benchmark),
        "version": frozen.get("version", ""),
        "generated": now or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "tasks": bundles,
        "config": {
            "seed": cfg.get("seed"),
            "n_per_bundle": cfg.get("n_per_bundle"),
            "max_tokens": cfg.get("max_tokens"),
            "tools": "off",
            "thinking": cfg.get("thinking"),
        },
        "task_meta": frozen.get("tasks", {}),
        "models": ranked,
        "floor": floor,
    }
