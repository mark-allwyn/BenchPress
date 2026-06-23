"""Console leaderboard. Every number comes from the shared stats helpers."""

from __future__ import annotations

from benchpress import stats
from benchpress.runner import persist


def leaderboard(paths) -> str:
    rows = []
    for path in paths:
        data = persist.load(path)
        results = persist.scored_results(path)
        acc = stats.accuracy(results)
        lo, hi = stats.bootstrap_ci(results)
        rows.append((data.get("model_name", "?"), acc, (lo, hi)))
    rows.sort(key=lambda r: -r[1]["accuracy"])

    lines = [f"{'model':<18} {'acc':>7}  {'95% CI':>16}  {'correct':>7}  {'refus':>5}  {'inval':>5}"]
    for name, acc, (lo, hi) in rows:
        ci = f"[{lo * 100:4.1f}, {hi * 100:4.1f}]"
        lines.append(
            f"{name:<18} {acc['accuracy'] * 100:6.1f}%  {ci:>16}  {acc['correct']:>7}  "
            f"{acc['refusals']:>5}  {acc['invalids']:>5}"
        )
    return "\n".join(lines)
