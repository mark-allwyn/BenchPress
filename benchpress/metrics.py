"""Reporting-layer metrics: token accounting + reasoning efficiency + resilience.

Computed AFTER a run from stored results; they never affect the frozen run-config,
so within-version comparability is preserved. See
docs/superpowers/specs/2026-07-23-efficiency-resilience-metrics-design.md
"""

from __future__ import annotations

# Providers whose ``output_tokens`` counts the VISIBLE answer only, so thinking
# tokens must be ADDED to recover total generated tokens. Every other provider
# already folds reasoning tokens into ``output_tokens``, so adding would double count.
_THINKING_SEPARATE = {"google"}

# Google's Gemini API hard-caps a single response (thinking + answer) at this many
# tokens. Verified via the models endpoint (outputTokenLimit) for every current
# gemini-* text model, 2026-07: it silently clamps a larger maxOutputTokens request
# down to this. So Gemini effectively runs below the frozen v2 budget.
GEMINI_OUTPUT_CAP = 65536


def output_cap(provider: str | None, budget_default: int | None) -> int | None:
    """The model's real single-response output ceiling: Gemini's hard cap, else the
    frozen run's budget (what the run intended to allow)."""
    if provider in _THINKING_SEPARATE:
        return GEMINI_OUTPUT_CAP
    return budget_default


def generated_tokens(provider: str | None, output_tokens: int | None,
                     thinking_tokens: int | None) -> int | None:
    """Total tokens the model emitted to answer (thinking + visible answer).

    Returns None when the base output count is missing, so the item is excluded
    from an efficiency denominator rather than treated as zero-cost.
    """
    if output_tokens is None:
        return None
    if provider in _THINKING_SEPARATE:
        return output_tokens + (thinking_tokens or 0)
    return output_tokens


# A model must clear this per-row accuracy for its efficiency NUMBER to be shown in
# the table. Efficiency (correct rows / tokens) rewards a tiny denominator, so a model
# that emits almost nothing and lands a couple of rows by luck would otherwise look
# "most efficient". 10% cleanly separates models doing the task (Gemini 3-flash and up,
# all Claudes, minimax) from near-random output (glm 0.8%, qwen-coder 3.5%). It only
# gates the column; the accuracy-vs-tokens frontier still plots every model.
EFFICIENCY_MIN_PER_ROW = 10.0


def efficiency_per_1k(correct_rows_total: int, generated_tokens_total: int | None) -> float | None:
    """Pooled correct rows per 1,000 generated tokens, or None if no token data.

    Pooled (total / total) rather than a mean of per-item ratios: less noisy at
    small n and immune to one cheap correct item dominating the average.
    """
    if not generated_tokens_total:
        return None
    return round(correct_rows_total / generated_tokens_total * 1000, 4)


def efficiency_shown(per_row_pct: float | None) -> bool:
    """Whether a model's efficiency number should appear in the table (vs the frontier,
    which shows all models). Guards against cheap-and-wrong models topping the metric."""
    return per_row_pct is not None and per_row_pct >= EFFICIENCY_MIN_PER_ROW


def resilience_pct(n: int, truncated: int) -> float | None:
    """Percent of items that produced a complete answer within the budget."""
    if not n:
        return None
    return round((n - truncated) / n * 100, 1)
