# Design: Reasoning-efficiency and truncation-resilience metrics

Date: 2026-07-23
Status: approved (design), pending implementation

## Motivation

Benchpress runs every model under one frozen budget (Simulate v2: 96k tokens).
While onboarding the Gemini panel we discovered the Gemini API hard-caps output at
**65,536 tokens** and silently clamps our 96k request down to it.
On the hardest tasks (ECA110/ECA30) `gemini-3.5-flash` spends ~63k tokens *thinking*
and truncates before printing the grid.

Rather than treat this as a mere footnote, we make how a model spends its budget a
first-class, measured property of the benchmark.
This adds insight (reasoning economy, single-shot reliability) without disturbing the
existing exact-match / per-row rankings.

## Metrics (additive - existing metrics unchanged)

### 1. Reasoning efficiency

Correct output rows produced per 1,000 generated tokens, **pooled** across all items:

```
efficiency = (Σ correct_rows) / (Σ generated_tokens) × 1000
```

- Numerator uses per-row correctness (the graded metric); exact-match is too sparse
  to form a stable ratio.
- Pooled (total ÷ total), NOT the mean of per-item ratios - far less noisy at
  n=25/task and immune to one cheap correct item spiking the average.
- Truncated items contribute their tokens but ~no correct rows, so truncation
  correctly depresses efficiency.
- Interpretation: higher = more economical reasoning ("N correct rows per 1k tokens").
  It measures reasoning *density*, a complement to accuracy - NOT a capability score
  on its own (a capable model can still be verbose).

### 2. Truncation resilience

Fraction of items that produced a complete answer within the budget:

```
resilience_pct = (n - truncated) / n × 100
```

- Board-wide and computable today from `stop_reason` alone.
- A genuine single-shot deployment property: "in one response, under a fixed budget,
  does the model actually deliver a usable answer?"

## Token accounting (the crux)

`generated_tokens` = total tokens the model emitted to answer (thinking + visible).
Providers report this differently; the canonical rule:

| Provider          | output_tokens semantics       | rule                              |
|-------------------|-------------------------------|-----------------------------------|
| google (Gemini)   | visible answer ONLY           | `output_tokens + thinking_tokens` |
| bedrock           | total incl. reasoning         | `output_tokens`                   |
| anthropic direct  | total incl. thinking          | `output_tokens`                   |
| openai            | total incl. reasoning         | `output_tokens`                   |
| ollama/others     | total                         | `output_tokens`                   |

Only Google reports answer and thoughts separately, so it is the only provider that
adds `thinking_tokens`. Implemented as a single provider-keyed helper
`generated_tokens(provider, output_tokens, thinking_tokens)`.

An item with no `output_tokens` recorded is excluded from that model's efficiency
denominator (and flagged), never counted as zero-cost.

## Data availability

- `output_tokens`: 100% of all runs -> efficiency is board-wide.
- `thinking_tokens`: only needed for Gemini, and present for all Gemini runs.
- No re-runs required; existing result files already carry the needed fields.

## Where it computes

`build_leaderboard` (benchpress/leaderboard.py) is the single site with both the raw
token data (`persist.load`) and provider metadata (config). It computes per-model
`efficiency` and `resilience_pct` and adds them to each entry's `overall` block.
`per_task_summary` gains a `resilience_pct` per task/overall for the console table.

The frozen manifest (`frozen_v2.json`) is NOT touched: these are reporting-layer
metrics, not run-config, so v2 comparability is preserved.

## Interface (docs/index.html) — as built

- **Efficiency column** in the table (sortable), gated: the number shows only for
  models above 10% per-row (others `—`), because a rows-per-token ratio flatters
  cheap-and-wrong models.
- **Resilience is NOT a column.** It was near-constant (~100% for most models, since
  the 96k budget was tuned to minimise truncation), so it added no signal. Instead the
  total truncation count surfaces as a `⚑N` flag in the Overall cell, shown only when
  >0 — concentrating the signal on the verbose outliers (Sonnet 4.6 ⚑25). resilience_pct
  stays in the JSON for analysis.
- **Cost-vs-payoff chart** (replaced an earlier accuracy-vs-tokens scatter, which read
  as an analyst's chart and implied "more tokens = better"). Diverging bars, one row per
  ranked model, sorted by accuracy: median generated tokens (cost, left) vs per-row
  accuracy (payoff, right). Short-left/long-right = efficient; long-left/short-right =
  wasteful. Cheap-and-wrong models are self-evidently weak here (tiny right bar), so no
  gating is needed in this view.
- **`64k` cap chip** on each affected Gemini row + a prominent budget caveat note.
- Explanations: methodology copy defining efficiency + the token rule, and the Gemini
  64k-cap caveat.

## README + docs

- Add a "Reasoning efficiency and resilience" subsection to the metrics explanation.
- Add the 64k-cap caveat to the results/caveats prose.

## Testing

- `generated_tokens` rule: one case per provider (esp. google adds, others don't).
- Efficiency pooled math incl. truncated-item handling and missing-token exclusion.
- Resilience math.
- Leaderboard payload includes the new fields; dashboard load stays backward-compatible
  when fields are absent (older exports).

## Out of scope

- The "accuracy at a fixed token cutoff" efficiency frontier (needs token-resolved
  traces we don't capture).
- Re-instrumenting Bedrock to separate reasoning tokens (AWS converse does not report
  them separately; not recoverable without a provider/runtime change).
