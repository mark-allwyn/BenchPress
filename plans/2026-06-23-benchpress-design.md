# Benchpress - v3 Eval System Design

## Context

The legacy harness (`legacy/`) evaluates LLMs with two benchmarks:
- **General** (80 prompts): scored by 1-5 LLM judge + DeepEval (also LLM-based). Saturated.
- **Causal** (100 MC items, 20 bundles x 5 variants): deterministically scored (extract A/B/C/D). Top models now cluster 75-78% - ceiling effect, no longer discriminating.

The thinking-adaptive experiment (commit 80416e0) showed enabling adaptive thinking did **not** change causal scores - the benchmark rewards pattern-matching to a correct letter, not reasoning. Models are right without thinking, and thinking doesn't rescue wrong answers.

Goal: a **new, very hard** evaluation system where frontier models cap around **50-60%**, scored **deterministically (no LLM judge)**, that **tracks models over time with useful statistics**, **compares heterogeneous models fairly**, and is **extensible** (add new metrics / benchmark tests).

## Decisions locked

1. **Fairness model = native config + Pareto reporting (+ optional budget-normalized view).**
   - Run each model in its own best native mode (thinking on if it has it; no one-size config that handicaps). A model without reasoning simply hits its own ceiling.
   - Never collapse to a single rank: report accuracy **against cost / latency / thinking-tokens** as a Pareto frontier. Tier models (reasoning / non-reasoning, size class, open / closed) so like compares with like.
   - Budget-normalized (same token/compute budget for all) kept as an optional secondary view, not the primary methodology.

2. **Name = "Benchpress".** A specialty benchmark for formal, verifiable-answer reasoning (not a generalist eval).

3. **Hardness engine = A + C.**
   - **A. Conjunctive multi-part structured items.** Each item demands several machine-checkable parts, all of which must be right (e.g. minimal sufficient adjustment set as an exact set, numeric estimate within tolerance, an ordering, a DAG edge list). Hardness falls out of conjunction (~0.9^5 ≈ 59%). No guessing floor, reasoning can't be pattern-matched, fully deterministic.
   - **C. Procedural / parametric generation.** Items generated from templates with held-out numeric params and randomized names/domains. Contamination-proof, infinite supply, difficulty is a tunable knob (re-harden by turning the knob, not hand-authoring traps).
   - Implication accepted: steers toward formal domains (causal / math / logic / algorithms / probability), not fuzzy ones that need a judge.

4. **Causal-first, modular architecture.** Build the module/registry abstraction from day one (a Benchpress "module" = generator + item schema + per-part scorers), but ship **causal inference as the only initial module**. Seed generators from the existing 20-bundle taxonomy (confounding, colliders, M-bias, Simpson's, mediation, selection, etc.). New formal modules (probability, algorithms, logic) slot in later without rework.

5. **Answer protocol = tagged text + per-part scoring.**
   - Output format is a tagged-text protocol (one labelled line per part, e.g. `ADJUSTMENT_SET: {SkillLevel, PriorWage}`), parsed by tolerant per-field extractors. No hard dependence on provider JSON/tool-calling modes (that would handicap models lacking them - unfair). The exact format + a worked example is included in every prompt, so format is disclosed, not a gotcha.
   - Score every part independently. Report two numbers: **headline = conjunctive item accuracy** (all parts correct), **diagnostic = per-part marginal accuracy** (which sub-skills fail). Single-shot, no reparse retries, no LLM fallback. Unparseable part = wrong part, not a voided item.

6. **Reproducibility & durability = versioned frozen set + fresh audit.**
   - Canonical leaderboard runs against a **fixed seeded set per version** ("Benchpress-Causal v1"): every model sees identical items, reproducible, directly comparable. Re-hardening = bump version (new seed / harder knob); old version scores persist as a historical line.
   - **Contamination audit**: periodically sample a fresh held-out set from the same distribution; a large public-vs-fresh score gap flags contamination/overfitting.
   - **Transparency posture**: generator code is public (methodology open); the **seed and full instantiated item set are private/held-out**; publish only a few example items + aggregate results. Open how, closed which.

7. **Statistics layer.**
   - **Bootstrapped 95% CIs on every accuracy** (point scores without uncertainty are how 78-vs-76 looked meaningful when it wasn't). CI overlap defines whether one model genuinely beats another.
   - **Pareto frontier** (accuracy vs thinking-tokens / latency / cost) - computed here, the fairness reporting.
   - **Per-skill / per-bundle marginals** - which causal concepts each model fails.
   - **Over-time tracking** - results tagged with benchmark version + model launch date; frontier-over-time trend; **saturation detector**: when top-N CIs overlap, auto-signal to re-harden (bump version).
   - **Item analysis: classical first** (per-item p-value + item-total discrimination to flag dead/broken items), **IRT deferred** until the tested-model population is large enough; slots in later as a stats module without changing data collection.
   - **Sampling: once per item** for the canonical leaderboard (bootstrap over items gives CIs); optional k-times repeated-sampling mode for a run-to-run variance estimate on a subset.

8. **Architecture = registry-based; reuse providers, rebuild the rest.**
   - Extensibility via three registries (generalizing the legacy `CHECKERS` dict): **MODULES** (benchmark generators), **PART_SCORERS** (keyed by part_type), **METRICS** (functions over stored results). "Add a benchmark / metric" = drop in a function, register it.
   - **Reuse (port):** `legacy/scripts/providers.py` (6 providers, native reasoning handling) + per-model JSON / multi-run-history persistence + root `config.yaml` model list.
   - **Rebuild:** scorers, generators, runner, stats. **Drop:** `judge.py`, `deepeval_scorer.py`, `general.json`. **Defer:** dashboard (ship lean stats CLI + JSON export first).

---

## Adopted from the retired v3 plan (the 12 June-11 slices, now closed)

A prior v3 plan existed as GitHub issues #1-12, now retired in favour of this document. Three pieces of it fix documented failure modes in our own causal work and are folded in; the rest (calibration pilot, consensus audit, runner hardening, cross-domain math/code/IF generators) was deliberately left out of the causal-first build.

1. **Dual-verification admission gate (answer keys must be independently confirmed).** No generated causal item enters the pool unless two independent methods agree on its answer: every graphical-solver answer (adjustment set / d-separation / transportability) is cross-checked against a seeded linear-Gaussian simulation + partial correlation, and any disagreement rejects the item; fallacy arithmetic (Simpson's reversal tables, base rates) is checked independently. This directly fixes the v2 miskeying problem (9/20 causal transfer keys were contradicted by cross-model consensus; B20_Q3 had to be hand-verified).

2. **Status taxonomy from `stop_reason`.** Every response is classified `ok | refusal | invalid_answer | truncated | api_error` from `stop_reason` + `stop_details` + extraction outcome. Refusal and invalid count as wrong but are reported as **separate columns**, never silently folded into accuracy. (The legacy harness did not record `stop_reason`, which is why Fable 5's 21 causal refusals looked like wrong answers.) Capture `stop_reason`/`stop_details` per run alongside token telemetry.

3. **Refusal-proofing at generation time.** Items use refusal-neutral cover stories only (logistics, manufacturing, agriculture, education, marketing); a **banned-vocabulary lint rejects health/bio/security terms at generation**. Any nonzero refusal rate in a sweep is treated as a content defect to investigate, not a model property. (Addresses Fable's bio-filter refusals on causal health bundles.)

---

## Implementation

### Directory layout (new `benchpress/` package at repo root; `legacy/` stays as read-only reference)

```
benchpress/
  cli.py                  # generate | run | stats | export | audit
  core/
    types.py              # Item, Part, PartResult, ItemResult, ModuleMeta dataclasses
    registry.py           # MODULES / PART_SCORERS / METRICS dicts + decorators
    tagged_text.py        # parse_tagged_fields(text) -> dict[label, raw]  (tolerant)
  providers/              # PORTED from legacy/scripts/providers.py (+ thinking_tokens, native_config, cost)
    base.py anthropic.py openai.py google.py ollama.py cohere.py bedrock.py errors.py __init__.py
  scorers/                # REBUILD: set_match, numeric_tolerance, sequence_match, edge_list, categorical
  modules/causal/         # REBUILD: generator(seed,difficulty); bundles.py dags.py naming.py render.py
  runner/                 # REBUILD: run.py (parallel), score.py (tagged->parts->conjunctive), persist.py
  stats/                  # REBUILD (decoupled, reads results only): load, bootstrap, metrics, report
  benchsets/benchpress-causal-v1.json   # seed+version manifest (materialized items held-out/gitignored)
  tests/
results/causal/<model>.json             # gitignored
```

### Core interfaces

- `Part{part_id, part_type, expected, params, skill_tags}` ; `Item{item_id, module, bundle_id, variant, difficulty, gen_params, prompt, parts}`.
- Module generator: `generator(seed:int, difficulty:str) -> (list[Item], ModuleMeta)` - pure/deterministic (same seed -> identical items; this is the reproducibility contract).
- Part-scorer: `(gold:Part, raw_field:str|None) -> PartResult{correct, parsed, expected, note}`. Unparseable/missing field -> `correct=False, parsed=None` (wrong part, not a voided item).
- Metric: `(ResultsView) -> dict`.
- Provider: `complete(prompt, params) -> (content, usage)`; `usage` gains `thinking_tokens`, `native_config_used`; runner computes `cost_usd` from a price table.

### Results schema (per model: `results/causal/<model>.json`)
Per-model file with `runs[item_id]` = list (multi-run history). Each run stores: `native_config_used`, `content`, `latency_s`, `input/output/thinking_tokens`, `cost_usd`, `benchmark_version`, `item_correct` (= all parts correct), and a `parts[]` array of `{part_id, part_type, parsed, expected, correct, note}` for marginal diagnostics. Gold values are NOT stored here - they regenerate from `(seed, difficulty)`, keeping the public generator separate from private golds.

### Provider port - thinking-token capture (the one real change)
Port the 6 classes verbatim; add a `thinking_tokens` extraction line + `native_config` to each `complete()`. Key paths:
- **Anthropic** (load-bearing): send `thinking:{"type":"adaptive"}` for 4.6+ (Opus 4.6/4.7/4.8, Sonnet 4.6); **omit** `thinking` for Fable 5 (adaptive always on; explicit disable 400s); no `budget_tokens`; raise `max_tokens` ~16000. Read `usage.output_tokens_details.thinking_tokens` (path proven in `legacy/scripts/experiment_thinking.py:55`).
- **OpenAI** o-series/gpt-5.x: `usage.completion_tokens_details.reasoning_tokens`. **Google** 2.5/3.x: `usageMetadata.thoughtsTokenCount`. **Ollama/xAI**: reasoning fallback already handled; capture reasoning token count where exposed. **Cohere/Bedrock**: `None` unless present.

### Build sequence
- **Phase 0 - spine:** `core/types.py`, `core/registry.py`, `core/tagged_text.py` + extractor tests; **status taxonomy** (ok/refusal/invalid_answer/truncated/api_error) from `stop_reason`.
- **Phase 1 - thin vertical slice:** all 5 part-scorers (unit-tested); causal module for **B01 only** end-to-end **with the dual-verification admission gate** (graphical solver vs linear-Gaussian simulation) + reproducibility test; port `anthropic.py`+`base.py`+`errors.py` (capturing `stop_reason`/`stop_details`/`thinking_tokens`); runner (serial) + persist; `cli generate` + `cli run` + `cli score`. **Exit:** `cli run --model claude-haiku-4-5 --benchmark causal` writes a results file with per-part rows, status, and conjunctive `item_correct`.
- **Phase 2 - stats slice:** `stats/` with `headline_accuracy_ci` (bootstrap) + `per_part_marginals`; refusal/invalid as separate columns; `cli stats`/`export`; tests on synthetic results.
- **Phase 3 - breadth:** remaining 5 providers; all 20 bundles x 5 variants (incl. `edge_list`/`sequence_match` for DAG `transfer`), all behind the dual-verification gate + **banned-vocabulary refusal lint**; parallelize runner (ThreadPoolExecutor + checkpoint flush, per `experiment_thinking.py`).
- **Phase 4 - full stats + canonical set:** `per_skill_marginals`, `pareto_frontier`, `saturation`, classical `item_stats`; freeze `benchpress-causal-v1` manifest (seed + version); `cli audit` fresh held-out sampling; optional `--runs k`.
- **Deferred:** dashboard.

### Verification
1. Reproducibility test: `generator(seed=42)` twice -> byte-identical items.
2. **Dual-verification gate:** solver vs simulation agree on N random instances; disagreement rejects the item. Banned-vocab lint green over the full generated set.
3. Scorer unit tests: exact / tolerant-format / unparseable cases for all 5 scorers.
4. **Status taxonomy:** a refusal `stop_reason` classifies as `refusal` (not `invalid`); re-running `score` never re-calls the API.
5. Live thin run on one cheap model: confirm parsing, `item_correct == all(parts.correct)`, and `stop_reason`/`thinking_tokens`/`latency`/`cost`/`native_config`/`version` present.
6. Stats: `cli stats` prints conjunctive accuracy **with bootstrapped 95% CI** + per-part marginals + separate refusal/invalid columns; invariant: each marginal >= conjunctive accuracy; refusal/invalid figures reconcile exactly with raw `stop_reason` counts.
7. Idempotency: re-run appends to `runs[item_id]` (history grows, no overwrite).
