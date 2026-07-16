# Benchpress: public multi-provider pipeline, README, and dashboard

_Design spec. Date: 2026-07-16. Status: approved ("build it")._

## Goal

Turn the frozen Benchpress-Simulate v1 benchmark into a public, easy-to-run, multi-provider evaluation pipeline with a professional README and a static GitHub Pages leaderboard dashboard.
The benchmark itself is unchanged and stays frozen; this work is packaging, generalization, and presentation.

## Principles carried over from the benchmark

- Deterministic, judge-free, tools-off. Gold is computed at runtime by a trivially-correct simulator, never stored.
- The frozen manifest (`benchpress/modules/simulate/frozen_v1.json`) is the immutable source of truth. Re-hardening is a new version, never an edit.
- Only `stop_reason == end_turn` (or a provider-equivalent completed stop) counts; truncation and timeouts are recorded, never silently scored as wrong.

## Decisions (locked)

- Claude access: support both Bedrock and direct Anthropic API key, selected per config entry.
- Anti-contamination: fresh-seed holdout audit. Public canonical set is seed 42; a different seed mints a structurally identical private set for the existing canonical-vs-fresh `audit` comparison.
- Dashboard data: commit a score-only `leaderboard.json` (numbers and metadata only; no prompts, no gold, no raw model text).
- Scope: build a module-agnostic harness; Simulate v1 is the first tier.
- Run workflow: local run then export then commit then Pages auto-publishes. No CI-run evals.
- Look and feel: neutral scientific leaderboard aesthetic, light and dark.
- License: MIT.
- Legacy `causal` and `frontier` modules stay in the codebase (they back a large share of the ~389 tests) but are not part of the public surface. The known-buggy frontier front-door gold remains internal and documented.

## Components

### 1. Unified multi-provider eval pipeline

New single command: `python -m benchpress eval --model <name> --benchmark simulate`.
It generates the frozen item set, runs the model, scores it, and prints a per-task summary in one step.

Mechanism:

- `--model` resolves to a `config.yaml` entry and `get_provider(spec)` builds the adapter, so any provider works (anthropic, bedrock, openai, google, cohere, ollama).
- If a frozen manifest exists for the benchmark, its `official_run_config` (seed, n_per_bundle, max_tokens, thinking, effort, read_timeout) is overlaid onto the model spec's params before building the provider. This makes every model run the identical frozen configuration regardless of vendor.
- Providers degrade honestly: those without a thinking mode ignore it; a model that cannot emit the full token budget simply truncates, and the run is flagged via the status taxonomy.

Provider hardening (backward compatible; existing defaults unchanged):

- Each provider constructor gains optional generation params (`max_tokens`, `timeout`, and where supported `thinking`/`effort`) with the current values as defaults.
- `get_provider(spec)` reads `spec["params"]` and passes the supported subset to each constructor.
- Truncation detection is unified through the existing status taxonomy (`core.status.classify_status`), so OpenAI `length`, Google `MAX_TOKENS`, and Anthropic `max_tokens` all classify as `truncated`. No summary code hardcodes a single provider's stop string.

### 2. Shared per-task summary

New `benchpress/runner/summary.py :: per_task_summary(items, results)` returns, per bundle and overall: exact-match %, per-row/per-part %, n, truncated, errors.
It is the single source for the `eval` console output, the `leaderboard` export, and (refactored) the official simulate runner, so the three never drift.

### 3. Score-only leaderboard export

New export format: `python -m benchpress export --format leaderboard --benchmark simulate --out docs/leaderboard.json`.
Per model it emits name, company, open/closed type, launch date, run date, overall and per-task exact and per-row percentages, n, truncated, and errors.
Model metadata is enriched from `config.yaml`.
No prompts, gold, or raw outputs are ever written.
The file is safe to commit publicly by construction.

### 4. GitHub Pages dashboard

Static site under `docs/` (Pages "deploy from /docs on main"; no CI, no secrets).
A single self-contained `index.html` (inline CSS and JS) fetches the same-origin `leaderboard.json`.
Features: sortable leaderboard (overall and per-task; exact vs per-row toggle), filter by open/closed and company, per-task breakdown, methodology and integrity notes, version badge, and visible truncation/error flags so a score is never misread.
Fully data-driven: adding a model is a re-export and commit, no HTML edits.
Seeded initially with the published Opus 4.8 baseline from the frozen manifest so the page has data on day one.

### 5. Professional README

Shields: tests-passing (from the pytest CI workflow), license, Python version, benchmark version, models-evaluated count, judge-free and tools-off badges, and a link to the live dashboard.
Sections: what it is; why (the saturation problem, briefly); design principles; the four tasks with a fully self-contained illustrative example that never uses a scored seed-42 gold; how scoring works; quickstart; how to add a provider or model; the contamination and integrity story; a leaderboard snapshot linking to the dashboard; versioning and freeze policy; citation.

### 6. Lightweight CI

One GitHub Actions workflow that runs `pytest` only, with no API keys and no eval cost, to back the tests badge and catch regressions.

### 7. Fresh-seed holdout audit (documentation)

Document the workflow: run `eval` with a non-42 seed to mint a fresh private set, then `audit --fresh-dir` to compare canonical vs fresh accuracy and flag a suspicious gap.
The plumbing already exists (`stats.audit_gap`, `cli audit`).

## Non-goals

No CI-run evals, no server or database, no new benchmark tiers now, no LLM judge, no committed raw transcripts.

## Build order

1. Provider hardening plus `get_provider` param passing, with tests.
2. `per_task_summary` helper, with tests.
3. `eval` command and `python -m benchpress` entry point, with tests.
4. Score-only `leaderboard` export, with tests.
5. Static `docs/` dashboard (built with the frontend-design skill) plus baseline-seeded `leaderboard.json`.
6. Professional README with shields and a non-leaking example.
7. Pytest CI workflow.
8. MIT license and fresh-seed audit docs.

## Testing

TDD for the code components (provider params, summary, eval, export).
The full existing suite (~389 tests) must stay green throughout.
The dashboard is verified by loading it against the committed `leaderboard.json`.
