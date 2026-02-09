# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Personal LLM evaluation harness. Run the same set of prompts against different models, auto-score responses via LLM-as-judge (1-5) and DeepEval G-Eval (0-1), and compare over time. Results persist as per-model JSON files in `results/`.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp config.example.yaml config.yaml  # then add API keys

# Evaluate a model (requires API key env vars set; scores via LLM judge inline)
python run.py eval claude-sonnet-4
python run.py eval claude-sonnet-4 --ids C01 L02
python run.py eval claude-sonnet-4 --category coding --difficulty hard
python run.py eval claude-sonnet-4 --rerun  # appends new run, keeps history

# Compare models
python run.py compare
python run.py compare --category coding --save

# Re-judge / DeepEval (retroactive scoring on existing results)
python run.py rejudge                              # re-judge all models
python run.py rejudge claude-sonnet-4 --force      # force re-judge
python run.py deepeval                             # DeepEval all models
python run.py deepeval gpt-4o --ids C01 --force    # specific prompts

# Dashboard (auto-generated after each eval too)
python run.py dashboard
python run.py dashboard --open  # opens in browser

# List
python run.py models
python run.py prompts
```

There are no tests. No linter or formatter is configured.

## Architecture

Single-entrypoint CLI (`run.py`) with five supporting modules:

- **`run.py`** - CLI dispatcher with argparse subcommands (`eval`, `compare`, `models`, `prompts`, `rejudge`, `deepeval`, `dashboard`). Also contains the data layer (load/save model results as JSON, filter prompts).
- **`scripts/providers.py`** - Provider abstraction. `Provider` ABC with `complete(prompt, params) -> (content, usage_dict)`. Implementations: `AnthropicProvider`, `OpenAIProvider` (also handles `openai_compatible`), `GoogleProvider`. All use raw `httpx` HTTP calls, no SDKs. Factory function: `get_provider(config)`.
- **`scripts/checks.py`** - Automated response checkers. `check_response(prompt_meta, response)` dispatches to per-`check_type` functions via `CHECKERS` dict. Returns `{flags, auto_scores, passed}`. Checkers are string-matching heuristics, not LLM-based.
- **`scripts/judge.py`** - LLM-as-judge scoring. `judge_response()` sends the prompt, ideal answer, criteria, and response to a separate LLM for 1-5 scoring with rationale. Runs inline during `eval`. Configured via `judge` section in `config.yaml`.
- **`scripts/deepeval_scorer.py`** - DeepEval G-Eval integration. Scores responses on correctness, coherence, and instruction following (0-1 scale). Runs inline during `eval` if enabled, or retroactively via `deepeval` subcommand.
- **`scripts/dashboard.py`** - Generates a self-contained HTML dashboard (`docs/dashboard.html`) with Chart.js visualizations. Shows leaderboard, category breakdowns, score distributions, latency comparisons, and auto-check flags. Also generates `categories.html` and `methodology.html`. Auto-regenerated after each eval, rejudge, and deepeval run.

## Key Design Decisions

- **No SDKs** - providers use `httpx` directly against each API's REST endpoints, not `anthropic`/`openai` Python packages.
- **Append-only history** - `--rerun` appends a new run entry; `latest_run()` picks the last one for scoring/comparison.
- **Auto-checks supplement judge scoring** - the `check_type` field on each prompt routes to a specific checker. Auto-check flags are passed to the LLM judge as additional signal. Many check types (`reasoning`, `calibration`, `format_check`, `checklist`) are no-ops that rely entirely on judge scoring.
- **Three-layer scoring** - auto-checks (heuristic), LLM judge (1-5), and DeepEval G-Eval (0-1) combine into a composite score for ranking.
- **Composite score** - weighted average of normalised judge score and DeepEval average, both on 0-1 scale. Configurable via `composite` section in `config.yaml`.
- **Model config in YAML, prompts in JSON** - `config.yaml` defines providers/keys/params; `evals/default.json` defines the prompt set.

## Adding New Providers

Add a `Provider` subclass in `scripts/providers.py`, implement `complete()`, and register it in `get_provider()`. The `openai_compatible` provider type already supports vLLM/Ollama/Together/Groq via `base_url`.

## Adding New Auto-Checks

Add a checker function in `scripts/checks.py` and register it in the `CHECKERS` dict. Each checker takes `(prompt_meta, response)` and returns `{"flags": [...], "auto_scores": {}}`.

## Prompt Schema

Each prompt in `evals/default.json` requires: `id`, `category`, `subcategory`, `difficulty`, `prompt`, `ideal`, `criteria`, `check_type`. Some check types require additional fields (`target_word_count`, `constraints`, `trap`, `should_refuse`).
