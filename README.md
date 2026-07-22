<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img src="docs/assets/logo-light.svg" alt="Benchpress" height="52">
</picture>

**A hard, deterministic, judge-free, tools-off benchmark for frontier language models.**

Faithfully run a machine for N steps, then report the exact final state.
No shortcut exists, one wrong step corrupts the rest, and the answer is scored by code, not a judge.

[![tests](https://github.com/mark-allwyn/BenchPress/actions/workflows/tests.yml/badge.svg)](https://github.com/mark-allwyn/BenchPress/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org)
[![benchmark](https://img.shields.io/badge/benchmark-Simulate%20v2-d4472a.svg)](https://mark-allwyn.github.io/BenchPress/)
[![scoring](https://img.shields.io/badge/scoring-judge--free-2ea043.svg)](#how-scoring-works)
[![eval](https://img.shields.io/badge/eval-tools--off-8957e5.svg)](#design-principles)

[**Live leaderboard**](https://mark-allwyn.github.io/BenchPress/) · [Quickstart](#quickstart) · [The tasks](#the-tasks) · [Integrity](#contamination-and-integrity)

</div>

---

## What Benchpress measures

One capability: whether a model can **hold a precise state in its head and update it correctly across many sequential steps, without drifting.**

Each item gives the model the exact rules of a small deterministic machine (a cellular automaton), a starting state, and a step count.
The model has to carry out every step itself, with no tools, and report the exact final state.
There is no formula that shortcuts the answer.
The systems are Turing-complete or chaotic, so the only path to the result is to actually simulate it, step by step.

## Why it is valuable

Long-horizon, step-by-step reliability is the foundation of agentic and procedural work: following a multi-step process, executing an algorithm, or holding state across a long tool-using trajectory.
A model that loses the thread at step 15 of a cellular automaton will lose it at step 15 of a real workflow.
Most benchmarks hide this failure mode, because they use short answers, allow tools, or grade with a rubric.
Benchpress isolates and quantifies it.

What makes the difficulty real:

- **No shortcut.** There is no closed form, so a score reflects actual execution, not a recalled fact or a clever trick.
- **Errors compound.** A single wrong cell early on corrupts every later step, so a good score requires sustained, error-free work rather than a lucky partial.
- **No saturation.** Because there is no trick to discover, a stronger model can only score higher by tracking more steps correctly. The benchmark keeps discriminating as models improve, instead of everyone clustering at 100%.

What makes the number trustworthy:

- **Judge-free.** The answer is computed by a reference simulator and compared byte-for-byte. No LLM judge, no rubric, no ambiguity.
- **Tools-off.** A code interpreter makes every task a five-line program, so removing tools measures the model's own execution rather than its ability to call a tool.
- **Contamination-resistant.** Questions are generated from a seed, so a fresh, never-seen variant can be minted at any time to check whether a model memorised the public set.

Across a 14-model panel the scores span a wide, non-saturated range - the frontier tops out near **67% exact** while every open-weight model floors **below 25%** - with Conway's Life as the hardest anchor. See [Results](#results) below.

## Design principles

- **Deterministic.** Items are generated from a fixed seed. The same seed reproduces the exact same questions for every model.
- **Judge-free.** Gold is computed by a trivially-correct reference simulator and compared byte-for-byte. There is no LLM judge, ever.
- **Tools-off, thinking-on.** Every model runs the identical frozen configuration: tools off, thinking on, one generous fixed token budget.
- **Truncation is quarantined.** A response cut off by the token cap, or a timeout, is recorded as such and never scored as a wrong answer.
- **Frozen once, then immutable.** Calibrated difficulty is locked in a manifest. Re-hardening is a new version (v2), never an edit to v1, so scores stay comparable over time.

## The tasks

Every prompt is fully self-contained.
The model is given the complete rule, the starting state, the number of steps, and the answer format.
Nothing is withheld, and the model does not need to recognise the automaton by name.

| Task | Family | Rule | Grid | Steps | Opus 4.8 (exact / per-row) |
|------|--------|------|------|-------|----------------------------|
| **LIFE** | 2D life-like | Conway `B3/S23` | 7×7 toroidal | 7 gen | 32% / 44.6% |
| **DAYNIGHT** | 2D life-like | `B3678/S34678` | 7×7 toroidal | 7 gen | 72% / 74.9% |
| **ECA110** | 1D elementary | Rule 110 | 30 cells cyclic | 35 | 44% / 44.0% |
| **ECA30** | 1D elementary | Rule 30 | 30 cells cyclic | 35 | 84% / 84.0% |

An **illustrative** prompt (a small toy, not one of the scored items):

```
Elementary cellular automaton, Rule 90, CYCLIC boundary. Each step every cell
is replaced simultaneously using (left,center,right):
  111 -> 0   110 -> 1   101 -> 0   100 -> 1
  011 -> 1   010 -> 0   001 -> 1   000 -> 0

Initial row (5 cells): 00100
Evolve for EXACTLY 2 steps.

Reply with exactly:
ROW1: <5 digits>
```

The real scored items use the sizes in the table above.
Their gold answers are never stored in this repository. They are recomputed at run time by the simulator in [`benchpress/modules/simulate/sim.py`](benchpress/modules/simulate/sim.py).

## How scoring works, and how to read the leaderboard

Each item's answer is a grid written as labelled rows (`ROW1: ...`, `ROW2: ...`).
Every item produces two numbers, and the leaderboard shows both.

**Exact match** is all-or-nothing: an item counts only if **every** row is correct.
It is the fraction of grids the model got completely right, and it is the leaderboard's default headline number.

**Per-row accuracy** is the fraction of individual rows the model got right, across all items.
It gives partial credit: get 5 of a Life grid's 7 rows right and that item contributes 5/7, even if the grid as a whole is wrong.
It is the graded, lower-variance, non-saturating secondary metric, useful because it keeps discriminating when no model gets a full grid.

On the leaderboard, each task cell shows both numbers (toggle which one is the large headline); on the ECA tasks they are identical.

A one-dimensional ECA item has a single scored row, so for those tasks the two metrics are identical.
A model can score well on per-row while scoring 0% exact if it drifts by only a cell or two per grid.

Worked example, Sonnet 5 on LIFE: **56% exact / 69.7% per-row**.
It got 56% of the Life grids completely right, and 69.7% of all individual rows right, meaning the grids it missed were mostly correct apart from a cell or two.

Reading the flags and labels:

- **Lower is harder.** These are hard tasks; a low score is the expected, informative result, not a bug.
- **⚑ (truncation)** means the response hit the 96000-token budget before finishing.
  Reasoning-heavy models can spend the whole budget thinking and never emit the final grid.
  A truncated item never produced a complete answer, so it counts as not correct, but it is flagged rather than silently scored wrong, because it reflects a capacity limit rather than a reasoning error.
- **Thinking-on only.** The ranked board contains only models run with extended thinking. Models with no thinking mode (e.g. Amazon Nova) aren't comparable, so they're reported separately as a non-reasoning floor control rather than ranked.

Scoring itself is pure byte-exact string comparison against the reference simulator, after a status check that separates a genuine answer from a refusal, a truncation, or an API error.
There is no LLM judge.

## Results

Simulate v2 (96k budget), 14-model panel. The full sortable board is on the [live dashboard](https://mark-allwyn.github.io/BenchPress/); the top of it:

| # | Model | Vendor | Exact | Per-row |
|---|-------|--------|------:|--------:|
| 1 | Claude Opus 4.6 | Anthropic | 67% | 78.8% |
| 2 | Claude Sonnet 4.6 | Anthropic | 67% | 54.2% |
| 3 | Claude Opus 4.8 | Anthropic | 58% | 60.2% |
| 4 | Claude Opus 4.7 | Anthropic | 49% | 58.0% |
| 5 | Claude Sonnet 5 | Anthropic | 47% | 56.0% |
| 6 | minimax-m2.5 | MiniMax (open) | 24% | 46.8% |
| 7-13 | gpt-oss-20b/120b, qwen3-235b/coder, minimax-m2.1, nemotron, glm | open | ≤9% | ≤14% |

The ranked board is **thinking-on models only**. A non-reasoning model (Amazon Nova Pro, no extended-thinking mode) was run as a separate control and **floored at 0% exact / 4% per-row** - confirming the task requires reasoning, not recall.

Two findings stand out:

1. **Vendor separation.** The top five are all Anthropic, then a sharp cliff. The strongest open-weight model (minimax-m2.5) reaches 24% exact; the rest floor near zero.
2. **Newer is not always better.** Opus 4.6 (67%) beats Opus 4.8 (58%) and Opus 4.7 (49%); Sonnet 4.6 (67%) beats Sonnet 5 (47%). The Opus 4.6 result reproduced across two independent runs (LIFE 92% then 88%), so it is a genuine regression on long-horizon simulation, not sampling noise.

## Quickstart

```bash
git clone https://github.com/mark-allwyn/BenchPress.git
cd BenchPress
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# declare the models you want to test
cp config.example.yaml config.yaml   # then edit, and export the API keys it references

# run one model end to end (generate -> run -> score -> summary)
python -m benchpress eval --model gpt-5 --benchmark simulate --workers 4
```

`eval` writes crash-safe, resume-by-content results to `results/simulate/<model>.json`.
Re-running the same command continues an interrupted evaluation rather than starting over.

Example output (layout only; numbers are placeholders, not a real run):

```
=== Benchpress-simulate v2 - <model> ===
task         exact   per-row    n  trunc  err
LIFE           . . .    . . .    25      .    .
DAYNIGHT       . . .    . . .    25      .    .
ECA110         . . .    . . .    25      .    .
ECA30          . . .    . . .    25      .    .
OVERALL        . . .    . . .   100      .    .
```

The only measured numbers published anywhere in this repo are the Opus 4.8 baseline in the task table above and in the frozen manifest.

## Adding a model or provider

Any provider works through the same path.
Add an entry to `config.yaml` and point `--model` at it.

```yaml
models:
  my-model:
    provider: openai        # anthropic | bedrock | openai | openai_compatible | google | cohere | ollama
    model: the-model-string
    company: Vendor
    type: closed            # open | closed
    api_key_env: OPENAI_API_KEY
```

Claude can run either via a direct Anthropic key (`provider: anthropic`) or via AWS Bedrock (`provider: bedrock`), against the identical frozen questions.
See [`config.example.yaml`](config.example.yaml) for one entry per provider.

To support a new backend, add an adapter under `benchpress/providers/` implementing `complete(prompt) -> CompletionResult` and register it in `get_provider`.

## Contamination and integrity

Because gold is generated at run time and never committed, this repository leaks no answers.
The remaining risk is that the frozen prompts, once sent to a provider, are logged and later trained on.

Generation is seeded and deterministic, which turns that risk into a test:

```bash
# canonical public set is seed 42; mint a fresh private holdout with any other seed
python -m benchpress eval --model my-model --benchmark simulate --seed 7 \
    --results-dir results-fresh

# compare canonical vs fresh accuracy; a large drop flags likely memorisation
python -m benchpress audit --benchmark simulate --results-dir results \
    --fresh-dir results-fresh
```

A model that memorised the canonical set scores well on seed 42 and poorly on the structurally identical fresh set.

## The dashboard

The [live leaderboard](https://mark-allwyn.github.io/BenchPress/) is a static page under [`docs/`](docs/), served by GitHub Pages.
It reads a score-only `docs/leaderboard.json` that contains numbers and model metadata only, never prompts, gold answers, or raw model output.

To publish your own runs:

```bash
python -m benchpress export --benchmark simulate --format leaderboard \
    --out docs/leaderboard.json
git add docs/leaderboard.json && git commit -m "update leaderboard" && git push
```

Pages redeploys on push. Adding a model is a re-export and a commit, with no changes to the page itself.

## Versioning and freeze policy

The current official config is **Simulate v2** (see [`benchpress/modules/simulate/frozen_v2.json`](benchpress/modules/simulate/frozen_v2.json)): seed 42, 25 items per task, tools off, thinking adaptive at high effort, a **96000-token budget**, and a 1800-second read timeout.
Only responses that finished cleanly (`stop_reason == end_turn`, or a provider equivalent) count toward a score.

**v2 keeps v1's tasks and seed unchanged and only raises the budget from 64000 to 96000.**
A budget ablation showed the 64k cap was invalidly penalising verbose reasoners: raising it to 96k lifted Sonnet 4.6 from 16% to 67% exact and minimax-m2.5 from 12% to 24%, with nothing else changed, while 96k captures essentially all reasoning (max output observed ~107k tokens).
Because a budget change alters scores, it is a new version ([`frozen_v1.json`](benchpress/modules/simulate/frozen_v1.json) is retained, immutable); scores are comparable within a version only.

Known limits: n is 25 per task (no confidence intervals, so close rankings such as within the Anthropic cluster are not statistically separable), and even at 96k the most verbose models (e.g. Sonnet 4.6) still truncate a handful of items, so their true scores may be marginally higher.

## Repository layout

```
benchpress/
  cli.py                     command-line entry (eval / run / score / export / audit / freeze)
  frozen.py                  loads a benchmark's frozen manifest + official run-config
  leaderboard.py             builds the score-only public leaderboard payload
  modules/simulate/          the Simulate benchmark (sim.py, generators, frozen_v1.json + frozen_v2.json)
  providers/                 vendor adapters (anthropic, bedrock, openai, google, cohere, ollama)
  runner/                    run / score / summary / persistence (atomic, resume-by-content)
  scorers/                   part scorers (categorical, numeric, set, sequence, edge-list)
  stats/                     accuracy, bootstrap CIs, item analysis, contamination audit
  tests/                     the pytest suite
docs/                        the static GitHub Pages dashboard + leaderboard.json
config.example.yaml          model registry template (copy to config.yaml)
```

## License

MIT. See [LICENSE](LICENSE).
