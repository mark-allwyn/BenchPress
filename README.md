<div align="center">

# Benchpress

**A hard, deterministic, judge-free, tools-off benchmark for frontier language models.**

Faithfully run a machine for N steps, then report the exact final state.
No shortcut exists, one wrong step corrupts the rest, and the answer is scored by code, not a judge.

[![tests](https://github.com/mark-allwyn/BenchPress/actions/workflows/tests.yml/badge.svg)](https://github.com/mark-allwyn/BenchPress/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org)
[![benchmark](https://img.shields.io/badge/benchmark-Simulate%20v1-d4472a.svg)](https://mark-allwyn.github.io/BenchPress/)
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

Opus 4.8 sits at a non-saturated **36 to 88 percent** across the four tasks, with Conway's Life as the hard, headroom-rich anchor.

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
| **LIFE** | 2D life-like | Conway `B3/S23` | 7×7 toroidal | 7 gen | 36% / 50.3% |
| **DAYNIGHT** | 2D life-like | `B3678/S34678` | 7×7 toroidal | 7 gen | 68% / 73.7% |
| **ECA110** | 1D elementary | Rule 110 | 30 cells cyclic | 35 | 56% / 56.0% |
| **ECA30** | 1D elementary | Rule 30 | 30 cells cyclic | 35 | 88% / 88.0% |

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

## How scoring works

Each item's answer is a set of labelled rows (`ROW1: ...`).
Two metrics come out of every item:

- **Exact match** is conjunctive: the item is correct only if every row is correct. This is the harsh headline.
- **Per-row accuracy** is the fraction of rows correct. This is the graded, lower-variance, non-saturating primary metric.

A one-dimensional ECA item has a single scored row, so its two metrics coincide.
Scoring is pure string comparison after a status check that separates a genuine answer from a refusal, a truncation, or an API error.

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
=== Benchpress-simulate v1 - <model> ===
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

Simulate **v1** is frozen (see [`benchpress/modules/simulate/frozen_v1.json`](benchpress/modules/simulate/frozen_v1.json)).
The official configuration is: seed 42, 25 items per task, tools off, thinking adaptive at high effort, a 64000-token budget, and a 900-second read timeout.
Only responses that finished cleanly (`stop_reason == end_turn`, or a provider equivalent) count toward a score.

A freeze-guard test asserts the live config cannot drift from the manifest.
Any change to the tasks, sizes, or budget is a new version, published alongside v1 rather than replacing it.

Known limits, accepted at freeze: the difficulty is calibrated against Opus 4.8 only, the hard signal leans on LIFE, and n is 25 per task (no confidence intervals).
Running a cross-model frontier panel to validate the ranking is the next step, and exactly what the pipeline in this repo is for.

## Repository layout

```
benchpress/
  cli.py                     command-line entry (eval / run / score / export / audit / freeze)
  frozen.py                  loads a benchmark's frozen manifest + official run-config
  leaderboard.py             builds the score-only public leaderboard payload
  modules/simulate/          the Simulate v1 benchmark (sim.py, generators, frozen_v1.json)
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
