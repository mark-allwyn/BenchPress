# Benchpress - Complete Learnings & Resume Guide

_Last updated: 2026-07-13. This is the single source of truth. Read it top to bottom to resume._

## STATUS: Benchpress-Simulate v1 is FROZEN (2026-07-13)

Manifest (immutable): `benchpress/modules/simulate/frozen_v1.json`. Official runner:
`scripts/run_simulate.py`. Freeze-guard test ensures live CONFIG can't drift from the manifest.
389 tests pass. Official config: seed 42, n=25, tools OFF, thinking adaptive/high, max_tokens
64000, read_timeout 900s; only trust `stop_reason == end_turn`.

Opus 4.8 baseline (exact-match / per-row): LIFE 36% / 50.3% - DAYNIGHT 68% / 73.7% -
ECA110 56% / 56% - ECA30 88% / 88%. Clean (only 2/25 LIFE truncated). Non-saturated 36-88% spread
with LIFE as the hard, headroom-rich anchor.

KNOWN LIMITS (accepted at freeze): calibrated vs Opus ONLY (cross-model spread UNVERIFIED - run a
panel when API keys are available, via `scripts/run_simulate.py --model ...`); hard signal leans on
LIFE; n=25 (no CIs). Re-hardening / new tasks = a NEW version (v2), NEVER edit v1.

To evaluate a model: `python scripts/run_simulate.py --model <id> --out results/simulate/<name>.json`
(resume-by-content; re-run to continue). Next high-value step: frontier panel to validate ranking.

---

Branch: `v3`. Repo: `mark-allwyn/BenchPress`. Claude models run **via AWS Bedrock only** (no API
keys - hard security constraint). Model id `eu.anthropic.claude-opus-4-8`, region `eu-central-1`,
account 996083107598. Creds + all keys in `.env` (gitignored). `results/` gitignored.

---

## 1. The goal (unchanged)

A **very hard**, **deterministic**, **judge-free** benchmark ("Benchpress") for **frontier**
models, where:
- Scoring is an exact algorithm (no LLM judge). Gold computed in code.
- Run **tools-off** (with a code interpreter these tasks are trivial - that's the point).
- Official config: **thinking ON**, identical frozen questions for every model (seeded).
- Target: frontier models score **low with headroom for future models** (originally "<55%",
  but the real requirement the user cares about is **NO SATURATION** - not all models clustered
  high with no room to grow). Lower is fine (0/5/50 all acceptable).
- Calibrate difficulty ONCE, then freeze immutably as v1. Re-hardening = a new version.

---

## 2. THE BIG FINDING: Opus 4.8 has saturated deterministic causal reasoning

We spent the bulk of the effort proving this, the hard way. **Do not re-litigate it.**

Across **16+ causal-judgment types**, a thinking Opus 4.8 (tools-off, fair token budget) is at
**~99-100%**. Everything with a clean procedure, it executes flawlessly.

- **Concept/arithmetic/counterfactual bundles**: ~100%. Dropped early.
- **Counting tasks** (linear extensions, open paths, v-structures, MEC, directed paths, adjustment-set
  counts, d-sep-pair counts, undirected-path counts): looked hard at a **stingy 16k token budget**
  (Opus 4-40%) but that was **TOKEN TRUNCATION, not reasoning**. Re-run at 40k they jump to 75-100%.
  A thinking model just enumerates in its scratchpad. **Counting is not hard for frontier models.**
- **All-or-nothing batteries** (amplify a sub-100% per-question error via p^N): DSEP (d-separation)
  was the only survivor at 16k. But at a fair 40k it's ~42-68% and **highly sample-variant**, because
  **errors cluster by instance, not independently** - so pass rate ≈ "fraction of easy instances,"
  NOT p^N. Lengthening the battery barely helps. Backdoor/frontdoor/IV/cond-change/minsep atomic
  judgments measured 96-100% per-question; only composite front-door (~96-97%) dipped.
- **Adversarial mining** (generate ~4000 sub-questions, keep the ones Opus fails): harvested only 34
  failures; on re-test 33 were **stochastic noise** (Opus gets them right on a retry).
- The lone "robust" signal (23 front-door cases failing 3/3) was a **GOLD BUG**: our
  `frontdoor_set_valid` was wrong and **Opus was right**. (See section 5.)

**Lesson distilled:** a thinking frontier model resists a task only when there is
**(a) no shortcut/closed form AND (b) no error-cancellation.** Counting has shortcuts or is
token-bound; batteries let errors cancel/cluster; single judgments are just executed correctly.

---

## 3. THE PIVOT THAT WORKS: long-horizon deterministic SIMULATION

"Faithfully run this machine for N steps; report the exact final state." This is the domain that
genuinely resists Opus, because:
- **No shortcut** - many systems are Turing-complete; you must actually simulate step by step.
- **Errors compound serially** - one wrong step corrupts the final state (no cancellation, unlike
  batteries; no self-correction, unlike counting).
- **Tools-off is the whole point** - with code it's a one-liner.
- **Gold is a trivially-correct simulator** (~zero bug risk - the opposite of the front-door mess).
- **No saturation risk from "finding the trick"**: unlike counting, a stronger model can't shortcut
  it - it can only track more steps accurately. So it measures a real capability with genuine headroom.

### Confirmed genuinely hard (not just token-bound)
- **Conway's Game of Life** is the anchor. Re-running the SAME items at 40k vs 64k did NOT rescue it
  (held ~38-50% exact / ~82% per-cell), so at the right settings the difficulty is real reasoning.

### The central tension (important, not yet fully resolved)
These tasks make Opus **write out the whole trajectory**, so tokens & wall-clock scale with
state x steps. The hard regime **collides with the token cap and the read-timeout**:
- Too few steps / small grid -> Opus nails it (Day&Night, Brian's Brain, small ECA all ~88-100%).
- Too many steps / big grid -> Opus **truncates** (hits max_tokens) or the request **read-times-out**
  (thinks >300s). Scores then reflect the cap, not reasoning.
- The clean-hard sweet spot (finishes with `stop=end_turn`, no truncation, still fails) is **narrow
  and noisy** at small sample sizes. Exact-match is very high-variance; **per-cell accuracy is the
  stable, graded, non-saturating metric with headroom** - USE IT as primary.

### Provider fixes made because of this
- `benchpress/providers/bedrock.py`: `read_timeout` is now a constructor arg, **default 900s**
  (was 300s). Hard sim items legitimately think for minutes. ALWAYS verify results are
  `stop_reason == end_turn` (not `max_tokens` = truncated, not `None`/error = timeout).
- 7x7 grids are reliable. 8x8+ and very long trajectories hit read-timeouts / large-output issues.

### Empirical difficulty readings (small-sample, NOISY - treat as directional)
| task | setting | exact | per-cell | notes |
|---|---|---|---|---|
| Conway's Life | 7x7, 7 gen | ~0-50% | ~54-82% | genuine; budget-sensitive; anchor |
| Conway's Life | 7x7, 8-12 gen | 12-62% | 49-92% | harder but pushes toward truncation |
| Day & Night | 7x7, small gen | ~88% | ~93% | easy unless many gens (then truncates) |
| Brian's Brain | 7x7 | - | - | DROPPED - dies out on small grids at every setting |
| ECA Rule 110 | W30, N30-32 | 0-68% | 49-68% | 1D, cheaper tokens; N is the knob |
| ECA Rule 30 | W30-40, N30-45 | 0-96% | - | chaotic; easy at small N |
| big-grid/few-gen (9x9/4) | - | 67% | 98% | EASY - few steps = accurate |

---

## 4. WHERE WE ARE RIGHT NOW (resume here)

### The `simulate` module is built and is the current benchmark
`benchpress/modules/simulate/` (registered as module `"simulate"`, VERSION "1"):
- `sim.py` - pure, trivially-correct simulators: `seeded_grid`, `seeded_row`, `life_like(grid,gens,
  born,survive)`, `brians_brain`, `eca(row,rule,steps)`, `rows_str`. All toroidal.
- `__init__.py` - `CONFIG` per bundle + builders + `generate(seed, "hard", n)`.
- **Bundles (tasks) & current CONFIG:**
  - `LIFE`     - Conway B3/S23, 7x7, **7 gen**, density 0.40
  - `DAYNIGHT` - B3678/S34678, 7x7, **7 gen**, density 0.45
  - `ECA110`   - Rule 110, W30, **N35**
  - `ECA30`    - Rule 30, W30, **N35**
- **Scoring** reuses the battery framework: each 2D item's PARTS = the grid ROWS (categorical
  exact-match). `item_correct` = exact grid match (harsh headline); **per-row marginal = graded
  metric** (per-part accuracy). ECA items = 1 part (final row). Answer format = labelled lines
  `ROW1: <digits>` ... which `parse_tagged_fields` maps to parts.
- **Tests**: `benchpress/tests/test_simulate.py` (7 tests) incl. an **independent gold-recompute**
  from stored `gen_params`. Full suite ~388 tests, all passing at last run.

### A run is IN FLIGHT (will die on your restart - that's fine, it resumes)
- `scratchpad/confirm_sim.py` -> `results/simulate/confirm-64k.json`
- Opus 4.8 + thinking, **max_tokens 64000**, 25 items x 4 tasks (100 items), workers=2.
- `run_model` writes atomically and is **resume-by-content**: on restart just re-run the script;
  completed items skip, errored/timed-out ones retry. **Nothing is lost.**
- It reports per-task exact-match + per-row accuracy + truncation + error counts.

### IMMEDIATE NEXT STEP after restart
1. Re-run the confirm to finish it (resumes):
   `python scratchpad/confirm_sim.py` (env-loads AWS creds from `.env`; note the scratchpad path
   may differ in a new session - the script is reproducible from this doc: `generate(42,"hard",25)`
   on module `simulate`, BedrockProvider `eu.anthropic.claude-opus-4-8` thinking=adaptive effort=high
   max_tokens=64000, run_model workers=2, then score per-task exact + per-row).
2. Read the stable 25-item table. Decide freeze on the **per-cell/per-row graded metric** (headroom),
   with exact-match as the harsh headline.
3. Tune any task off-target via `gens` (2D) / `N` (ECA). Verify `stop_reason` is `end_turn`
   (NOT `max_tokens`) for ~all items - if truncating, either raise the fixed budget or lower gens/N.
4. Freeze v1: lock seed(=42) + CONFIG + budget(64k) + timeout(900s) in an immutable manifest.

---

## 5. KNOWN BUGS / CLEANUP DEBT
- **`benchpress/modules/frontier/graphs.py :: frontdoor_set_valid` IS BUGGY** (over-accepts). Its
  FD3 deletes out-edges of every Z node, which disconnects Y (a mediator's child) and falsely
  reports "valid". The **correct** implementation is the explicit active-trail version in
  `scratchpad/verify_fd_gold.py` (`fd_independent`). The whole `frontier` module (causal batteries:
  DSEP/BACKDOOR/FRONTDOOR/CONDCHANGE/IV) is **superseded/saturated** - decide whether to delete it or
  fix the gold and keep it as an archived "causal (saturated)" tier. Do NOT ship it as-is.
- The `frontier` causal module and the new `simulate` module currently coexist. Benchpress v1 =
  `simulate`. Old causal `causal` module also still present (legacy).

## 6. LOCKED DESIGN RULES
- **Tools-off, thinking-on, FIXED generous token budget (64k), read_timeout 900s.** Scores must
  reflect reasoning, not truncation/timeout: always check `stop_reason == end_turn`.
- Deterministic: `generate(seed=42,"hard",n)` -> identical items. Freeze = lock seed+CONFIG+budget.
- No LLM judge, ever. Gold = trivially-correct simulator.
- Per-cell/per-row accuracy is the primary (graded, headroom, stable) metric; exact-match is the
  headline. Report both.
- Calibrate once, freeze; re-hardening = a new version, never edit v1.

## 7. KEY SCRIPTS (in the session scratchpad; reproducible from this doc)
- `confirm_sim.py` - the 25-item simulate confirm (IN FLIGHT).
- `probe_sim.py`, `probe_sim2.py`, `calibrate_sim.py`, `calibrate2.py`, `probe_budget.py` - the
  simulation probes/calibration (ECA, Life, Day&Night, Brian's Brain, VM, Turing machine, rewriting).
- `probe_atomic*.py`, `probe_harden*.py`, `mine_pool.py`, `verify_mined.py`, `verify_fd_gold.py`,
  `verify_tokens.py`, `verify_dsep.py` - the causal-domain exploration (why it's saturated + the
  front-door gold bug proof).
- Bedrock provider pattern: load AWS_* from `.env` into os.environ, then
  `BedrockProvider("eu.anthropic.claude-opus-4-8", region=..., thinking="adaptive", effort="high",
  max_tokens=64000)`. Score with `benchpress.runner.score.score_response(item, content, stop_reason, error)`.

## 8. DEAD ENDS (do not repeat)
- Any pure-counting task (token-bound or shortcut-able).
- Causal-graph structure judgments (saturated: d-sep, backdoor, front-door, IV, cond-change, minsep,
  MEC, v-structures, path-activity, intervention).
- SEM numeric (total effect/covariance/mediation): 98-100%, saturated.
- Turing-machine sim / string-rewriting / register-VM at small sizes: ~75-100%, easy. (Larger sizes
  truncate before getting reliably hard.)
- Brian's Brain on small grids: dies out.
- Stingy token budgets to manufacture difficulty: that measures tokens, not reasoning. FORBIDDEN.
