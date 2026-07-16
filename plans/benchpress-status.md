# Benchpress - current status & resume guide

_Last updated: 2026-07-07. Read this first to continue the work._

## LATEST (2026-07-10): SIMULATION module built; confirming.

New module `benchpress/modules/simulate/` (VERSION 1), registered "simulate". Sim gold in
`sim.py` (trivially-correct: Conway/life-like, Brian's Brain, ECA - all toroidal). Tasks =
bundles: LIFE (Conway B3/S23, 7x7, 12 gen), DAYNIGHT (B3678/S34678, 7x7, 16 gen), ECA110,
ECA30 (both W30 N30, 1D). Dropped BRAIN (Brian's Brain dies out on 7x7 at every setting).
Each 2D item's PARTS = grid ROWS (categorical exact-match): item_correct = exact grid match
(headline), per-row marginal = graded score with headroom. ECA = 1 part (final row).
Answer format: "ROW1: <digits>" labelled lines. Budget FIXED 48k (7x7 reliable; 8x8/64k hit
Bedrock output errors). 388 tests pass (incl. test_simulate.py w/ independent gold recompute).

Confirmed-genuine (not token-bound): LIFE held ~38-50% exact / ~82% cell going 40k->64k.
Probed difficulty (8-item): LIFE 62%/92% cell, DAYNIGHT easy@8gen->pushing to 16, ECA ~75-88%.
IN FLIGHT: `scratchpad/confirm_sim.py` -> results/simulate/confirm-48k.json (25 items x4 tasks,
workers=2, resumable). NEXT: read per-task exact + per-row; tune gens so each lands ~40-65% exact;
freeze v1. If DAYNIGHT/ECA too easy, LIFE is the anchor + tune others or add Langton's ant.

NOTE: old `frontier` module has a KNOWN-BUGGY frontdoor gold (see below) - fix or remove at cleanup.

## (superseded) CAUSAL DOMAIN IS DEAD (saturated) - 2026-07-09

Adversarial mining proved the causal domain is fully saturated for Opus 4.8: from ~4000
mined sub-questions only 34 "failures", and on re-test 33 were stochastic noise (0/3). The
lone "robust" signal (23 FRONTDOOR cases failing 3/3) turned out to be a GOLD BUG: my
`frontdoor_set_valid` FD3 deletes out-edges of every Z node, which disconnects Y (a mediator's
child) and falsely reports "valid". An independent explicit-enumeration checker AND Opus both
disagree with it -> Opus was RIGHT, my oracle was wrong. So: Opus out-reasoned my causal oracle;
the domain gives zero headroom. `frontdoor_set_valid` in graphs.py is KNOWN-BUGGY (fix or drop
if reused; the correct impl is the explicit-enumeration one in scratchpad/verify_fd_gold.py).

**NEW DOMAIN THAT WORKS: long-horizon deterministic SIMULATION.** "Faithfully run this machine
N steps, report the exact final state." No shortcut (must simulate), errors COMPOUND serially
(no cancellation, no self-correction), tools-off is the whole point, gold = a trivially-correct
simulator (near-zero bug risk, unlike causal criteria), precisely tunable (state-width x steps),
graded via per-cell accuracy for headroom. First probe @40k (0 truncation = real reasoning, not
tokens):
- ECA (Rule 110, cyclic, W=25 N=20 = 500 updates): exact 60%, per-cell 93.6%, ~17k tok. HARD + tunable.
- VM (register machine w/ loop): 90% - too easy, hardenable (nested loops, reg-to-reg mul).
Probing more sim tasks for a 4-5 test suite: Game of Life (2D), Turing machine, string rewriting,
harder ECA. See scratchpad/probe_sim.py, probe_sim2.py. NEXT: pick suite, calibrate each to ~40%
exact (tune W,N), build a NEW module (e.g. `simulate`), 25-item confirm, freeze.

## (superseded 2026-07-09) causal-battery notes below

## (2026-07-07): batteries can't cap Opus; pivoted to ADVERSARIAL MINING

Rebuilt the module to 3 battery tests (FRONTDOOR/IV/SEM, VERSION "2"). The 25-item
confirm @40k (`scratchpad/confirm_battery.py`) came back TOO EASY:
FRONTDOOR 64%, IV 96%, SEM 84% (0 truncation). Two reasons the battery idea under-delivers:
1. **Errors cluster by instance, not independently** - so pass rate ≈ fraction of easy
   graphs, NOT p^N. Lengthening batteries barely helps. (SEM: per-q 98.4% predicts ~48%
   pass, actual 84% - 21/25 items perfect, errors piled in 4 hard graphs.)
2. **Huge item-draw variance** - FRONTDOOR swung 44%->64% just from the generator seed
   (bundle index feeds the RNG base, so lineup order changes which items appear).

Conclusion: random-instance batteries CANNOT reliably hold Opus 4.8 under 55% here -
Opus is at/near ceiling on ~13 of 16 causal-judgment types probed. User's decisive
concern = SATURATION (all frontier models high, no headroom), which rules out a plain
discrimination panel. **Chosen path: adversarial hard-instance mining** (how FrontierMath/
ARC/HLE are built) - generate a big pool, keep only what Opus fails -> low score + headroom.
Bedrock-only means we mine against Opus alone (hardness is Opus-calibrated but structurally
objective; verify with a panel later if API keys ever allowed).

**In flight:** `scratchpad/mine_pool.py` - generate(seed=777, n=40) = 120 candidate items,
run Opus @40k (results/frontier/mine-pool.json, resumable), grade per-question, harvest the
FAILED questions (+graph+params) to results/frontier/mined_hard.json. Next: assemble frozen
hard items from the harvest, baseline Opus per-question, freeze v1. Open design issue: yes/no
tasks need concentration (multiple hard Qs per graph) to beat the 50% guess floor; SEM is
numeric (no floor). See yields in scratchpad/mine_pool.log.

---
## (superseded) earlier battery design notes below

## CRITICAL PIVOT (2026-07-06): counting tests are dead; batteries are the design

The prior "5 counting tests" were calibrated at a **16k token budget**. That was a mistake:
their hardness was **token truncation, not reasoning**. Re-running the SAME items at a fair
**40k budget** collapsed them:

| Old test | @16k | @40k (fair) | verdict |
|---|---|---|---|
| LINEXT (count orderings) | 40% | ~75-90% | truncation artifact - DEAD |
| OPENPATH (count active paths) | 4% | 92% | truncation artifact - DEAD |
| UPATHS / DIRPATHS / ADJSETS / DCPAIRS | - | ~100% | DP/shortcut or truncation - DEAD |
| DSEP (50-q all-or-nothing battery) | 28% | **42%** | GENUINE - survives |

**Lesson:** a thinking frontier model with enough tokens just enumerates in its scratchpad.
The ONLY robust mechanism is an **all-or-nothing battery of an atomic causal judgment the model
cannot reason to 100% on**. DSEP works because per-question d-sep accuracy plateaus at ~98% and
0.98^50 ≈ 40%. Robustness = amplify a sub-100% per-question ceiling; immune to truncation
(short Y/N output); precisely tunable via battery length N; natural headroom (a 99.5%/q future
model scores 0.995^N).

## The new design: suite of all-or-nothing causal-inference batteries (fair 40k budget)

Measured per-QUESTION accuracy on Opus 4.8 + thinking @ 40k (K=30 per battery, 6 items each),
then N for <55% = ceil(ln .55 / ln p). ALL had 0 truncation.

| Test | atomic judgment | per-q | N for <55% | notes |
|---|---|---|---|---|
| FRONTDOOR | is Z a valid front-door set for X->Y? | 96.1% | ~16 | STRONGEST (short battery) |
| DSEP | X ⊥ Y \| Z ? | 98.2% | ~50 | proven anchor |
| BACKDOOR | is Z a valid backdoor adjustment set? | 98.3% | ~40 | good |
| CONDCHANGE | does adding W to Z flip X⊥Y? | 98.9% | ~54 | marginal - long battery |
| IV | is V a valid instrument for X->Y? | 98.9% | ~54 | marginal - long battery |

DEAD (per-q = 100%, model is perfect -> infeasible): PATHACT (single-path activity), INTERV
(one do-mutilation then d-sep), MINSEP (minimal separator - small subset check), MECEQUIV
(couldn't even build a yes/no mix). Dead ones share a trait: a bounded/local check. Viable ones
force reasoning over ALL paths at once.

**Design tension:** per-q near 100% -> long battery -> the THINKING may truncate at 40k, which
would reintroduce the artifact. So HARDEN CONDCHANGE/IV with denser/bigger graphs to pull per-q
down to ~97% (shorter battery). FRONTDOOR is the model of what we want.

## Gold formulas (all networkx, verified in probes)
- d-sep: `nx.is_d_separator(G, {x}, {y}, set(Z))`
- backdoor set valid: no descendant of X (nor Y) in Z, AND is_d_separator(G_minus_outX, X, Y, Z)
- front-door(Z for X->Y): (1) removing Z leaves no directed X->Y path; (2) is_d_separator(G_minus_outX, X, Z, ∅); (3) is_d_separator(G_minus_outZ, Z, Y, {X})
- instrument(V for X->Y): NOT d-sep(G, V, X, ∅) [relevance] AND is_d_separator(G_minus_outX, V, Y, ∅) [exclusion]
- condchange(x,y,Z,w): d_separated(G,x,y,Z) != d_separated(G,x,y,Z∪{w})
- G_minus_outS = copy G, remove all out-edges of every node in S.

## Build plan (NEXT - awaiting user go-ahead as of 2026-07-06)
1. Replace the 5 counting builders in `benchpress/modules/frontier/__init__.py` with 5 battery
   builders (DSEP kept; add BACKDOOR, FRONTDOOR, CONDCHANGE, IV). Each item = one battery of N
   Y/N sub-questions, scored all-or-nothing (conjunctive) - the existing part-scoring already
   supports this (DSEP does it).
2. Tune each N to target ~40% pass (not just <55%): N = ln(0.40)/ln(p). Harden CONDCHANGE/IV
   graphs so per-q ~97% -> shorter N -> no thinking-truncation.
3. Freeze spec: **40k max_tokens, thinking-on (adaptive, effort high), tools-off**.
4. 25-item Opus confirm at 40k; then freeze v1 (lock seed+config+N per test).
5. Frontier panel (GPT-5/Gemini-3/Grok) later to confirm spread.

## Key rules (locked)
- **Tools-off, thinking-on, FIXED 40k token budget** is the official config. Scores must reflect
  reasoning, not truncation - always verify stop_reason=end_turn, never max_tokens.
- Deterministic: `generate(seed=42, "hard", n=N)` -> identical items. Freeze = lock seed+config.
- Scoring: conjunctive (all sub-questions correct = item correct); per-test = per-bundle marginal;
  bootstrap CIs. No LLM judge, ever.
- Calibrate once, then freeze; re-hardening = a new version, never edit v1.

## Resilience / how to resume
- `runner.run_model` writes atomically (tmp + os.replace) and is resume-by-content: re-running
  skips completed items, retries errored ones. Any run can be safely relaunched.
- Bedrock: 300s read timeout + retry on throttle/timeout. Claude via **Bedrock only** (no API keys).
  Model id `eu.anthropic.claude-opus-4-8`, region eu-central-1, account 996083107598. Creds in `.env`.
- Probe scripts (reproducible) under the session scratchpad: probe_atomic.py / probe_atomic2.py /
  probe_atomic3.py (per-q measurement), verify_tokens.py / verify_dsep.py (the 40k re-verify that
  exposed the truncation artifact).
