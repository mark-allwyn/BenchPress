# Benchpress — current status & resume guide

_Last updated: 2026-06-25. Read this first to continue the work._

## What Benchpress is now
A **frontier-only, causal-structure** benchmark. Frozen-tier candidate = **5 tests**, all
verifiable by networkx (no judge), run **tools-off**, **thinking-on** (the official config).
Identity: *exhaustive causal-graph reasoning under load* — the only thing that still caps a
thinking frontier model is producing an exact count too large to enumerate in working memory.

Code: `benchpress/` (committed). The 5 tests live in one registered module, `frontier`
(`benchpress/modules/frontier/`), each test = a bundle; gold in `graphs.py`. 203 tests pass.

## The 5 tests + Opus 4.8 (thinking on) scores — small-sample (8 items)
| Test (bundle) | Opus+think | Knobs (in `frontier/__init__.py` CONFIG) |
|---|---|---|
| LINEXT (count topological orderings) | 0% | 10 nodes, density 0.30, count 1000-50000 |
| OPENPATH (count active paths given a set) | 0% | 10 nodes, density 0.44, ≥25 paths |
| DSEP (50-question d-separation battery, conjunctive) | 25% | 22 nodes, density 0.19, 50 queries |
| MINSEP_COUNT (count minimal separating sets) | 25% | 10 nodes, density 0.32, ≥3 sets |
| VSTRUCT (count colliders) | 50% | 24 nodes, density 0.40, ≥80 colliders |
Overall ≈ 20%. VSTRUCT is the borderline/spread test.

Dropped during calibration (thinking solved them): COMPELLED, MINSEP_SIZE, MEC.

## Key design rules (locked)
- **Tools-off, thinking-on** is the official run config. Code execution makes it trivial.
- **Deterministic**: `generate(seed=42, "hard", n=N)` → identical items every run. Freeze = lock seed+config.
- Scoring: numeric_tolerance(tol 0.4) for counts, categorical for d-sep; item correct = all parts (conjunctive). Per-test = per-bundle marginal.
- Difficulty is calibrated **once** then frozen; re-hardening = a new version, never edit v1.

## Resilience / how to resume (THIS IS THE IMPORTANT PART)
- **Results are crash-safe.** `runner.run_model` writes atomically (tmp + os.replace) and is
  **resume-by-content**: re-running skips completed items and retries errored ones. So any run
  can be safely re-launched and it continues from where it stopped — no work lost.
- **Provider hardening**: Bedrock has 300s read timeout + retry on throttle/timeout; OpenAI/
  Anthropic retry 429 with backoff.

### In-flight run (as of last update)
`scratchpad/confirm25.py` — Opus 4.8 + thinking, all 5 tests at **25 items each**, writing to
`results/frontier/opus-think-25.json`. **To resume if interrupted: just re-run that script** —
completed items skip. (Reproduce: `generate(42,"hard",n=25)` + BedrockProvider thinking=adaptive.)

## Next steps (in order)
1. Finish `confirm25` → tight per-test CIs on Opus (esp. is VSTRUCT really <55%?).
2. **Frontier panel run**: GPT-5 (OpenAI), Gemini-3 (Google), Grok (xAI), Opus — ~25 items,
   thinking-on — to confirm the field SPREADS (does it discriminate?). This is the make-or-break
   for it being a ranking benchmark vs a "nobody's solved it" flag.
3. If spread is good → **freeze Benchpress-v1** (lock seed+config in a manifest; held-out).
4. Refresh dashboard/leaderboard later (deferred).

## Run commands
- Generate/inspect: `python -c "from benchpress.modules.frontier import generate; ..."`
- Bedrock Claude id used: `eu.anthropic.claude-opus-4-8` (region eu-central-1, account 996083107598).
- Model keys live in `.env` (anthropic/openai/google/cohere/mistral/xai + AWS). `.env`/`results/` gitignored.
- Scratch run scripts under the session scratchpad (reproducible from this doc if lost).
