# Multi-Judge Scoring Design

## Problem

Single-judge evaluation (GPT-4.1) introduces bias. Different LLM judges have different scoring tendencies. Using multiple judges and averaging scores produces a more balanced evaluation.

## Design Decisions

- Equal-weight averaging across all judges (no weighted scoring)
- Clean break from old `judge:` config key - replaced entirely by `judges:` array
- Self-exclusion: a model cannot judge itself, average computed from available judges only
- Configurable judge list starting with GPT-4.1, Claude Sonnet 4, Gemini 2.5 Pro

---

## 1. Config Format

Replace `judge:` with `judges:` array in config.yaml:

```yaml
judges:
  - model: gpt-4.1
    params: { max_tokens: 2000, temperature: 0.0 }
  - model: claude-sonnet-4
    params: { max_tokens: 2000, temperature: 0.0 }
  - model: gemini-2.5-pro
    params: { max_tokens: 2000, temperature: 0.0 }
```

Each judge model must exist in the `models:` section. No fallback to old `judge:` key - all config files updated.

## 2. Results Data Model

Old per-run fields (`judge_score`, `judge_rationale`, `judge_model`) replaced with:

```json
{
  "judge_scores": {
    "gpt-4.1": {
      "score": 4,
      "rationale": "...",
      "judged_at": "2026-03-06T14:22:01"
    },
    "claude-sonnet-4": {
      "score": 3,
      "rationale": "...",
      "judged_at": "2026-03-07T09:15:33"
    }
  },
  "judge_score_avg": 3.5,
  "judge_count": 2
}
```

- `judge_scores` - dict keyed by judge model name
- `judge_score_avg` - simple mean of available scores
- `judge_count` - number of judges that scored (transparency for self-exclusion)
- Each judge entry includes `judged_at` timestamp

## 3. Eval Workflow

**`eval <model>`:**
1. For each prompt: call model, run auto-checks
2. Loop through all configured judges
3. Skip any judge whose model name matches the evaluated model (print skip message)
4. If a judge call fails, store `null` score for that judge, continue with others
5. Compute `judge_score_avg` and `judge_count` from successful scores

**`rejudge <model>`:**
- No `--judge` flag: re-runs ALL configured judges
- `--judge <name>`: re-runs only that specific judge
- Merges updated scores into existing `judge_scores` dict on latest run
- Recomputes `judge_score_avg` and `judge_count`

**Self-exclusion:**
- Compare judge model config name against evaluated model config name
- Print: `Skipping judge gpt-4.1 (cannot self-judge)`
- `judge_count` reflects actual judges that scored

## 4. Composite Score

No change to composite config. Updated formula uses `judge_score_avg`:

```
normalized_judge = (judge_score_avg - 1) / 4
composite = judge_weight * normalized_judge + deepeval_weight * deepeval_avg
```

## 5. Dashboard

Keep the main leaderboard table clean - same width as today:

- **Judge Score column** shows `judge_score_avg` with judge count on hover tooltip
- **Agreement indicator** - small colored dot next to score (green = judges agree, yellow/red = disagreement based on std dev)
- **Expandable row** - click a model row to see per-judge scores, rationales, and agreement metric
- **Per-prompt detail** - tabbed/stacked view showing each judge's score and rationale (not side-by-side columns)

## 6. Migration

One-time command `python run.py migrate-judges`:

1. Read each `results/*.json` file
2. Convert old fields to new format:
   - `judge_score` + `judge_rationale` + `judge_model` becomes entry in `judge_scores` dict
   - `judged_at` set to `null` for migrated entries
   - Compute `judge_score_avg` and `judge_count`
3. Remove old `judge_score`, `judge_rationale`, `judge_model` fields
4. Back up originals to `results/backup/` before modifying

After migration, run `rejudge` with new judges to fill in additional scores.

## Files Affected

- `config.yaml` / `config.example.yaml` - new `judges:` format
- `run.py` - eval loop, rejudge command, compare logic, new migrate command
- `scripts/judge.py` - multi-judge loop, self-exclusion
- `scripts/dashboard.py` - expandable rows, agreement indicators, per-judge detail
- `results/*.json` - migrated to new schema
