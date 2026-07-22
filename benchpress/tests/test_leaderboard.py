"""Score-only leaderboard export: correct shape, no answer leakage, baseline
anchor, and real runs superseding the baseline row."""

import json

import benchpress.modules.simulate  # noqa: F401  (register module)
import benchpress.scorers  # noqa: F401  (register part-scorers)
from benchpress.core import registry
from benchpress.frozen import load_frozen
from benchpress.leaderboard import build_leaderboard
from benchpress.runner import persist, run_model, score_model
from benchpress.providers.base import CompletionResult


def _frozen_items():
    cfg = load_frozen("simulate")["official_run_config"]
    return registry.get_module("simulate")(cfg["seed"], "hard", cfg["n_per_bundle"])[0]


def test_baseline_only_when_no_results(tmp_path):
    items = _frozen_items()
    board = build_leaderboard("simulate", items, [], {}, now="2026-07-16")
    assert board["benchmark"] == "Benchpress-Simulate"
    assert board["version"] == "v2"
    assert board["tasks"] == ["LIFE", "DAYNIGHT", "ECA110", "ECA30"]
    assert board["config"]["max_tokens"] == 96000
    assert len(board["models"]) == 1
    base = board["models"][0]
    assert base["baseline"] is True and base["name"] == "claude-opus-4.8"
    assert base["tasks"]["LIFE"]["exact_pct"] == 32


class _GoldProvider:
    def __init__(self, items):
        self._by_prompt = {
            it.prompt: "\n".join(f"{p.part_id}: {p.expected}" for p in it.parts)
            for it in items
        }

    def complete(self, prompt):
        return CompletionResult(content=self._by_prompt[prompt], stop_reason="end_turn")


def test_real_run_appears_and_no_answers_leak(tmp_path):
    items = _frozen_items()
    path = tmp_path / "gpt-x.json"
    run_model(_GoldProvider(items), items, path, model_name="gpt-x",
              benchmark="simulate", version="1")
    score_model(items, path)

    meta = {"gpt-x": {"company": "OpenAI", "type": "closed", "launch_date": "2026-01-01"}}
    board = build_leaderboard("simulate", items, [path], meta, now="2026-07-16")

    names = [m["name"] for m in board["models"]]
    assert "gpt-x" in names
    gpt = next(m for m in board["models"] if m["name"] == "gpt-x")
    assert gpt["company"] == "OpenAI"
    assert gpt["overall"]["exact_pct"] == 100.0        # gold answers
    assert gpt["tasks"]["LIFE"]["exact_pct"] == 100.0

    # No prompt / gold / raw-output content anywhere in the payload. (Public task
    # names and rules in task_meta are fine - the model gets those in-prompt; only
    # initial states, gold final states, and raw outputs must never appear.)
    blob = json.dumps(board)
    assert "ROW1:" not in blob          # no answer-format markers / model output
    assert "Initial grid" not in blob   # no rendered prompt / initial state
    for it in items[:5]:
        assert it.prompt not in blob
        # ECA gold rows are 30-digit strings - unambiguous leak canaries.
        for p in it.parts:
            if isinstance(p.expected, str) and len(p.expected) >= 20:
                assert p.expected not in blob


def test_real_run_supersedes_baseline_for_same_model(tmp_path):
    items = _frozen_items()
    path = tmp_path / "claude-opus-4.8.json"
    run_model(_GoldProvider(items), items, path, model_name="claude-opus-4.8",
              benchmark="simulate", version="1")
    score_model(items, path)
    board = build_leaderboard("simulate", items, [path], {}, now="2026-07-16")
    opus_rows = [m for m in board["models"] if m["name"] == "claude-opus-4.8"]
    assert len(opus_rows) == 1                          # baseline row replaced, not duplicated
    assert opus_rows[0]["baseline"] is False
    assert opus_rows[0]["overall"]["exact_pct"] == 100.0


def test_models_sorted_by_per_row_desc(tmp_path):
    items = _frozen_items()
    # one real (perfect) run + baseline (< 100) -> real run should rank first
    path = tmp_path / "gpt-x.json"
    run_model(_GoldProvider(items), items, path, model_name="gpt-x",
              benchmark="simulate", version="1")
    score_model(items, path)
    board = build_leaderboard("simulate", items, [path], {}, now="2026-07-16")
    prs = [m["overall"]["per_row_pct"] for m in board["models"]]
    assert prs == sorted(prs, reverse=True)
