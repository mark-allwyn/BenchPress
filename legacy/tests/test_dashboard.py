"""Tests for scripts/dashboard.py - compute_stats and dashboard generation."""

import json
import pytest
from scripts.dashboard import compute_stats, latest_run


def _make_run(judge_scores=None, deepeval_scores=None, error=None, flags=None):
    """Helper to build a run entry for testing."""
    run = {
        "timestamp": "2026-01-01T00:00:00",
        "api_model": "test-model",
        "content": "Test response content here.",
        "latency_s": 1.0,
        "input_tokens": 50,
        "output_tokens": 30,
        "auto_checks": {
            "flags": flags or [],
            "auto_scores": {},
            "passed": not bool(flags),
        },
        "judge_scores": judge_scores or {},
        "judge_score_avg": None,
        "judge_count": 0,
    }
    if error:
        run["error"] = error
    if judge_scores:
        valid = [v["score"] for v in judge_scores.values() if v.get("score") is not None]
        run["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
        run["judge_count"] = len(valid)
    if deepeval_scores:
        run["deepeval_scores"] = deepeval_scores
        vals = [v for v in deepeval_scores.values() if v is not None]
        run["deepeval_avg"] = round(sum(vals) / len(vals), 2) if vals else None
    return run


def _make_judge(score, rationale="OK"):
    return {"score": score, "rationale": rationale, "judged_at": "2026-01-01"}


@pytest.fixture
def basic_prompts():
    return [
        {"id": "C01", "category": "coding", "subcategory": "basics", "difficulty": "easy",
         "prompt": "p", "ideal": "i", "criteria": [], "check_type": "code_runnable"},
        {"id": "R01", "category": "reasoning", "subcategory": "logic", "difficulty": "hard",
         "prompt": "p", "ideal": "i", "criteria": [], "check_type": "reasoning"},
    ]


class TestLatestRun:
    def test_returns_last(self):
        data = {"runs": {"C01": [{"a": 1}, {"a": 2}]}}
        assert latest_run(data, "C01")["a"] == 2

    def test_empty(self):
        assert latest_run({"runs": {}}, "C01") == {}


class TestComputeStats:
    def test_basic_aggregation(self, basic_prompts):
        models = {
            "model-a": {
                "runs": {
                    "C01": [_make_run(judge_scores={"j1": _make_judge(4), "j2": _make_judge(5)})],
                    "R01": [_make_run(judge_scores={"j1": _make_judge(3), "j2": _make_judge(4)})],
                },
            },
        }
        stats = compute_stats(models, basic_prompts, judge_models=["j1", "j2"])
        lb = stats["leaderboard"]
        assert len(lb) == 1
        assert lb[0]["name"] == "model-a"
        assert lb[0]["avg_score"] > 0
        assert lb[0]["total"] == 2

    def test_only_complete_judges_counted(self, basic_prompts):
        """A judge that scored only 1 of 2 prompts should be excluded from avg."""
        models = {
            "model-a": {
                "runs": {
                    "C01": [_make_run(judge_scores={
                        "complete-j": _make_judge(5),
                        "partial-j": _make_judge(2),
                    })],
                    "R01": [_make_run(judge_scores={
                        "complete-j": _make_judge(4),
                        # partial-j did NOT score this prompt
                    })],
                },
            },
        }
        stats = compute_stats(models, basic_prompts)
        lb = stats["leaderboard"]
        entry = lb[0]
        # avg_score should be based only on complete-j (4.5), not partial-j
        assert entry["avg_score"] == 4.5

    def test_handles_partial_data(self, basic_prompts):
        """Model with only one prompt scored."""
        models = {
            "model-a": {
                "runs": {
                    "C01": [_make_run(judge_scores={"j1": _make_judge(4)})],
                },
            },
        }
        stats = compute_stats(models, basic_prompts)
        lb = stats["leaderboard"]
        assert len(lb) == 1

    def test_handles_error_runs(self, basic_prompts):
        models = {
            "model-a": {
                "runs": {
                    "C01": [_make_run(error="API timeout")],
                    "R01": [_make_run(judge_scores={"j1": _make_judge(4)})],
                },
            },
        }
        stats = compute_stats(models, basic_prompts)
        lb = stats["leaderboard"]
        assert lb[0]["errors"] == 1

    def test_category_breakdown(self, basic_prompts):
        models = {
            "model-a": {
                "runs": {
                    "C01": [_make_run(judge_scores={"j1": _make_judge(5)})],
                    "R01": [_make_run(judge_scores={"j1": _make_judge(3)})],
                },
            },
        }
        stats = compute_stats(models, basic_prompts)
        lb = stats["leaderboard"]
        cats = lb[0]["cat_scores"]
        assert "coding" in cats
        assert "reasoning" in cats

    def test_composite_score(self, basic_prompts):
        models = {
            "model-a": {
                "runs": {
                    "C01": [_make_run(
                        judge_scores={"j1": _make_judge(4)},
                        deepeval_scores={"correctness": 0.8, "coherence": 0.9, "instruction_following": 0.85},
                    )],
                    "R01": [_make_run(
                        judge_scores={"j1": _make_judge(4)},
                        deepeval_scores={"correctness": 0.7, "coherence": 0.8, "instruction_following": 0.75},
                    )],
                },
            },
        }
        comp_cfg = {"judge_weight": 0.5, "deepeval_weight": 0.5}
        stats = compute_stats(models, basic_prompts, composite_config=comp_cfg)
        lb = stats["leaderboard"]
        assert lb[0]["composite_score"] is not None
        assert 0 < lb[0]["composite_score"] < 1

    def test_empty_models(self, basic_prompts):
        stats = compute_stats({}, basic_prompts)
        assert stats["leaderboard"] == []

    def test_multiple_models_sorted(self, basic_prompts):
        models = {
            "low-model": {
                "runs": {
                    "C01": [_make_run(judge_scores={"j1": _make_judge(2)})],
                    "R01": [_make_run(judge_scores={"j1": _make_judge(2)})],
                },
            },
            "high-model": {
                "runs": {
                    "C01": [_make_run(judge_scores={"j1": _make_judge(5)})],
                    "R01": [_make_run(judge_scores={"j1": _make_judge(5)})],
                },
            },
        }
        stats = compute_stats(models, basic_prompts)
        lb = stats["leaderboard"]
        # Should be sorted by composite/score descending
        assert lb[0]["name"] == "high-model"


class TestLeaderboardRowEscaping:
    """Model names and companies with HTML special chars must be escaped."""

    def test_html_special_chars_escaped(self, basic_prompts):
        from scripts.dashboard import _leaderboard_row
        model_entry = {
            "name": '<script>alert("xss")</script>',
            "company": 'Foo & "Bar"',
            "avg_score": 3.5,
            "composite_score": 0.6,
            "deepeval_avg": 0.7,
            "scored": 2,
            "de_scored": 2,
            "total": 2,
            "errors": 0,
            "flagged": 0,
            "avg_latency": 1.0,
            "avg_tokens": 100,
            "efficiency": 0.5,
            "avg_divergence": None,
            "judge_std_dev": None,
            "judge_averages": {},
            "cat_scores": {},
        }
        html = _leaderboard_row(0, model_entry)
        # Raw script tag should NOT appear
        assert "<script>" not in html
        # Escaped versions should appear
        assert "&lt;script&gt;" in html
        assert "&amp;" in html
        assert "&quot;" in html
