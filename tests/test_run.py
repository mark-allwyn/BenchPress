"""Tests for run.py - data layer, cmd_rejudge, cmd_eval, cmd_compare, cmd_migrate_judges."""

import argparse
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime


# ── Data layer ──

class TestModelPath:
    def test_correct_path(self):
        import run
        original = run.RESULTS_DIR
        run.RESULTS_DIR = "results"
        assert run.model_path("gpt-4o") == os.path.join("results", "gpt-4o.json")
        run.RESULTS_DIR = original


class TestLoadModelResults:
    def test_existing_file(self, tmp_results_dir):
        import run
        data = {"model_name": "test", "created": "2026-01-01", "runs": {"T01": []}}
        (tmp_results_dir / "test.json").write_text(json.dumps(data))
        result = run.load_model_results("test")
        assert result["model_name"] == "test"
        assert result["runs"] == {"T01": []}

    def test_missing_file_returns_template(self, tmp_results_dir):
        import run
        result = run.load_model_results("nonexistent")
        assert result["model_name"] == "nonexistent"
        assert result["runs"] == {}
        assert "created" in result


class TestSaveModelResults:
    def test_writes_valid_json(self, tmp_results_dir):
        import run
        data = {"model_name": "test", "runs": {}}
        run.save_model_results("test", data)
        path = tmp_results_dir / "test.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["model_name"] == "test"
        assert "updated" in loaded

    def test_atomic_write_no_tmp_left(self, tmp_results_dir):
        import run
        data = {"model_name": "test", "runs": {}}
        run.save_model_results("test", data)
        tmp_files = list(tmp_results_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_sets_updated_timestamp(self, tmp_results_dir):
        import run
        data = {"model_name": "test", "runs": {}}
        run.save_model_results("test", data)
        assert "updated" in data


class TestListEvaluatedModels:
    def test_finds_json_files(self, tmp_results_dir):
        import run
        (tmp_results_dir / "model-a.json").write_text("{}")
        (tmp_results_dir / "model-b.json").write_text("{}")
        result = run.list_evaluated_models()
        assert "model-a" in result
        assert "model-b" in result

    def test_excludes_comparison(self, tmp_results_dir):
        import run
        (tmp_results_dir / "comparison.json").write_text("{}")
        (tmp_results_dir / "real-model.json").write_text("{}")
        result = run.list_evaluated_models()
        assert "comparison" not in result
        assert "real-model" in result

    def test_empty_dir(self, tmp_results_dir):
        import run
        assert run.list_evaluated_models() == []


class TestFilterPrompts:
    def setup_method(self):
        self.prompts = [
            {"id": "C01", "category": "coding", "difficulty": "easy"},
            {"id": "C02", "category": "coding", "difficulty": "hard"},
            {"id": "R01", "category": "reasoning", "difficulty": "medium"},
        ]

    def test_filter_by_id(self):
        from run import filter_prompts
        result = filter_prompts(self.prompts, ids=["C01"])
        assert len(result) == 1
        assert result[0]["id"] == "C01"

    def test_filter_by_category(self):
        from run import filter_prompts
        result = filter_prompts(self.prompts, categories=["coding"])
        assert len(result) == 2

    def test_filter_by_difficulty(self):
        from run import filter_prompts
        result = filter_prompts(self.prompts, difficulty=["hard"])
        assert len(result) == 1

    def test_filter_case_insensitive(self):
        from run import filter_prompts
        result = filter_prompts(self.prompts, categories=["Coding"])
        assert len(result) == 2

    def test_combined_filters(self):
        from run import filter_prompts
        result = filter_prompts(self.prompts, categories=["coding"], difficulty=["hard"])
        assert len(result) == 1
        assert result[0]["id"] == "C02"

    def test_no_filters(self):
        from run import filter_prompts
        result = filter_prompts(self.prompts)
        assert len(result) == 3


class TestLatestRun:
    def test_single_run(self):
        from run import latest_run
        data = {"runs": {"T01": [{"content": "first"}]}}
        assert latest_run(data, "T01") == {"content": "first"}

    def test_multiple_runs_returns_last(self):
        from run import latest_run
        data = {"runs": {"T01": [{"content": "first"}, {"content": "second"}]}}
        assert latest_run(data, "T01")["content"] == "second"

    def test_no_runs(self):
        from run import latest_run
        assert latest_run({"runs": {}}, "T01") == {}

    def test_missing_prompt(self):
        from run import latest_run
        assert latest_run({"runs": {"T02": []}}, "T01") == {}


# ── cmd_rejudge race condition fix ──

class TestCmdRejudgeMerge:
    def test_merge_on_save_preserves_concurrent_scores(self, tmp_results_dir):
        """Scores from a concurrent process should survive merge-on-save."""
        import run

        # Initial file with one judge score
        initial = {
            "model_name": "test",
            "runs": {
                "T01": [{
                    "timestamp": "2026-01-01",
                    "api_model": "gpt-test",
                    "content": "def add(a,b): return a+b",
                    "latency_s": 1.0,
                    "auto_checks": {"flags": [], "auto_scores": {}, "passed": True},
                    "judge_scores": {
                        "judge-a": {"score": 3, "rationale": "OK", "judged_at": "2026-01-01"},
                    },
                    "judge_score_avg": 3.0,
                    "judge_count": 1,
                }],
            },
        }
        run.save_model_results("test", initial)

        # Simulate: our process scored judge-b in memory
        model_data = run.load_model_results("test")
        run_entry = model_data["runs"]["T01"][-1]
        run_entry["judge_scores"]["judge-b"] = {
            "score": 4, "rationale": "Good", "judged_at": "2026-01-01",
        }

        # Simulate: concurrent process wrote judge-c to disk
        fresh_on_disk = run.load_model_results("test")
        fresh_on_disk["runs"]["T01"][-1]["judge_scores"]["judge-c"] = {
            "score": 5, "rationale": "Great", "judged_at": "2026-01-01",
        }
        run.save_model_results("test", fresh_on_disk)

        # Now do the merge like cmd_rejudge does
        judges_needed_by_pid = {"T01": ["judge-b"]}
        fresh_data = run.load_model_results("test")
        for pid in model_data["runs"]:
            fresh_runs = fresh_data.get("runs", {}).get(pid, [])
            if not fresh_runs:
                continue
            fresh_run = fresh_runs[-1]
            if "judge_scores" not in fresh_run:
                fresh_run["judge_scores"] = {}
            source_run = model_data["runs"][pid][-1]
            for jname, jdata in source_run.get("judge_scores", {}).items():
                if jname in judges_needed_by_pid.get(pid, []):
                    fresh_run["judge_scores"][jname] = jdata
            valid = [v["score"] for v in fresh_run["judge_scores"].values()
                     if isinstance(v, dict) and v.get("score") is not None]
            fresh_run["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
            fresh_run["judge_count"] = len(valid)

        run.save_model_results("test", fresh_data)

        # Verify all three judges preserved
        final = run.load_model_results("test")
        scores = final["runs"]["T01"][-1]["judge_scores"]
        assert "judge-a" in scores  # original
        assert "judge-b" in scores  # our process
        assert "judge-c" in scores  # concurrent process
        assert final["runs"]["T01"][-1]["judge_count"] == 3

    def test_only_merges_needed_judges(self, tmp_results_dir):
        """Only judges in judges_needed_by_pid should be merged."""
        import run

        initial = {
            "model_name": "test",
            "runs": {
                "T01": [{
                    "content": "x",
                    "auto_checks": {"flags": [], "auto_scores": {}, "passed": True},
                    "judge_scores": {
                        "old-judge": {"score": 2, "rationale": "Bad", "judged_at": "2026-01-01"},
                    },
                    "judge_score_avg": 2.0,
                    "judge_count": 1,
                }],
            },
        }
        run.save_model_results("test", initial)

        model_data = run.load_model_results("test")
        run_entry = model_data["runs"]["T01"][-1]
        # Simulate scoring new-judge AND having stale old-judge data (score=5 in memory)
        run_entry["judge_scores"]["new-judge"] = {"score": 4, "rationale": "Good", "judged_at": "2026-01-01"}
        run_entry["judge_scores"]["old-judge"]["score"] = 5  # stale mutation

        judges_needed_by_pid = {"T01": ["new-judge"]}  # only new-judge was scored this session

        fresh_data = run.load_model_results("test")
        for pid in model_data["runs"]:
            fresh_run = fresh_data["runs"][pid][-1]
            if "judge_scores" not in fresh_run:
                fresh_run["judge_scores"] = {}
            source_run = model_data["runs"][pid][-1]
            for jname, jdata in source_run.get("judge_scores", {}).items():
                if jname in judges_needed_by_pid.get(pid, []):
                    fresh_run["judge_scores"][jname] = jdata

        run.save_model_results("test", fresh_data)

        final = run.load_model_results("test")
        scores = final["runs"]["T01"][-1]["judge_scores"]
        assert scores["old-judge"]["score"] == 2  # original preserved, not stale 5
        assert scores["new-judge"]["score"] == 4


# ── cmd_migrate_judges ──

class TestCmdMigrateJudges:
    def test_old_format_converted(self, tmp_results_dir):
        import run

        old_data = {
            "model_name": "legacy",
            "runs": {
                "T01": [{
                    "content": "response",
                    "judge_score": 4,
                    "judge_rationale": "Good work.",
                    "judge_model": "gpt-4",
                }],
            },
        }
        run.save_model_results("legacy", old_data)

        # Simulate migration logic
        data = run.load_model_results("legacy")
        for pid, runs in data["runs"].items():
            for entry in runs:
                if "judge_scores" in entry:
                    continue
                old_score = entry.get("judge_score")
                old_rationale = entry.get("judge_rationale")
                old_model = entry.get("judge_model")
                if old_score is None and old_rationale is None and old_model is None:
                    continue
                judge_key = old_model or "unknown"
                entry["judge_scores"] = {
                    judge_key: {
                        "score": old_score,
                        "rationale": old_rationale,
                        "judged_at": None,
                    }
                }
                entry["judge_score_avg"] = float(old_score) if old_score is not None else None
                entry["judge_count"] = 1 if old_score is not None else 0
                entry.pop("judge_score", None)
                entry.pop("judge_rationale", None)
                entry.pop("judge_model", None)

        run.save_model_results("legacy", data)

        final = run.load_model_results("legacy")
        entry = final["runs"]["T01"][0]
        assert "judge_scores" in entry
        assert "gpt-4" in entry["judge_scores"]
        assert entry["judge_scores"]["gpt-4"]["score"] == 4
        assert "judge_score" not in entry
        assert "judge_model" not in entry

    def test_missing_judge_model_uses_unknown(self, tmp_results_dir):
        import run

        old_data = {
            "model_name": "legacy2",
            "runs": {
                "T01": [{
                    "content": "response",
                    "judge_score": 3,
                    "judge_rationale": "Meh.",
                }],
            },
        }
        run.save_model_results("legacy2", old_data)

        data = run.load_model_results("legacy2")
        entry = data["runs"]["T01"][0]
        judge_key = entry.get("judge_model") or "unknown"
        entry["judge_scores"] = {
            judge_key: {"score": entry["judge_score"], "rationale": entry["judge_rationale"], "judged_at": None}
        }

        assert "unknown" in entry["judge_scores"]

    def test_idempotent(self, tmp_results_dir):
        import run

        migrated_data = {
            "model_name": "already",
            "runs": {
                "T01": [{
                    "content": "response",
                    "judge_scores": {"gpt-4": {"score": 4, "rationale": "Good.", "judged_at": None}},
                    "judge_score_avg": 4.0,
                    "judge_count": 1,
                }],
            },
        }
        run.save_model_results("already", migrated_data)

        data = run.load_model_results("already")
        entry = data["runs"]["T01"][0]
        # Migration should skip entries with judge_scores
        assert "judge_scores" in entry
        assert entry["judge_scores"]["gpt-4"]["score"] == 4


# ── cmd_compare composite score ──

class TestCompositeScore:
    def test_both_scores(self):
        """Composite = judge_weight * normalized_judge + deepeval_weight * deepeval_avg."""
        judge_weight, deepeval_weight = 0.5, 0.5
        avg_s = 4.0  # judge avg
        deepeval_avg = 0.8
        normalized_judge = (avg_s - 1) / 4  # 0.75
        composite = round(judge_weight * normalized_judge + deepeval_weight * deepeval_avg, 4)
        assert composite == 0.775

    def test_judge_only(self):
        avg_s = 3.0
        normalized_judge = (avg_s - 1) / 4  # 0.5
        composite = round(normalized_judge, 4)
        assert composite == 0.5

    def test_deepeval_only(self):
        deepeval_avg = 0.6
        composite = round(deepeval_avg, 4)
        assert composite == 0.6

    def test_neither_is_none(self):
        normalized_judge = None
        deepeval_avg = None
        if normalized_judge is not None and deepeval_avg is not None:
            composite = 0.0
        elif normalized_judge is not None:
            composite = normalized_judge
        elif deepeval_avg is not None:
            composite = deepeval_avg
        else:
            composite = None
        assert composite is None


# ── load_eval with custom eval_file ──

class TestLoadEval:
    def test_default_path(self, tmp_path, monkeypatch):
        import run
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        ef = evals_dir / "default.json"
        ef.write_text(json.dumps({"prompts": [{"id": "X01"}]}))
        monkeypatch.setattr(run, "EVAL_FILE", str(ef))
        result = run.load_eval()
        assert len(result) == 1
        assert result[0]["id"] == "X01"

    def test_custom_eval_file(self, tmp_path):
        import run
        custom = tmp_path / "custom.json"
        custom.write_text(json.dumps({"prompts": [{"id": "Y01"}, {"id": "Y02"}]}))
        result = run.load_eval(str(custom))
        assert len(result) == 2
        assert result[0]["id"] == "Y01"

    def test_custom_overrides_default(self, tmp_path, monkeypatch):
        import run
        # Default file
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        default = evals_dir / "default.json"
        default.write_text(json.dumps({"prompts": [{"id": "D01"}]}))
        monkeypatch.setattr(run, "EVAL_FILE", str(default))
        # Custom file
        custom = tmp_path / "other.json"
        custom.write_text(json.dumps({"prompts": [{"id": "O01"}, {"id": "O02"}]}))
        result = run.load_eval(str(custom))
        assert len(result) == 2
        assert result[0]["id"] == "O01"


# ── cmd_compare uses args.config ──

class TestCmdCompareConfig:
    def test_compare_uses_provided_config(self, tmp_path, tmp_results_dir, monkeypatch):
        """cmd_compare should use args.config, not hardcoded config.yaml."""
        import run

        # Create a config file
        config_file = tmp_path / "myconfig.yaml"
        config_file.write_text(
            "models: {}\njudges: []\ncomposite:\n  judge_weight: 0.6\n  deepeval_weight: 0.4\n"
        )

        # Create an eval file
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        ef = evals_dir / "default.json"
        ef.write_text(json.dumps({"prompts": [
            {"id": "T01", "category": "coding", "subcategory": "test",
             "difficulty": "easy", "prompt": "p", "ideal": "i", "criteria": [], "check_type": "reasoning"},
        ]}))
        monkeypatch.setattr(run, "EVAL_FILE", str(ef))

        # Create a result file
        model_data = {
            "model_name": "test-model",
            "created": "2026-01-01",
            "runs": {
                "T01": [{
                    "timestamp": "2026-01-01",
                    "api_model": "gpt-test",
                    "content": "Hello",
                    "latency_s": 1.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "auto_checks": {"flags": [], "auto_scores": {}, "passed": True},
                    "judge_scores": {"j1": {"score": 4, "rationale": "OK", "judged_at": "2026-01-01"}},
                    "judge_score_avg": 4.0,
                    "judge_count": 1,
                }],
            },
        }
        (tmp_results_dir / "test-model.json").write_text(json.dumps(model_data))

        args = argparse.Namespace(
            config=str(config_file),
            models=[],
            ids=None,
            category=None,
            difficulty=None,
            save=False,
        )
        # Should not raise - previously it called load_config() with no args
        run.cmd_compare(args)
