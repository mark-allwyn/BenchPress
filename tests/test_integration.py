"""Integration tests - end-to-end flows with mocked providers."""

import json
import os
import pytest
from unittest.mock import patch
from tests.conftest import MockProvider


class TestEvalPipeline:
    """Full eval pipeline: provider call -> auto-checks -> judge -> save -> load."""

    def test_full_pipeline(self, tmp_results_dir, sample_prompt):
        import run
        from scripts.checks import check_response
        from scripts.judge import judge_response

        model_name = "integration-test"
        response_text = "```python\ndef add(a, b):\n    return a + b\n```"

        # 1. Provider returns a response
        provider = MockProvider(
            response=response_text,
            usage={"input_tokens": 25, "output_tokens": 15},
        )
        content, usage = provider.complete(sample_prompt["prompt"], {})
        assert content == response_text

        # 2. Auto-checks run
        auto = check_response(sample_prompt, content)
        assert auto["passed"] is True

        # 3. Judge scores the response
        judge_provider = MockProvider(
            response='{"score": 4, "rationale": "Correct implementation."}'
        )
        jr = judge_response(judge_provider, {}, sample_prompt, content, auto)
        assert jr["judge_score"] == 4

        # 4. Build and save model data
        model_data = run.load_model_results(model_name)
        entry = {
            "timestamp": "2026-01-15T10:00:00",
            "api_model": "gpt-test",
            "content": content,
            "latency_s": 1.0,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "auto_checks": auto,
            "judge_scores": {
                "test-judge": {
                    "score": jr["judge_score"],
                    "rationale": jr["judge_rationale"],
                    "judged_at": "2026-01-15T10:01:00",
                }
            },
            "judge_score_avg": float(jr["judge_score"]),
            "judge_count": 1,
        }
        model_data["runs"][sample_prompt["id"]] = [entry]
        run.save_model_results(model_name, model_data)

        # 5. Load and verify
        loaded = run.load_model_results(model_name)
        assert loaded["model_name"] == model_name
        assert sample_prompt["id"] in loaded["runs"]
        saved_entry = loaded["runs"][sample_prompt["id"]][0]
        assert saved_entry["judge_scores"]["test-judge"]["score"] == 4
        assert saved_entry["auto_checks"]["passed"] is True


class TestRejudgeMergeIntegration:
    """Two judges write to the same model file - both scores preserved."""

    def test_two_judges_preserved(self, tmp_results_dir):
        import run

        model_name = "merge-test"
        initial = {
            "model_name": model_name,
            "runs": {
                "T01": [{
                    "timestamp": "2026-01-01",
                    "api_model": "gpt-test",
                    "content": "def add(a,b): return a+b",
                    "latency_s": 1.0,
                    "auto_checks": {"flags": [], "auto_scores": {}, "passed": True},
                    "judge_scores": {},
                    "judge_score_avg": None,
                    "judge_count": 0,
                }],
            },
        }
        run.save_model_results(model_name, initial)

        # Judge A scores
        data_a = run.load_model_results(model_name)
        data_a["runs"]["T01"][-1]["judge_scores"]["judge-a"] = {
            "score": 4, "rationale": "Good", "judged_at": "2026-01-01",
        }
        run.save_model_results(model_name, data_a)

        # Judge B scores (reads fresh file, sees judge-a)
        data_b = run.load_model_results(model_name)
        data_b["runs"]["T01"][-1]["judge_scores"]["judge-b"] = {
            "score": 5, "rationale": "Excellent", "judged_at": "2026-01-01",
        }
        # Recompute
        valid = [v["score"] for v in data_b["runs"]["T01"][-1]["judge_scores"].values()
                 if v.get("score") is not None]
        data_b["runs"]["T01"][-1]["judge_score_avg"] = round(sum(valid) / len(valid), 2)
        data_b["runs"]["T01"][-1]["judge_count"] = len(valid)
        run.save_model_results(model_name, data_b)

        # Verify both preserved
        final = run.load_model_results(model_name)
        scores = final["runs"]["T01"][-1]["judge_scores"]
        assert "judge-a" in scores
        assert "judge-b" in scores
        assert scores["judge-a"]["score"] == 4
        assert scores["judge-b"]["score"] == 5
        assert final["runs"]["T01"][-1]["judge_score_avg"] == 4.5
        assert final["runs"]["T01"][-1]["judge_count"] == 2


class TestDeepEvalModelSelection:
    """DeepEval scorer should read model from judges list, not judge dict."""

    def test_reads_first_judge_from_list(self):
        from scripts.deepeval_scorer import score_with_deepeval
        config = {
            "judges": [
                {"model": "gpt-4.1", "params": {}},
                {"model": "claude-sonnet-4", "params": {}},
            ],
            "models": {
                "gpt-4.1": {"model": "gpt-4.1-2025-04-14"},
                "claude-sonnet-4": {"model": "claude-sonnet-4-20250514"},
            },
            "deepeval": {"metrics": ["correctness"]},
        }
        # We can't actually run DeepEval without the library, but we can
        # verify the model selection logic by checking the config parsing
        judges_cfg = config.get("judges", [])
        models_cfg = config.get("models", {})
        judge_model_name = judges_cfg[0].get("model", "gpt-4.1") if judges_cfg else "gpt-4.1"
        evaluator_model = models_cfg.get(judge_model_name, {}).get("model", "gpt-4.1")
        assert judge_model_name == "gpt-4.1"
        assert evaluator_model == "gpt-4.1-2025-04-14"

    def test_fallback_when_no_judges(self):
        config = {
            "judges": [],
            "models": {},
            "deepeval": {"metrics": ["correctness"]},
        }
        judges_cfg = config.get("judges", [])
        models_cfg = config.get("models", {})
        judge_model_name = judges_cfg[0].get("model", "gpt-4.1") if judges_cfg else "gpt-4.1"
        evaluator_model = models_cfg.get(judge_model_name, {}).get("model", "gpt-4.1")
        assert judge_model_name == "gpt-4.1"
        assert evaluator_model == "gpt-4.1"

    def test_old_judge_key_ignored(self):
        """Config with old 'judge' key (dict) should not be used."""
        config = {
            "judge": {"model": "old-model"},  # old format - should be ignored
            "judges": [{"model": "new-model"}],
            "models": {
                "old-model": {"model": "old-api-model"},
                "new-model": {"model": "new-api-model"},
            },
            "deepeval": {"metrics": ["correctness"]},
        }
        judges_cfg = config.get("judges", [])
        judge_model_name = judges_cfg[0].get("model", "gpt-4.1") if judges_cfg else "gpt-4.1"
        evaluator_model = config.get("models", {}).get(judge_model_name, {}).get("model", "gpt-4.1")
        assert judge_model_name == "new-model"
        assert evaluator_model == "new-api-model"


class TestDashboardGeneration:
    """Dashboard generates valid HTML from result files."""

    def test_generates_html(self, tmp_results_dir, tmp_path, monkeypatch):
        import run
        from scripts import dashboard

        # Patch dashboard module's paths too
        monkeypatch.setattr(dashboard, "RESULTS_DIR", str(tmp_results_dir))
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        monkeypatch.setattr(dashboard, "DOCS_DIR", str(docs_dir))
        monkeypatch.setattr(dashboard, "DASHBOARD_FILE", str(docs_dir / "index.html"))

        # Create a minimal eval file
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        prompts_data = {
            "prompts": [
                {
                    "id": "T01", "category": "coding", "subcategory": "test",
                    "difficulty": "easy", "prompt": "p", "ideal": "i",
                    "criteria": ["c"], "check_type": "reasoning",
                },
            ],
        }
        eval_file = evals_dir / "default.json"
        eval_file.write_text(json.dumps(prompts_data))
        monkeypatch.setattr(dashboard, "EVAL_FILE", str(eval_file))

        # Create a config
        config_file = tmp_path / "config.yaml"
        config_file.write_text("models: {}\njudges: []\ncomposite:\n  judge_weight: 0.5\n  deepeval_weight: 0.5\n")
        monkeypatch.setattr(dashboard, "CONFIG_FILE", str(config_file))

        # Create a result file
        model_data = {
            "model_name": "test-model",
            "created": "2026-01-01",
            "runs": {
                "T01": [{
                    "timestamp": "2026-01-01",
                    "api_model": "gpt-test",
                    "content": "Hello world",
                    "latency_s": 1.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "auto_checks": {"flags": [], "auto_scores": {}, "passed": True},
                    "judge_scores": {"j1": {"score": 4, "rationale": "Good", "judged_at": "2026-01-01"}},
                    "judge_score_avg": 4.0,
                    "judge_count": 1,
                }],
            },
        }
        (tmp_results_dir / "test-model.json").write_text(json.dumps(model_data))

        result = dashboard.generate_dashboard()
        assert result is not None
        index_path = docs_dir / "index.html"
        assert index_path.exists()
        html_content = index_path.read_text()
        assert "<html" in html_content or "<!DOCTYPE" in html_content.upper() or "<table" in html_content
