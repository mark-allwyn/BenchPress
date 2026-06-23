"""Shared fixtures for the eval harness test suite."""

import json
import os
import pytest
from datetime import datetime
from scripts.providers import Provider


class MockProvider(Provider):
    """Test double that returns canned responses without API calls."""

    def __init__(self, response="Mock response", usage=None, error=None):
        self.response = response
        self.usage = usage or {"input_tokens": 10, "output_tokens": 20}
        self.error = error
        self.calls = []

    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        self.calls.append({"prompt": prompt, "params": params})
        if self.error:
            raise self.error
        return self.response, self.usage


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch):
    """Temporary results directory - patches RESULTS_DIR in run.py."""
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr("run.RESULTS_DIR", str(results))
    return results


@pytest.fixture
def sample_config():
    """Minimal valid config dict."""
    return {
        "models": {
            "test-model": {
                "provider": "openai",
                "model": "gpt-test",
                "api_key_env": "none",
                "params": {"temperature": 0, "max_tokens": 1024},
            },
            "judge-model": {
                "provider": "openai",
                "model": "gpt-judge",
                "api_key_env": "none",
                "params": {"temperature": 0.5},
            },
        },
        "judges": [
            {"model": "judge-model", "params": {"temperature": 0.5}},
        ],
        "composite": {"judge_weight": 0.5, "deepeval_weight": 0.5},
        "eval": {"delay_between_calls": 0},
    }


@pytest.fixture
def sample_prompt():
    """Single eval prompt with all fields."""
    return {
        "id": "T01",
        "category": "coding",
        "subcategory": "test-prompt",
        "difficulty": "medium",
        "prompt": "Write a function that adds two numbers.",
        "ideal": "A simple function that takes two parameters and returns their sum.",
        "criteria": ["Correctness", "Code quality", "Handles edge cases"],
        "check_type": "code_runnable",
    }


@pytest.fixture
def sample_run_entry():
    """Complete run entry with all scoring layers."""
    return {
        "timestamp": "2026-01-15T10:00:00",
        "api_model": "gpt-test",
        "content": "def add(a, b):\n    return a + b",
        "latency_s": 1.5,
        "input_tokens": 50,
        "output_tokens": 20,
        "auto_checks": {
            "flags": [],
            "auto_scores": {},
            "passed": True,
        },
        "judge_scores": {
            "judge-model": {
                "score": 4,
                "rationale": "Good implementation.",
                "judged_at": "2026-01-15T10:01:00",
            }
        },
        "judge_score_avg": 4.0,
        "judge_count": 1,
        "deepeval_scores": {
            "correctness": 0.85,
            "coherence": 0.90,
            "instruction_following": 0.88,
        },
        "deepeval_avg": 0.877,
    }


@pytest.fixture
def sample_model_data(sample_run_entry):
    """Full model result structure."""
    return {
        "model_name": "test-model",
        "created": "2026-01-15T09:00:00",
        "updated": "2026-01-15T10:01:00",
        "runs": {
            "T01": [sample_run_entry],
        },
    }


@pytest.fixture
def mock_provider():
    """MockProvider instance returning a default response."""
    return MockProvider()


@pytest.fixture
def mock_judge_provider():
    """MockProvider that returns valid judge JSON."""
    return MockProvider(
        response='{"score": 4, "rationale": "Good response."}'
    )


@pytest.fixture
def eval_prompts_file(tmp_path):
    """Create a minimal evals/default.json for testing."""
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir()
    prompts = {
        "prompts": [
            {
                "id": "T01",
                "category": "coding",
                "subcategory": "test-prompt",
                "difficulty": "medium",
                "prompt": "Write a function that adds two numbers.",
                "ideal": "A simple add function.",
                "criteria": ["Correctness"],
                "check_type": "code_runnable",
            },
            {
                "id": "T02",
                "category": "reasoning",
                "subcategory": "logic",
                "difficulty": "hard",
                "prompt": "Solve this logic puzzle.",
                "ideal": "The answer is 42.",
                "criteria": ["Accuracy"],
                "check_type": "reasoning",
            },
        ]
    }
    path = evals_dir / "default.json"
    path.write_text(json.dumps(prompts))
    return str(path)
