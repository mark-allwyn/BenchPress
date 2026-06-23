"""Tests for scripts/judge.py - JSON extraction, parsing, prompt building, scoring."""

import pytest
from scripts.judge import (
    _extract_json_object,
    parse_judge_response,
    build_judge_prompt,
    judge_response,
    JUDGE_PROMPT,
)


# ── _extract_json_object ──

class TestExtractJsonObject:
    def test_simple_json(self):
        assert _extract_json_object('{"score": 4}') == '{"score": 4}'

    def test_nested_braces(self):
        text = '{"outer": {"inner": 1}}'
        assert _extract_json_object(text) == text

    def test_strings_with_braces(self):
        text = '{"rationale": "use {x} for formatting"}'
        assert _extract_json_object(text) == text

    def test_no_json(self):
        assert _extract_json_object("no json here") is None

    def test_leading_text(self):
        text = 'Some preamble {"score": 3}'
        assert _extract_json_object(text) == '{"score": 3}'

    def test_escaped_quotes(self):
        text = r'{"text": "say \"hello\""}'
        result = _extract_json_object(text)
        assert result is not None
        assert result.startswith("{")
        assert result.endswith("}")

    def test_multiple_objects_returns_first(self):
        text = '{"a": 1} {"b": 2}'
        assert _extract_json_object(text) == '{"a": 1}'


# ── parse_judge_response ──

class TestParseJudgeResponse:
    def test_clean_json(self):
        result = parse_judge_response('{"score": 4, "rationale": "Good job."}')
        assert result["score"] == 4
        assert result["rationale"] == "Good job."

    def test_markdown_fenced(self):
        raw = '```json\n{"score": 3, "rationale": "Okay."}\n```'
        result = parse_judge_response(raw)
        assert result["score"] == 3

    def test_leading_text(self):
        raw = 'Here is my evaluation: {"score": 5, "rationale": "Excellent."}'
        result = parse_judge_response(raw)
        assert result["score"] == 5

    def test_invalid_score_too_high(self):
        result = parse_judge_response('{"score": 6, "rationale": "Great."}')
        assert result["score"] is None

    def test_invalid_score_too_low(self):
        result = parse_judge_response('{"score": 0, "rationale": "Bad."}')
        assert result["score"] is None

    def test_missing_score(self):
        result = parse_judge_response('{"rationale": "No score given."}')
        assert result["score"] is None

    def test_non_integer_score(self):
        result = parse_judge_response('{"score": 3.5, "rationale": "Half score."}')
        assert result["score"] is None

    def test_empty_input(self):
        result = parse_judge_response("")
        assert result["score"] is None

    def test_garbage_input(self):
        result = parse_judge_response("This is not JSON at all!!!")
        assert result["score"] is None
        assert "Failed to parse" in result["rationale"]

    def test_missing_rationale_defaults_empty(self):
        result = parse_judge_response('{"score": 3}')
        assert result["score"] == 3
        assert result["rationale"] == ""


# ── build_judge_prompt ──

class TestBuildJudgePrompt:
    def test_includes_all_sections(self):
        meta = {
            "prompt": "Test prompt",
            "ideal": "Ideal answer",
            "criteria": ["Criterion A", "Criterion B"],
        }
        auto = {"flags": ["SOME_FLAG"]}
        result = build_judge_prompt(meta, "Test response", auto)

        assert "Test prompt" in result
        assert "Ideal answer" in result
        assert "Criterion A" in result
        assert "Criterion B" in result
        assert "SOME_FLAG" in result
        assert "Test response" in result
        assert JUDGE_PROMPT in result

    def test_no_flags(self):
        meta = {"prompt": "P", "ideal": "I", "criteria": []}
        result = build_judge_prompt(meta, "R", {"flags": []})
        assert "AUTO-CHECK FLAGS" not in result

    def test_criteria_as_string(self):
        meta = {"prompt": "P", "ideal": "I", "criteria": "Single criterion"}
        result = build_judge_prompt(meta, "R", {"flags": []})
        assert "Single criterion" in result


# ── judge_response ──

class TestJudgeResponse:
    def test_successful_scoring(self, mock_judge_provider):
        meta = {"prompt": "P", "ideal": "I", "criteria": []}
        auto = {"flags": []}
        result = judge_response(mock_judge_provider, {}, meta, "Response", auto)
        assert result["judge_score"] == 4
        assert result["judge_rationale"] == "Good response."

    def test_provider_error_returns_none(self):
        from tests.conftest import MockProvider
        provider = MockProvider(error=RuntimeError("API down"))
        meta = {"prompt": "P", "ideal": "I", "criteria": []}
        result = judge_response(provider, {}, meta, "Response", {"flags": []})
        assert result["judge_score"] is None
        assert "Judge error" in result["judge_rationale"]

    def test_parse_failure_returns_none(self):
        from tests.conftest import MockProvider
        provider = MockProvider(response="Not valid JSON at all")
        meta = {"prompt": "P", "ideal": "I", "criteria": []}
        result = judge_response(provider, {}, meta, "Response", {"flags": []})
        assert result["judge_score"] is None

    def test_never_raises(self):
        from tests.conftest import MockProvider
        provider = MockProvider(error=Exception("Unexpected"))
        meta = {"prompt": "P", "ideal": "I", "criteria": []}
        # Should not raise
        result = judge_response(provider, {}, meta, "Response", {"flags": []})
        assert result["judge_score"] is None
