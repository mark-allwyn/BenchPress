import json
import subprocess

from benchpress.providers import get_provider
from benchpress.providers.claude_code import ClaudeCodeProvider


def _ok_payload(**over):
    payload = {
        "type": "result", "subtype": "success", "is_error": False,
        "result": "ROW1: 10001", "stop_reason": "end_turn", "api_error_status": None,
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 690,
                  "cache_read_input_tokens": 10, "output_tokens": 336,
                  "output_tokens_details": {"thinking_tokens": 326}},
    }
    payload.update(over)
    return payload


class _Runner:
    """Stands in for subprocess.run; records each call and replays canned outputs."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        out = self.outputs.pop(0)
        if isinstance(out, BaseException):
            raise out
        stdout, code = out
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")


def _provider(runner, **kw):
    return ClaudeCodeProvider("claude-opus-5-5", runner=runner, backoff_base=0, **kw)


def test_parses_answer_stop_reason_and_tokens():
    runner = _Runner((json.dumps(_ok_payload()), 0))
    r = _provider(runner).complete("prompt")
    assert r.error is None
    assert r.content == "ROW1: 10001"
    assert r.stop_reason == "end_turn"
    assert r.input_tokens == 2 + 690 + 10
    assert r.output_tokens == 336
    assert r.thinking_tokens == 326


def test_runs_tools_off_minimal_prompt_and_frozen_budget():
    runner = _Runner((json.dumps(_ok_payload()), 0))
    _provider(runner, max_tokens=96000, effort="high", timeout=1800).complete("the prompt")
    argv, kw = runner.calls[0]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--safe-mode" in argv and "--no-session-persistence" in argv
    assert "--system-prompt" in argv
    assert kw["input"] == "the prompt"
    assert kw["timeout"] == 1800
    assert kw["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "96000"


def test_never_uses_an_api_key_so_the_subscription_login_is_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    runner = _Runner((json.dumps(_ok_payload()), 0))
    _provider(runner).complete("x")
    env = runner.calls[0][1]["env"]
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env


def test_truncation_passes_through_as_max_tokens():
    runner = _Runner((json.dumps(_ok_payload(stop_reason="max_tokens", result="")), 0))
    r = _provider(runner).complete("x")
    assert r.error is None and r.stop_reason == "max_tokens"


def test_rate_limit_is_retried_then_succeeds():
    limited = _ok_payload(is_error=True, subtype="error", api_error_status=429,
                          result="rate limited", stop_reason=None)
    runner = _Runner((json.dumps(limited), 1), (json.dumps(_ok_payload()), 0))
    r = _provider(runner).complete("x")
    assert r.error is None and r.content == "ROW1: 10001"
    assert len(runner.calls) == 2


def test_persistent_error_is_reported_not_raised():
    bad = _ok_payload(is_error=True, subtype="error", api_error_status=400,
                      result="invalid model", stop_reason=None)
    runner = _Runner((json.dumps(bad), 1))
    r = _provider(runner).complete("x")
    assert r.content == "" and "invalid model" in r.error
    assert len(runner.calls) == 1


def test_timeout_is_an_error_with_no_stop_reason():
    runner = _Runner(subprocess.TimeoutExpired("claude", 5))
    r = _provider(runner, max_retries=0).complete("x")
    assert r.error and "timed out" in r.error and r.stop_reason is None


def test_unparseable_output_is_an_error():
    runner = _Runner(("not json", 1))
    r = _provider(runner, max_retries=0).complete("x")
    assert r.error and r.content == ""


def test_factory_threads_frozen_params():
    p = get_provider({"provider": "claude_code", "model": "claude-opus-5-5",
                      "params": {"max_tokens": 96000, "effort": "high", "read_timeout": 1800}})
    assert isinstance(p, ClaudeCodeProvider)
    assert p.native_config["max_tokens"] == 96000
    assert p.native_config["effort"] == "high"
    assert p.timeout == 1800


def test_subscription_usage_limit_fails_fast_without_retries():
    # A plan usage limit resets in hours, so backoff retries only burn time; the
    # item is recorded as an error and picked up by the next resume sweep.
    capped = _ok_payload(is_error=True, subtype="success", api_error_status=429,
                         result="You've hit your 5-hour limit · resets 5pm (Europe/London)",
                         stop_reason=None)
    runner = _Runner((json.dumps(capped), 1))
    r = _provider(runner).complete("x")
    assert r.content == "" and "hit your 5-hour limit" in r.error
    assert len(runner.calls) == 1
