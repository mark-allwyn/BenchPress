import httpx

from benchpress.providers.anthropic import AnthropicProvider


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com")


def test_native_config_enables_adaptive_thinking_for_fable():
    p = AnthropicProvider("claude-fable-5", "sk-test")
    assert p.native_config["thinking"] == {"type": "adaptive"}
    assert "temperature" not in p.native_config
    assert p.native_config["max_tokens"] >= 16000


def test_native_config_no_thinking_for_legacy_model():
    p = AnthropicProvider("claude-3-5-haiku", "sk-test")
    assert "thinking" not in p.native_config


def test_native_config_enables_thinking_across_claude_4x():
    for model in ["claude-opus-4-5-20251101", "claude-opus-4-1-20250805",
                  "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]:
        assert AnthropicProvider(model, "sk-test").native_config.get("thinking") == {"type": "adaptive"}


def test_complete_parses_content_status_and_thinking_tokens():
    def handler(request):
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "ESTIMATE: 0.42"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 612,
                "output_tokens": 340,
                "output_tokens_details": {"thinking_tokens": 1875},
            },
        })

    p = AnthropicProvider("claude-fable-5", "sk-test", client=_client(handler))
    r = p.complete("solve this")
    assert r.content == "ESTIMATE: 0.42"
    assert r.stop_reason == "end_turn"
    assert r.input_tokens == 612
    assert r.output_tokens == 340
    assert r.thinking_tokens == 1875
    assert r.error is None
    assert r.native_config["thinking"] == {"type": "adaptive"}


def test_complete_surfaces_refusal_stop_reason():
    def handler(request):
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    r = AnthropicProvider("claude-fable-5", "sk-test", client=_client(handler)).complete("x")
    assert r.stop_reason == "refusal"
    assert r.thinking_tokens is None


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="overloaded")
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}})

    r = AnthropicProvider("claude-fable-5", "sk-test", client=_client(handler), backoff_base=0).complete("x")
    assert r.error is None and r.content == "ok" and calls["n"] == 2


def test_complete_captures_error_without_raising_and_sanitizes_key():
    def handler(request):
        return httpx.Response(429, text="rate limited for x-api-key: sk-ant-api-SECRET")

    r = AnthropicProvider("claude-fable-5", "sk-test", client=_client(handler),
                          backoff_base=0, max_retries=1).complete("x")
    assert r.error is not None
    assert "SECRET" not in r.error
    assert r.content == ""
