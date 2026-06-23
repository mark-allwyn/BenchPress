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


def test_complete_captures_error_without_raising_and_sanitizes_key():
    def handler(request):
        return httpx.Response(429, text="rate limited for x-api-key: sk-ant-api-SECRET")

    r = AnthropicProvider("claude-fable-5", "sk-test", client=_client(handler)).complete("x")
    assert r.error is not None
    assert "SECRET" not in r.error
    assert r.content == ""
