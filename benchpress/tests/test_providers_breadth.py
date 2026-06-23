import httpx

from benchpress.providers import get_provider
from benchpress.providers.openai import OpenAIProvider
from benchpress.providers.google import GoogleProvider
from benchpress.providers.ollama import OllamaProvider
from benchpress.providers.cohere import CohereProvider
from benchpress.providers.bedrock import BedrockProvider


def _client(handler, base="https://example.com"):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base)


# ---- OpenAI ------------------------------------------------------------------

def test_openai_parses_content_finish_reason_and_reasoning_tokens():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ESTIMATE: 0.4"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 512}},
        })
    r = OpenAIProvider("gpt-4o", "k", client=_client(handler)).complete("x")
    assert r.content == "ESTIMATE: 0.4"
    assert r.stop_reason == "stop"
    assert r.thinking_tokens == 512
    assert r.input_tokens == 100 and r.output_tokens == 20


def test_openai_reasoning_model_uses_max_completion_tokens():
    captured = {}

    def handler(request):
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})

    OpenAIProvider("gpt-5.5", "k", client=_client(handler)).complete("x")
    assert "max_completion_tokens" in captured and "max_tokens" not in captured


def test_openai_nonreasoning_model_uses_max_tokens():
    captured = {}

    def handler(request):
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})

    OpenAIProvider("gpt-4o", "k", client=_client(handler)).complete("x")
    assert "max_tokens" in captured


def test_openai_error_is_sanitized():
    def handler(request):
        return httpx.Response(400, text="bad Bearer sk-proj-SECRET")
    r = OpenAIProvider("gpt-4o", "k", client=_client(handler)).complete("x")
    assert r.error and "SECRET" not in r.error and r.content == ""


def test_openai_joins_list_content_from_reasoning_models():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": [
                {"type": "thinking", "thinking": "reasoning..."},
                {"type": "text", "text": "ESTIMATE: 0.5"},
            ]}, "finish_reason": "stop"}],
            "usage": {},
        })
    r = OpenAIProvider("magistral-small", "k", client=_client(handler)).complete("x")
    assert r.content == "ESTIMATE: 0.5"


def test_openai_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})

    r = OpenAIProvider("gpt-4o", "k", client=_client(handler), backoff_base=0).complete("x")
    assert r.error is None and r.content == "ok" and calls["n"] == 2


def test_openai_gives_up_after_max_retries_on_429():
    def handler(request):
        return httpx.Response(429, text="slow down")
    r = OpenAIProvider("gpt-4o", "k", client=_client(handler), backoff_base=0, max_retries=2).complete("x")
    assert r.error is not None and r.content == ""


# ---- Google ------------------------------------------------------------------

def test_google_parses_thoughts_tokens_and_finish_reason():
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "yes"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 5, "thoughtsTokenCount": 333},
        })
    r = GoogleProvider("gemini-3-pro", "k", client=_client(handler)).complete("x")
    assert r.content == "yes"
    assert r.stop_reason == "STOP"
    assert r.thinking_tokens == 333


# ---- Ollama ------------------------------------------------------------------

def test_ollama_falls_back_to_reasoning_when_content_empty():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "", "reasoning": "FALLBACK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })
    r = OllamaProvider("glm-5", client=_client(handler)).complete("x")
    assert r.content == "FALLBACK"


# ---- Cohere ------------------------------------------------------------------

def test_cohere_parses_content_and_tokens():
    def handler(request):
        return httpx.Response(200, json={
            "message": {"content": [{"type": "text", "text": "no"}]},
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 12, "output_tokens": 3}},
        })
    r = CohereProvider("command-a", "k", client=_client(handler)).complete("x")
    assert r.content == "no"
    assert r.input_tokens == 12 and r.output_tokens == 3


# ---- Bedrock (injected fake client) ------------------------------------------

class _FakeBedrock:
    def converse(self, **kwargs):
        return {
            "output": {"message": {"content": [{"text": "A->B"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 7, "outputTokens": 4},
        }


def test_bedrock_parses_via_injected_client():
    r = BedrockProvider("anthropic.x", client=_FakeBedrock()).complete("x")
    assert r.content == "A->B"
    assert r.stop_reason == "end_turn"
    assert r.input_tokens == 7


# ---- factory -----------------------------------------------------------------

def test_get_provider_dispatches_by_provider():
    assert isinstance(get_provider({"provider": "openai", "model": "gpt-4o", "api_key_env": "X"}), OpenAIProvider)
    assert isinstance(get_provider({"provider": "google", "model": "g", "api_key_env": "X"}), GoogleProvider)
    assert isinstance(get_provider({"provider": "cohere", "model": "c", "api_key_env": "X"}), CohereProvider)
    assert isinstance(get_provider({"provider": "ollama", "model": "o"}), OllamaProvider)
    assert isinstance(get_provider({"provider": "openai_compatible", "model": "m", "api_key_env": "X"}), OpenAIProvider)
