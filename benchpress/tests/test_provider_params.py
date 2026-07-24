"""Providers honor injected generation params (max_tokens / timeout), and the
get_provider factory threads a benchmark's frozen run-config to every vendor."""

import json

import httpx

from benchpress.providers import get_provider
from benchpress.providers.anthropic import AnthropicProvider
from benchpress.providers.bedrock import BedrockProvider
from benchpress.providers.cohere import CohereProvider
from benchpress.providers.google import GoogleProvider
from benchpress.providers.ollama import OllamaProvider
from benchpress.providers.openai import OpenAIProvider


def _capturing_client(captured, base="https://example.com"):
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "message": {"content": [{"type": "text", "text": "ok"}]},
            "content": [{"type": "text", "text": "ok"}],
            "usage": {}, "stop_reason": "end_turn",
        })
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base)


def test_openai_honors_max_tokens():
    cap = {}
    OpenAIProvider("gpt-4o", "k", client=_capturing_client(cap), max_tokens=64000).complete("x")
    assert cap["max_tokens"] == 64000


def test_openai_reasoning_model_honors_max_completion_tokens():
    cap = {}
    OpenAIProvider("gpt-5", "k", client=_capturing_client(cap), max_tokens=64000).complete("x")
    assert cap["max_completion_tokens"] == 64000


def test_google_honors_max_tokens():
    cap = {}
    GoogleProvider("gemini-3-pro", "k", client=_capturing_client(cap), max_tokens=64000).complete("x")
    assert cap["generationConfig"]["maxOutputTokens"] == 64000


def test_google_3x_uses_thinking_level_at_effort():
    # Gemini 3.x controls reasoning with generationConfig.thinkingConfig.thinkingLevel.
    cap = {}
    GoogleProvider("gemini-3.1-pro-preview", "k", client=_capturing_client(cap),
                   thinking="adaptive", effort="high").complete("x")
    assert cap["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


def test_google_25_uses_thinking_budget_not_level():
    # Gemini 2.5 rejects thinkingLevel; adaptive effort maps to a dynamic budget (-1).
    cap = {}
    GoogleProvider("gemini-2.5-pro", "k", client=_capturing_client(cap),
                   thinking="adaptive", effort="high").complete("x")
    assert cap["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}


def test_google_no_thinking_config_by_default():
    # With no thinking/effort injected, send no thinkingConfig (model default).
    cap = {}
    GoogleProvider("gemini-2.5-flash", "k", client=_capturing_client(cap),
                   max_tokens=1000).complete("x")
    assert "thinkingConfig" not in cap["generationConfig"]


def test_ollama_honors_max_tokens():
    cap = {}
    OllamaProvider("glm-5", client=_capturing_client(cap), max_tokens=32000).complete("x")
    assert cap["max_tokens"] == 32000


def test_cohere_honors_max_tokens():
    cap = {}
    CohereProvider("command-a", "k", client=_capturing_client(cap), max_tokens=32000).complete("x")
    assert cap["max_tokens"] == 32000


def test_anthropic_honors_max_tokens():
    cap = {}
    AnthropicProvider("claude-3-5-sonnet", "k", client=_capturing_client(cap), max_tokens=64000).complete("x")
    assert cap["max_tokens"] == 64000


def test_defaults_unchanged_when_no_params():
    cap = {}
    OpenAIProvider("gpt-4o", "k", client=_capturing_client(cap)).complete("x")
    assert cap["max_tokens"] == 16000


def test_get_provider_threads_params_to_openai():
    spec = {"provider": "openai", "model": "gpt-4o", "api_key_env": "X",
            "params": {"max_tokens": 64000, "timeout": 900}}
    prov = get_provider(spec)
    assert isinstance(prov, OpenAIProvider)
    assert prov.native_config["max_tokens"] == 64000


def test_get_provider_threads_params_to_google():
    spec = {"provider": "google", "model": "gemini-3-pro", "api_key_env": "X",
            "params": {"max_tokens": 48000}}
    assert get_provider(spec).native_config["max_output_tokens"] == 48000


def test_get_provider_threads_thinking_and_effort_to_google():
    spec = {"provider": "google", "model": "gemini-3.1-pro-preview", "api_key_env": "X",
            "params": {"max_tokens": 96000, "thinking": "adaptive", "effort": "high"}}
    prov = get_provider(spec)
    assert isinstance(prov, GoogleProvider)
    assert prov.thinking == "adaptive" and prov.effort == "high"
    assert prov.native_config["thinking_config"] == {"thinkingLevel": "high"}


def test_get_provider_threads_thinking_and_tokens_to_bedrock():
    spec = {"provider": "bedrock", "model": "eu.anthropic.claude-opus-4-8", "region": "eu-central-1",
            "params": {"max_tokens": 64000, "thinking": "adaptive", "effort": "high",
                       "read_timeout": 900}}
    prov = get_provider(spec)
    assert isinstance(prov, BedrockProvider)
    assert prov.native_config["max_tokens"] == 64000
    assert prov.native_config["thinking"] == "adaptive"
    assert prov.thinking == "adaptive" and prov.effort == "high"


def test_get_provider_read_timeout_alias_for_bedrock():
    # `read_timeout` and `timeout` are interchangeable in params; both parse and
    # yield a BedrockProvider carrying the requested budget.
    spec = {"provider": "bedrock", "model": "m", "region": "eu-central-1",
            "params": {"timeout": 600, "max_tokens": 40000}}
    prov = get_provider(spec)
    assert isinstance(prov, BedrockProvider)
    assert prov.native_config["max_tokens"] == 40000
