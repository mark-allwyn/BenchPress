"""Provider factory + per-vendor best-native-config (resolved in each class)."""

from __future__ import annotations

import os

from benchpress.providers.anthropic import AnthropicProvider
from benchpress.providers.base import CompletionResult, Provider
from benchpress.providers.bedrock import BedrockProvider
from benchpress.providers.cohere import CohereProvider
from benchpress.providers.google import GoogleProvider
from benchpress.providers.ollama import OllamaProvider
from benchpress.providers.openai import OpenAIProvider

__all__ = [
    "CompletionResult", "Provider", "get_provider",
    "AnthropicProvider", "OpenAIProvider", "GoogleProvider",
    "OllamaProvider", "CohereProvider", "BedrockProvider",
]


def _key(spec: dict) -> str:
    env = spec.get("api_key_env", "")
    if not env or env == "none":
        return ""
    return os.environ.get(env, "")


def get_provider(spec: dict) -> Provider:
    """Build the provider adapter for a config entry.

    Generation params under ``spec["params"]`` are threaded to the adapter so a
    benchmark's frozen run-config (max_tokens, timeout/read_timeout, thinking,
    effort) drives every vendor identically. Unknown/unsupported keys are simply
    not forwarded to a given provider - it degrades to its native default.
    """
    provider = spec["provider"]
    model = spec["model"]
    params = spec.get("params") or {}
    max_tokens = params.get("max_tokens")
    timeout = params.get("timeout") or params.get("read_timeout")
    thinking = params.get("thinking")
    effort = params.get("effort")

    if provider == "anthropic":
        return AnthropicProvider(model, _key(spec), max_tokens=max_tokens, timeout=timeout)
    if provider in ("openai", "openai_compatible"):
        return OpenAIProvider(model, _key(spec), spec.get("base_url", "https://api.openai.com/v1"),
                              max_tokens=max_tokens, timeout=timeout)
    if provider == "google":
        return GoogleProvider(model, _key(spec), max_tokens=max_tokens, timeout=timeout)
    if provider == "ollama":
        return OllamaProvider(model, spec.get("base_url", "http://localhost:11434/v1"),
                              max_tokens=max_tokens, timeout=timeout)
    if provider == "cohere":
        return CohereProvider(model, _key(spec), max_tokens=max_tokens, timeout=timeout)
    if provider == "bedrock":
        kw = {}
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        if thinking is not None:
            kw["thinking"] = thinking
        if effort is not None:
            kw["effort"] = effort
        if timeout is not None:
            kw["read_timeout"] = timeout
        return BedrockProvider(model, spec.get("region"), **kw)
    raise ValueError(f"unsupported provider: {provider}")
