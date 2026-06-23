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
    provider = spec["provider"]
    model = spec["model"]
    if provider == "anthropic":
        return AnthropicProvider(model, _key(spec))
    if provider in ("openai", "openai_compatible"):
        return OpenAIProvider(model, _key(spec), spec.get("base_url", "https://api.openai.com/v1"))
    if provider == "google":
        return GoogleProvider(model, _key(spec))
    if provider == "ollama":
        return OllamaProvider(model, spec.get("base_url", "http://localhost:11434/v1"))
    if provider == "cohere":
        return CohereProvider(model, _key(spec))
    if provider == "bedrock":
        return BedrockProvider(model, spec.get("region"))
    raise ValueError(f"unsupported provider: {provider}")
