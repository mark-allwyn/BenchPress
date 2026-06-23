"""Provider factory. More vendors are added in the breadth slice."""

from __future__ import annotations

import os

from benchpress.providers.anthropic import AnthropicProvider
from benchpress.providers.base import CompletionResult, Provider

__all__ = ["CompletionResult", "Provider", "AnthropicProvider", "get_provider"]


def get_provider(spec: dict) -> Provider:
    provider = spec["provider"]
    model = spec["model"]
    if provider == "anthropic":
        return AnthropicProvider(model, os.environ.get(spec.get("api_key_env", "ANTHROPIC_API_KEY"), ""))
    raise ValueError(f"unsupported provider: {provider}")
