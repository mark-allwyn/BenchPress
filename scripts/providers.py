"""API providers for different LLM services."""

import os
import json
import time
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    model: str
    prompt_id: str
    content: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class Provider(ABC):
    @abstractmethod
    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        """Returns (content, usage_dict)."""
        ...


class AnthropicProvider(Provider):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120,
        )

    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }
        resp = self.client.post("/v1/messages", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        usage = {
            "input_tokens": data.get("usage", {}).get("input_tokens"),
            "output_tokens": data.get("usage", {}).get("output_tokens"),
        }
        return content, usage


class OpenAIProvider(Provider):
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        p = dict(params)
        # OpenAI reasoning models (o-series) don't support temperature
        if self.model.startswith(("o1", "o3", "o4")):
            p.pop("temperature", None)
        # Newer OpenAI models require max_completion_tokens instead of max_tokens
        if "max_tokens" in p and self.model.startswith(("gpt-5", "gpt-4.1", "o1", "o3", "o4")):
            p["max_completion_tokens"] = p.pop("max_tokens")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **p,
        }
        resp = self.client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = {
            "input_tokens": data.get("usage", {}).get("prompt_tokens"),
            "output_tokens": data.get("usage", {}).get("completion_tokens"),
        }
        return content, usage


class GoogleProvider(Provider):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=120)

    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": params.get("max_tokens", 4096),
                "temperature": params.get("temperature", 0),
            },
        }
        resp = self.client.post(url, json=body, params={"key": self.api_key})
        resp.raise_for_status()
        data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "input_tokens": usage_meta.get("promptTokenCount"),
            "output_tokens": usage_meta.get("candidatesTokenCount"),
        }
        return content, usage


class OllamaProvider(Provider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )

    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }
        resp = self.client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = {
            "input_tokens": data.get("usage", {}).get("prompt_tokens"),
            "output_tokens": data.get("usage", {}).get("completion_tokens"),
        }
        return content, usage


class BedrockProvider(Provider):
    def __init__(self, model: str, region: str = None):
        import boto3
        self.model = model
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, prompt: str, params: dict) -> tuple[str, dict]:
        response = self.client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": params.get("max_tokens", 4096),
                "temperature": params.get("temperature", 0),
            },
        )
        content = response["output"]["message"]["content"][0]["text"]
        usage = {
            "input_tokens": response["usage"]["inputTokens"],
            "output_tokens": response["usage"]["outputTokens"],
        }
        return content, usage


def get_provider(config: dict) -> Provider:
    provider_type = config["provider"]

    if provider_type == "ollama":
        base_url = config.get("base_url", "http://localhost:11434/v1")
        return OllamaProvider(config["model"], base_url)

    api_key_env = config.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env != "none" else "none"

    if not api_key and api_key_env != "none":
        raise ValueError(f"Set env var {api_key_env} with your API key")

    if provider_type == "anthropic":
        return AnthropicProvider(config["model"], api_key)
    elif provider_type in ("openai", "openai_compatible"):
        base_url = config.get("base_url", "https://api.openai.com/v1")
        return OpenAIProvider(config["model"], api_key, base_url)
    elif provider_type == "google":
        return GoogleProvider(config["model"], api_key)
    elif provider_type == "bedrock":
        region = config.get("region")
        return BedrockProvider(config["model"], region)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
