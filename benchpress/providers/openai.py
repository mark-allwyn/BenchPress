"""OpenAI (and OpenAI-compatible) provider."""

from __future__ import annotations

import time

import httpx

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider

# Models that require max_completion_tokens instead of max_tokens.
_COMPLETION_TOKENS = ("gpt-5", "gpt-4.1", "o1", "o3", "o4")


class OpenAIProvider(Provider):
    def __init__(self, model, api_key, base_url="https://api.openai.com/v1", client=None):
        self.model = model
        self.native_config = {"max_tokens": 16000}
        self.client = client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=300,
        )

    def complete(self, prompt: str) -> CompletionResult:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        mt = self.native_config["max_tokens"]
        if self.model.startswith(_COMPLETION_TOKENS):
            body["max_completion_tokens"] = mt
        else:
            body["max_tokens"] = mt
        start = time.perf_counter()
        try:
            resp = self.client.post("/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return error_result(e, self.native_config, start)
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return CompletionResult(
            content=content,
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            thinking_tokens=details.get("reasoning_tokens"),
            native_config=self.native_config,
            latency_s=time.perf_counter() - start,
        )
