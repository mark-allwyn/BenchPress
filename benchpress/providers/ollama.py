"""Ollama / local OpenAI-compatible provider (with reasoning fallback)."""

from __future__ import annotations

import time

import httpx

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider


class OllamaProvider(Provider):
    def __init__(self, model, base_url="http://localhost:11434/v1", client=None,
                 *, max_tokens=None, timeout=None):
        self.model = model
        self.native_config = {"max_tokens": max_tokens or 16000}
        self.client = client or httpx.Client(
            base_url=base_url, headers={"Content-Type": "application/json"}, timeout=timeout or 600
        )

    def complete(self, prompt: str) -> CompletionResult:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.native_config["max_tokens"],
        }
        start = time.perf_counter()
        try:
            resp = self.client.post("/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return error_result(e, self.native_config, start)
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        # Reasoning models may exhaust tokens before content; fall back so we
        # never return empty when a reasoning trace exists.
        if not content.strip() and message.get("reasoning"):
            content = message["reasoning"]
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
