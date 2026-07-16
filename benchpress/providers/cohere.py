"""Cohere provider."""

from __future__ import annotations

import time

import httpx

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider


class CohereProvider(Provider):
    def __init__(self, model, api_key, client=None, *, max_tokens=None, timeout=None):
        self.model = model
        self.native_config = {"max_tokens": max_tokens or 16000}
        self.client = client or httpx.Client(
            base_url="https://api.cohere.com",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout or 120,
        )

    def complete(self, prompt: str) -> CompletionResult:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": self.native_config["max_tokens"],
        }
        start = time.perf_counter()
        try:
            resp = self.client.post("/v2/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return error_result(e, self.native_config, start)
        data = resp.json()
        blocks = (data.get("message") or {}).get("content") or []
        content = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        tokens = (data.get("usage") or {}).get("tokens") or {}
        return CompletionResult(
            content=content,
            stop_reason=data.get("finish_reason"),
            input_tokens=tokens.get("input_tokens"),
            output_tokens=tokens.get("output_tokens"),
            native_config=self.native_config,
            latency_s=time.perf_counter() - start,
        )
