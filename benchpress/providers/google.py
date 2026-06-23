"""Google Gemini provider."""

from __future__ import annotations

import time

import httpx

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider


class GoogleProvider(Provider):
    def __init__(self, model, api_key, client=None):
        self.model = model
        self.api_key = api_key
        self.native_config = {"max_output_tokens": 16000}
        self.client = client or httpx.Client(timeout=120)

    def complete(self, prompt: str) -> CompletionResult:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": self.native_config["max_output_tokens"]},
        }
        start = time.perf_counter()
        try:
            resp = self.client.post(url, json=body, params={"key": self.api_key})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return error_result(e, self.native_config, start)
        data = resp.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        content = "".join(p.get("text", "") for p in parts)
        meta = data.get("usageMetadata") or {}
        return CompletionResult(
            content=content,
            stop_reason=cand.get("finishReason"),
            input_tokens=meta.get("promptTokenCount"),
            output_tokens=meta.get("candidatesTokenCount"),
            thinking_tokens=meta.get("thoughtsTokenCount"),
            native_config=self.native_config,
            latency_s=time.perf_counter() - start,
        )
