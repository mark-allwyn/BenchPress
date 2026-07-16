"""OpenAI (and OpenAI-compatible) provider."""

from __future__ import annotations

import time

import httpx

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider

# Models that require max_completion_tokens instead of max_tokens.
_COMPLETION_TOKENS = ("gpt-5", "gpt-4.1", "o1", "o3", "o4")


def _extract_content(message: dict) -> str:
    """Content may be a string or, for reasoning models, a list of segments;
    keep only the answer text."""
    raw = message.get("content")
    if isinstance(raw, list):
        return "".join(
            seg.get("text", "") for seg in raw
            if isinstance(seg, dict) and seg.get("type") == "text"
        )
    return raw or ""


class OpenAIProvider(Provider):
    def __init__(self, model, api_key, base_url="https://api.openai.com/v1", client=None,
                 max_retries=4, backoff_base=1.0, *, max_tokens=None, timeout=None):
        self.model = model
        self.native_config = {"max_tokens": max_tokens or 16000}
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.client = client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout or 300,
        )

    def complete(self, prompt: str) -> CompletionResult:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        mt = self.native_config["max_tokens"]
        if self.model.startswith(_COMPLETION_TOKENS):
            body["max_completion_tokens"] = mt
        else:
            body["max_tokens"] = mt
        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = self.client.post("/chat/completions", json=body)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code == 429 or e.response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
                return error_result(e, self.native_config, start)
            except httpx.HTTPError as e:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
                return error_result(e, self.native_config, start)
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = _extract_content(choice.get("message") or {})
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
