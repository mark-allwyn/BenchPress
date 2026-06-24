"""Anthropic provider (ported from v2), extended to capture thinking tokens,
stop reason, and the native config used."""

from __future__ import annotations

import re
import time

import httpx

from benchpress.providers.base import CompletionResult, Provider
from benchpress.providers.errors import sanitize_error

# Claude 4.x (opus/sonnet/haiku) and Fable run with adaptive extended thinking.
_THINKING = re.compile(r"^claude-(opus|sonnet|haiku)-4")


def _supports_thinking(model: str) -> bool:
    return model == "claude-fable-5" or bool(_THINKING.match(model))


class AnthropicProvider(Provider):
    def __init__(self, model: str, api_key: str, client: httpx.Client | None = None,
                 max_retries: int = 4, backoff_base: float = 1.0):
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.native_config: dict = {"max_tokens": 16000}
        if _supports_thinking(model):
            self.native_config["thinking"] = {"type": "adaptive"}
        self.client = client or httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120,
        )

    def complete(self, prompt: str) -> CompletionResult:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.native_config,
        }
        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = self.client.post("/v1/messages", json=body)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code == 429 or e.response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
                text = getattr(e.response, "text", "")
                return CompletionResult(
                    content="", error=sanitize_error(f"{e}: {text}"),
                    native_config=self.native_config, latency_s=time.perf_counter() - start,
                )
            except httpx.HTTPError as e:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
                return CompletionResult(
                    content="", error=sanitize_error(str(e)),
                    native_config=self.native_config, latency_s=time.perf_counter() - start,
                )

        latency = time.perf_counter() - start
        data = resp.json()
        content = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        usage = data.get("usage", {}) or {}
        details = usage.get("output_tokens_details") or {}
        return CompletionResult(
            content=content,
            stop_reason=data.get("stop_reason"),
            stop_details={"stop_sequence": data.get("stop_sequence")}
            if data.get("stop_sequence") else None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            thinking_tokens=details.get("thinking_tokens"),
            native_config=self.native_config,
            latency_s=latency,
        )
