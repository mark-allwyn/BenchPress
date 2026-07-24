"""Google Gemini provider."""

from __future__ import annotations

import re
import time

import httpx

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider

_EFFORT_LEVELS = ("minimal", "low", "medium", "high")
_MAX_RETRY_WAIT = 90.0  # cap a single backoff sleep; the API's hint can be large


def _gemini_major(model: str) -> int | None:
    """Major version from a Gemini model id (``gemini-3.1-pro`` -> 3), or None."""
    m = re.search(r"gemini-(\d+)", model)
    return int(m.group(1)) if m else None


def _server_retry_hint(response) -> float | None:
    """Seconds the server asked us to wait, from a 429/503 response, or None.

    Gemini surfaces this two ways: a ``Retry-After`` header (integer seconds) or a
    ``RetryInfo.retryDelay`` (e.g. ``"38s"``) in the error body. Honoring it lets a
    throttled request ride out the quota window instead of failing on a too-short
    fixed backoff. Falls back to None so the caller uses exponential backoff.
    """
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        for detail in (response.json().get("error") or {}).get("details") or []:
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1])
    except (ValueError, AttributeError):
        pass
    return None


class GoogleProvider(Provider):
    def __init__(self, model, api_key, client=None, max_retries=6, backoff_base=1.0,
                 *, max_tokens=None, timeout=None, thinking=None, effort=None):
        self.model = model
        self.api_key = api_key
        self.thinking = thinking
        self.effort = effort
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.native_config = {"max_output_tokens": max_tokens or 16000}
        thinking_config = self._thinking_config()
        if thinking_config is not None:
            self.native_config["thinking_config"] = thinking_config
        self.client = client or httpx.Client(timeout=timeout or 120)

    def _thinking_config(self) -> dict | None:
        """Map the frozen thinking/effort knobs to Gemini's per-family control.

        Gemini 3.x exposes ``thinkingLevel`` (minimal|low|medium|high). Gemini 2.5
        rejects that field and instead takes a numeric ``thinkingBudget``, where
        -1 means dynamic (uncapped) - the faithful match for adaptive high effort.
        With neither knob injected, return None so the model keeps its own default.
        """
        if not self.effort and not self.thinking:
            return None
        major = _gemini_major(self.model)
        if major is not None and major >= 3:
            level = self.effort if self.effort in _EFFORT_LEVELS else "high"
            return {"thinkingLevel": level}
        return {"thinkingBudget": -1}

    def complete(self, prompt: str) -> CompletionResult:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        gen_config = {"maxOutputTokens": self.native_config["max_output_tokens"]}
        thinking_config = self.native_config.get("thinking_config")
        if thinking_config is not None:
            gen_config["thinkingConfig"] = thinking_config
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }
        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = self.client.post(url, json=body, params={"key": self.api_key})
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code == 429 or e.response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    hint = _server_retry_hint(e.response)
                    backoff = self.backoff_base * (2 ** attempt)
                    time.sleep(min(hint if hint is not None else backoff, _MAX_RETRY_WAIT))
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
