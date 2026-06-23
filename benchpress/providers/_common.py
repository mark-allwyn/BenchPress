"""Shared helpers for provider implementations."""

from __future__ import annotations

import time

from benchpress.providers.base import CompletionResult
from benchpress.providers.errors import sanitize_error


def error_result(exc: Exception, native_config: dict, start: float) -> CompletionResult:
    response = getattr(exc, "response", None)
    text = getattr(response, "text", "") if response is not None else ""
    msg = f"{exc}: {text}" if text else str(exc)
    return CompletionResult(
        content="", error=sanitize_error(msg),
        native_config=native_config, latency_s=time.perf_counter() - start,
    )
