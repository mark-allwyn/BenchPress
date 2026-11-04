"""Claude Code provider: runs Claude through the local ``claude`` CLI in print mode,
so evaluations bill against the logged-in claude.ai subscription instead of an API key.

Each prompt is one fresh, stateless CLI call configured to match the frozen run-config
as closely as the CLI allows: every tool disabled, Claude Code's own system prompt
replaced by a minimal one, user customizations (CLAUDE.md, skills, plugins, hooks, MCP)
skipped, adaptive thinking at the requested effort, and the output budget set through
``CLAUDE_CODE_MAX_OUTPUT_TOKENS``. Any Anthropic API key in the environment is stripped
so the CLI can only authenticate with the subscription login.

The CLI still adds a short identity preamble of its own, so scores from this provider
are tagged as a separate harness rather than mixed with direct-API runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable

from benchpress.providers.base import CompletionResult, Provider
from benchpress.providers.errors import sanitize_error

SYSTEM_PROMPT = "You are a helpful assistant."
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


class ClaudeCodeProvider(Provider):
    def __init__(self, model: str, *, max_tokens: int | None = None,
                 effort: str | None = None, timeout: float | None = None,
                 binary: str = "claude", max_retries: int = 4, backoff_base: float = 30.0,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.model = model
        self.binary = binary
        self.timeout = timeout or 1800
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.runner = runner
        self.native_config: dict = {
            "harness": "claude-code-cli",
            "auth": "claude.ai subscription",
            "tools": "off",
            "thinking": "adaptive",
            "max_tokens": max_tokens or 16000,
        }
        if effort:
            self.native_config["effort"] = effort

    def _argv(self) -> list[str]:
        argv = [
            self.binary, "-p",
            "--model", self.model,
            "--tools", "",
            "--system-prompt", SYSTEM_PROMPT,
            "--safe-mode", "--strict-mcp-config", "--no-session-persistence",
            "--output-format", "json",
        ]
        if "effort" in self.native_config:
            argv += ["--effort", self.native_config["effort"]]
        return argv

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.native_config["max_tokens"])
        return env

    def _error(self, msg: str, start: float) -> CompletionResult:
        return CompletionResult(content="", error=sanitize_error(msg),
                                native_config=self.native_config,
                                latency_s=time.perf_counter() - start)

    def complete(self, prompt: str) -> CompletionResult:
        start = time.perf_counter()
        attempt = 0
        # An empty scratch cwd keeps any project files out of the CLI's view.
        with tempfile.TemporaryDirectory(prefix="benchpress-cc-") as cwd:
            while True:
                try:
                    proc = self.runner(self._argv(), input=prompt, capture_output=True,
                                       text=True, timeout=self.timeout, env=self._env(),
                                       cwd=cwd)
                except subprocess.TimeoutExpired:
                    return self._error(f"claude CLI timed out after {self.timeout}s", start)
                except OSError as e:
                    return self._error(f"could not start claude CLI: {e}", start)

                try:
                    data = json.loads(proc.stdout)
                except (json.JSONDecodeError, TypeError):
                    detail = (proc.stderr or proc.stdout or "").strip()[:500]
                    return self._error(
                        f"claude CLI exited {proc.returncode} without JSON: {detail}", start)

                if not data.get("is_error"):
                    return self._parse(data, start)

                status = data.get("api_error_status")
                if status in _RETRYABLE_STATUS and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
                return self._error(
                    f"claude CLI error ({data.get('subtype')}, status {status}): "
                    f"{data.get('result')}", start)

    def _parse(self, data: dict, start: float) -> CompletionResult:
        usage = data.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        input_tokens = sum(usage.get(k) or 0 for k in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return CompletionResult(
            content=data.get("result") or "",
            stop_reason=data.get("stop_reason"),
            input_tokens=input_tokens if usage else None,
            output_tokens=usage.get("output_tokens"),
            thinking_tokens=details.get("thinking_tokens"),
            native_config=self.native_config,
            latency_s=time.perf_counter() - start,
        )
