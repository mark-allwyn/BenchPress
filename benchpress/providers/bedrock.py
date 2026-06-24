"""AWS Bedrock provider (boto3 Converse API)."""

from __future__ import annotations

import time

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider


_TRANSIENT = ("Throttl", "TooManyRequests", "ServiceUnavailable", "503", "429",
              "Timeout", "timeout", "Read timeout")


class BedrockProvider(Provider):
    def __init__(self, model, region=None, client=None, max_retries=4, backoff_base=1.0,
                 max_tokens=16000, thinking=None, effort="high"):
        self.model = model
        self.thinking = thinking  # None or "adaptive"
        self.effort = effort
        if thinking:
            max_tokens = max(max_tokens, 16000)  # room for the thinking output
        self.native_config = {"max_tokens": max_tokens}
        if thinking:
            self.native_config["thinking"] = thinking
            self.native_config["effort"] = effort
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        if client is not None:
            self.client = client
        else:
            import boto3  # imported lazily so boto3 is only needed when used
            from botocore.config import Config

            # Hard reasoning items can take well over the 60s default read timeout.
            self.client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(read_timeout=300, connect_timeout=15,
                              retries={"max_attempts": 2, "mode": "standard"}),
            )

    def complete(self, prompt: str) -> CompletionResult:
        kwargs = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": self.native_config["max_tokens"]},
        }
        if self.thinking:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": self.thinking},
                "output_config": {"effort": self.effort},
            }
        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = self.client.converse(**kwargs)
                break
            except Exception as e:  # boto exceptions are not httpx errors
                if any(s in str(e) for s in _TRANSIENT) and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                    continue
                return error_result(e, self.native_config, start)
        blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        content = "\n".join(b["text"] for b in blocks if "text" in b)
        usage = resp.get("usage") or {}
        return CompletionResult(
            content=content,
            stop_reason=resp.get("stopReason"),
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            native_config=self.native_config,
            latency_s=time.perf_counter() - start,
        )
