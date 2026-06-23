"""AWS Bedrock provider (boto3 Converse API)."""

from __future__ import annotations

import time

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider


class BedrockProvider(Provider):
    def __init__(self, model, region=None, client=None):
        self.model = model
        self.native_config = {"max_tokens": 16000}
        if client is not None:
            self.client = client
        else:
            import boto3  # imported lazily so boto3 is only needed when used

            self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, prompt: str) -> CompletionResult:
        start = time.perf_counter()
        try:
            resp = self.client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.native_config["max_tokens"]},
            )
        except Exception as e:  # boto exceptions are not httpx errors
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
