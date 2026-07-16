"""AWS Bedrock provider (boto3 Converse API)."""

from __future__ import annotations

import re
import time

from benchpress.providers._common import error_result
from benchpress.providers.base import CompletionResult, Provider


_TRANSIENT = ("Throttl", "TooManyRequests", "ServiceUnavailable", "503", "429",
              "Timeout", "timeout", "Read timeout")


class BedrockProvider(Provider):
    def __init__(self, model, region=None, client=None, max_retries=4, backoff_base=1.0,
                 max_tokens=16000, thinking=None, effort="high", read_timeout=900):
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
        # Capabilities are discovered on first use: some models accept `thinking`
        # but not the newer `output_config.effort`, and some accept neither. Once a
        # model rejects a field we stop sending it (and record the downgrade).
        self._no_effort = False
        self._no_thinking = False
        if client is not None:
            self.client = client
        else:
            import boto3  # imported lazily so boto3 is only needed when used
            from botocore.config import Config

            # Hard simulation items make the model think for many minutes; the default
            # 60s (and even 300s) read timeout is far too short. Default to 15 min.
            self.client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(read_timeout=read_timeout, connect_timeout=15,
                              retries={"max_attempts": 2, "mode": "standard"}),
            )

    def complete(self, prompt: str) -> CompletionResult:
        kwargs = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": self.native_config["max_tokens"]},
        }
        if self.thinking and not self._no_thinking:
            amrf = {"thinking": {"type": self.thinking}}
            if not self._no_effort:
                amrf["output_config"] = {"effort": self.effort}
            kwargs["additionalModelRequestFields"] = amrf
        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = self.client.converse(**kwargs)
                break
            except Exception as e:  # boto exceptions are not httpx errors
                msg = str(e)
                amrf = kwargs.get("additionalModelRequestFields")
                # A model that accepts `thinking` but not the newer effort field:
                # drop effort (keep thinking) and retry.
                if amrf and "output_config" in amrf and ("output_config" in msg or "effort" in msg):
                    self._no_effort = True
                    amrf.pop("output_config", None)
                    continue
                # A model that does not support this thinking mode at all: drop
                # thinking entirely and record the downgrade for transparency.
                # We are sending the (Anthropic-style) thinking field and the model
                # rejects it - unsupported mode, or a non-Anthropic model that does
                # not accept the field at all (Nova: "extraneous key [thinking]").
                # Drop it, keep running, and record the downgrade.
                if amrf and "thinking" in msg:
                    self._no_thinking = True
                    self.native_config["thinking_downgraded"] = "unsupported_by_model"
                    kwargs.pop("additionalModelRequestFields", None)
                    continue
                # Model caps output below the requested budget (e.g. Nova Pro at
                # 10k): clamp to the stated limit and retry. Task answers are tiny,
                # so this does not disadvantage the model; the cap is recorded.
                mlim = re.search(r"model limit of (\d+)", msg)
                if mlim and kwargs["inferenceConfig"]["maxTokens"] > int(mlim.group(1)):
                    cap = int(mlim.group(1))
                    kwargs["inferenceConfig"]["maxTokens"] = cap
                    self.native_config["max_tokens"] = cap
                    self.native_config["max_tokens_capped"] = cap
                    continue
                if any(s in msg for s in _TRANSIENT) and attempt < self.max_retries:
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
