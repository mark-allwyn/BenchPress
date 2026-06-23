"""Minimal sweep runner: execute pending (model x item) jobs, save raw responses.

Resume-by-content: an item already carrying a response is skipped unless rerun.
"""

from __future__ import annotations

from datetime import datetime, timezone

from benchpress.runner import persist


def run_model(provider, items, path, *, model_name, benchmark, version, rerun=False) -> int:
    data = persist.load(path)
    data.setdefault("model_name", model_name)
    data.setdefault("benchmark", benchmark)
    data.setdefault("benchmark_version", version)
    data.setdefault("runs", {})

    ran = 0
    for item in items:
        runs = data["runs"].setdefault(item.item_id, [])
        if runs and not rerun and runs[-1].get("content") is not None:
            continue
        r = provider.complete(item.prompt)
        runs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": r.content,
            "stop_reason": r.stop_reason,
            "stop_details": r.stop_details,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "thinking_tokens": r.thinking_tokens,
            "latency_s": r.latency_s,
            "cost_usd": None,
            "native_config": r.native_config,
            "error": r.error,
            "scored": None,
        })
        ran += 1
        persist.save(path, data)
    return ran
