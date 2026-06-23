"""Sweep runner: execute pending (model x item) jobs, save raw responses.

Resume-by-content: an item already carrying a response is skipped unless rerun.
With workers>1 the provider calls run concurrently; writes are serialized under
a lock so the results file is never corrupted.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from benchpress.runner import persist


def _run_dict(r) -> dict:
    return {
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
    }


def _completed(data: dict, item) -> bool:
    runs = data["runs"].get(item.item_id)
    # An errored run is NOT complete - it should be retried on the next sweep.
    return bool(runs) and runs[-1].get("content") is not None and not runs[-1].get("error")


def run_model(provider, items, path, *, model_name, benchmark, version, rerun=False, workers=1) -> int:
    data = persist.load(path)
    data.setdefault("model_name", model_name)
    data.setdefault("benchmark", benchmark)
    data.setdefault("benchmark_version", version)
    data.setdefault("runs", {})

    pending = [it for it in items if rerun or not _completed(data, it)]
    lock = threading.Lock()

    def work(item):
        r = provider.complete(item.prompt)
        with lock:
            data["runs"].setdefault(item.item_id, []).append(_run_dict(r))
            persist.save(path, data)

    if workers > 1 and pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, pending))
    else:
        for item in pending:
            work(item)
    return len(pending)
