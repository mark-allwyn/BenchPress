import threading
import time

import benchpress.modules.causal  # noqa: F401
from benchpress.core import registry
from benchpress.providers.base import CompletionResult
from benchpress.runner import persist, run_model


class _ConcurrencyProbe:
    def __init__(self):
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_seen = 0

    def complete(self, prompt):
        with self._lock:
            self.inflight += 1
            self.max_seen = max(self.max_seen, self.inflight)
        time.sleep(0.02)
        with self._lock:
            self.inflight -= 1
        return CompletionResult(content="ESTIMATE: 0.0", stop_reason="end_turn")


def test_parallel_run_caps_concurrency_and_persists_all(tmp_path):
    items, meta = registry.get_module("causal")(seed=5)
    probe = _ConcurrencyProbe()
    path = tmp_path / "m.json"

    ran = run_model(probe, items, path, model_name="m", benchmark="causal",
                    version=meta.version, workers=4)

    assert ran == len(items)
    assert probe.max_seen >= 2          # genuinely ran in parallel
    assert probe.max_seen <= 4          # respected the cap
    data = persist.load(path)           # file is valid JSON, all items present
    assert len(data["runs"]) == len(items)
    assert all(r[-1]["content"] is not None for r in data["runs"].values())


class _FlakyProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(content="", stop_reason=None, error="429 rate limited")
        return CompletionResult(content="ESTIMATE: 0.0", stop_reason="end_turn")


def test_errored_run_is_retried_on_next_sweep(tmp_path):
    items, meta = registry.get_module("causal")(seed=5)
    item = items[0]
    provider = _FlakyProvider()
    path = tmp_path / "f.json"

    run_model(provider, [item], path, model_name="f", benchmark="causal", version=meta.version)
    assert provider.calls == 1  # first attempt errored

    # Resume: the errored item is not treated as complete, so it is retried.
    run_model(provider, [item], path, model_name="f", benchmark="causal", version=meta.version)
    assert provider.calls == 2
    data = persist.load(path)
    assert data["runs"][item.item_id][-1]["content"] == "ESTIMATE: 0.0"


def test_parallel_resume_skips_completed(tmp_path):
    items, meta = registry.get_module("causal")(seed=5)
    path = tmp_path / "m.json"
    run_model(_ConcurrencyProbe(), items, path, model_name="m", benchmark="causal",
              version=meta.version, workers=4)
    probe2 = _ConcurrencyProbe()
    ran = run_model(probe2, items, path, model_name="m", benchmark="causal",
                    version=meta.version, workers=4)
    assert ran == 0
    assert probe2.max_seen == 0  # no calls made
