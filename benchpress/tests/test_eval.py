"""The `eval` command: frozen-config loading + one-shot run/score/summarize,
provider-agnostic via a scripted fake provider."""

import argparse

import benchpress.modules.simulate  # noqa: F401  (register)
from benchpress import cli
from benchpress.core import registry
from benchpress.frozen import load_frozen, manifest_path, run_params_from_config
from benchpress.providers.base import CompletionResult


def test_load_frozen_simulate_has_official_config():
    frozen = load_frozen("simulate")
    assert frozen and frozen["official_run_config"]["seed"] == 42
    assert frozen["official_run_config"]["max_tokens"] == 96000


def test_manifest_path_none_for_unfrozen_benchmark():
    assert manifest_path("causal") is None


def test_run_params_from_config_drops_none_and_maps_timeout():
    cfg = {"max_tokens": 64000, "thinking": "adaptive", "effort": "high",
           "read_timeout_seconds": 900}
    rp = run_params_from_config(cfg)
    assert rp == {"max_tokens": 64000, "thinking": "adaptive", "effort": "high",
                  "read_timeout": 900}


class _GoldProvider:
    """Returns each item's gold answer verbatim, so eval should score 100%."""

    def __init__(self, items):
        self._by_prompt = {
            it.prompt: "\n".join(f"{p.part_id}: {p.expected}" for p in it.parts)
            for it in items
        }

    def complete(self, prompt):
        return CompletionResult(content=self._by_prompt[prompt], stop_reason="end_turn")


def _write_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: ollama\n"
        "    model: fake\n"
        "    company: Test\n"
        "    type: open\n"
    )
    return str(cfg)


def _args(**kw):
    d = {"benchmark": "simulate", "config": None, "model": "fake-model",
         "results_dir": None, "rerun": False, "workers": 1, "seed": 0}
    d.update(kw)
    return argparse.Namespace(**d)


def test_eval_runs_scores_and_reports_perfect_on_gold(tmp_path, monkeypatch, capsys):
    cfg = load_frozen("simulate")["official_run_config"]
    items, _ = registry.get_module("simulate")(cfg["seed"], "hard", cfg["n_per_bundle"])
    monkeypatch.setattr(cli, "get_provider", lambda spec: _GoldProvider(items))

    rc = cli.cmd_eval(_args(config=_write_config(tmp_path), results_dir=str(tmp_path / "results")))
    assert rc == 0
    out = capsys.readouterr().out
    assert "OVERALL" in out and "100.0%" in out          # gold answers -> perfect
    # results file was written and is resumable
    assert (tmp_path / "results" / "simulate" / "fake-model.json").exists()


class _NullProvider:
    def complete(self, prompt):
        return CompletionResult(content="", stop_reason="end_turn")


def test_eval_frozen_config_overrides_model_params(tmp_path, monkeypatch):
    captured = {}

    def fake_get_provider(spec):
        captured.update(spec)
        return _NullProvider()

    monkeypatch.setattr(cli, "get_provider", fake_get_provider)
    # model declares a puny budget; the frozen config must override it to 96000.
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(
        "models:\n  fake-model:\n    provider: ollama\n    model: fake\n"
        "    params:\n      max_tokens: 100\n"
    )
    cli.cmd_eval(_args(config=str(cfgfile), results_dir=str(tmp_path / "r")))
    assert captured["params"]["max_tokens"] == 96000


def test_eval_seed_override_mints_fresh_holdout(monkeypatch):
    # seed 0 (default) -> canonical frozen seed 42; nonzero -> that seed instead.
    canon, _, _, _ = cli._frozen_items("simulate", None)
    fresh, _, _, _ = cli._frozen_items("simulate", 7)
    canon_ids = {i.item_id for i in canon}
    fresh_ids = {i.item_id for i in fresh}
    assert canon_ids != fresh_ids                        # different questions
    assert len(canon) == len(fresh)                      # same shape/count
    # canonical matches the frozen seed exactly
    cfg = load_frozen("simulate")["official_run_config"]
    seeded, _ = registry.get_module("simulate")(cfg["seed"], "hard", cfg["n_per_bundle"])
    assert canon_ids == {i.item_id for i in seeded}


def test_eval_unknown_model_returns_2(tmp_path, capsys):
    rc = cli.cmd_eval(_args(config=_write_config(tmp_path), model="nope",
                            results_dir=str(tmp_path / "r")))
    assert rc == 2
    assert "not found" in capsys.readouterr().out
