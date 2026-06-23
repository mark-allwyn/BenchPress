"""Benchpress command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import benchpress.modules.causal  # noqa: F401  (registers the causal module)
import benchpress.scorers  # noqa: F401  (registers part-scorers)
from benchpress import stats
from benchpress.config import load_models
from benchpress.core import registry
from benchpress.providers import get_provider
from benchpress.runner import persist, run_model, score_model
from benchpress.runner.board import leaderboard


def _items(benchmark: str, seed: int):
    return registry.get_module(benchmark)(seed)


def _model_path(results_dir: str, benchmark: str, model: str) -> Path:
    return Path(results_dir) / benchmark / f"{model}.json"


def _paths(results_dir: str, benchmark: str):
    return sorted(Path(results_dir, benchmark).glob("*.json"))


def cmd_generate(a) -> int:
    items, meta = _items(a.benchmark, a.seed)
    print(f"generate: {len(items)} {a.benchmark} items (v{meta.version}), all passed dual-verification")
    return 0


def cmd_run(a) -> int:
    items, meta = _items(a.benchmark, a.seed)
    spec = load_models(a.config)[a.model]
    provider = get_provider(spec)
    path = _model_path(a.results_dir, a.benchmark, a.model)
    ran = run_model(provider, items, path, model_name=a.model, benchmark=a.benchmark,
                    version=meta.version, rerun=a.rerun, workers=a.workers)
    print(f"run: {a.model} on {len(items)} items, {ran} new responses -> {path}")
    return 0


def cmd_score(a) -> int:
    items, _ = _items(a.benchmark, a.seed)
    path = _model_path(a.results_dir, a.benchmark, a.model)
    score_model(items, path)
    print(f"score: {a.model} scored -> {path}")
    return 0


def cmd_stats(a) -> int:
    print(leaderboard(_paths(a.results_dir, a.benchmark)))
    return 0


def cmd_export(a) -> int:
    items, _ = _items(a.benchmark, a.seed)
    out = {}
    for p in _paths(a.results_dir, a.benchmark):
        data = persist.load(p)
        out[data.get("model_name", p.stem)] = stats.report(persist.scored_results(p), items)
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(text)
        print(f"export: wrote {a.out}")
    else:
        print(text)
    return 0


def cmd_freeze(a) -> int:
    from benchpress.manifest import build_manifest, write_manifest
    items, meta = _items(a.benchmark, a.seed)
    manifest = build_manifest(a.benchmark, a.seed, meta.version, [i.item_id for i in items])
    path = Path(a.benchsets_dir) / f"benchpress-{a.benchmark}-v{meta.version}.json"
    write_manifest(path, manifest)
    print(f"freeze: {manifest['n_items']} items -> {path} (seed={a.seed}, v{meta.version})")
    return 0


def cmd_audit(a) -> int:
    model_results, canonical_acc = {}, {}
    for p in _paths(a.results_dir, a.benchmark):
        data = persist.load(p)
        name = data.get("model_name", p.stem)
        rs = persist.scored_results(p)
        model_results[name] = rs
        canonical_acc[name] = stats.accuracy(rs)["accuracy"]

    if len(model_results) >= 2:
        q = stats.review_queue(stats.item_stats(model_results))
        print(f"audit: all-wrong (miskey/hardest): {q['all_wrong'] or 'none'}; "
              f"dead (too easy): {q['dead'] or 'none'}")
    else:
        print("audit: need >=2 models with results for item analysis")

    if a.fresh_dir:
        fresh_acc = {}
        for p in _paths(a.fresh_dir, a.benchmark):
            data = persist.load(p)
            fresh_acc[data.get("model_name", p.stem)] = stats.accuracy(persist.scored_results(p))["accuracy"]
        for m, g in stats.audit_gap(canonical_acc, fresh_acc).items():
            flag = " FLAG" if g["flagged"] else ""
            print(f"  {m}: canonical {g['canonical']:.2f} vs fresh {g['fresh']:.2f} "
                  f"gap {g['gap']:+.2f}{flag}")
    return 0


HANDLERS = {
    "generate": cmd_generate, "run": cmd_run, "score": cmd_score,
    "stats": cmd_stats, "export": cmd_export, "freeze": cmd_freeze, "audit": cmd_audit,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchpress")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in HANDLERS:
        p = sub.add_parser(name)
        p.add_argument("--benchmark", default="causal")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--results-dir", default="results")
        p.add_argument("--config", default="config.yaml")
        p.add_argument("--model")
        if name == "run":
            p.add_argument("--rerun", action="store_true")
            p.add_argument("--workers", type=int, default=1)
        if name == "export":
            p.add_argument("--out", default=None)
        if name == "freeze":
            p.add_argument("--benchsets-dir", default="benchpress/benchsets")
        if name == "audit":
            p.add_argument("--fresh-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
