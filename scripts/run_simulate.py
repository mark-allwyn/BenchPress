"""Official evaluation runner for Benchpress-Simulate v1 (frozen).

Runs a model over the frozen simulation set and reports per-task exact-match +
per-row accuracy + a truncation/timeout guard. Results are crash-safe and
resume-by-content: re-run to continue an interrupted evaluation.

Config is pinned by benchpress/modules/simulate/frozen_v1.json. Default model is
Claude Opus 4.8 via Bedrock (the only path allowed here); point --model / the
provider elsewhere to evaluate other models on the identical frozen questions.

Usage:
    python scripts/run_simulate.py [--model <bedrock-model-id>] [--out <path>]
"""
import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = json.loads((ROOT / "benchpress/modules/simulate/frozen_v1.json").read_text())
CFG = MANIFEST["official_run_config"]


def _load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")):
            k, v = line.split("=", 1)
            os.environ[k] = v.strip().strip('"').strip("'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eu.anthropic.claude-opus-4-8")
    ap.add_argument("--out", default=str(ROOT / "results/simulate/v1-eval.json"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    _load_env()
    import benchpress.scorers  # noqa: F401
    import benchpress.modules.simulate  # noqa: F401 (register)
    from benchpress.core import registry
    from benchpress.providers.bedrock import BedrockProvider
    from benchpress.runner import run_model
    from benchpress.runner.score import score_response

    items, meta = registry.get_module("simulate")(CFG["seed"], "hard", CFG["n_per_bundle"])
    by_id = {i.item_id: i for i in items}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    prov = BedrockProvider(args.model, region=os.environ.get("AWS_DEFAULT_REGION"),
                           thinking=CFG["thinking"], effort=CFG["effort"],
                           max_tokens=CFG["max_tokens"], read_timeout=CFG["read_timeout_seconds"])
    print(f"Benchpress-Simulate {MANIFEST['version']}: {len(items)} items, model={args.model}, "
          f"tools OFF, {CFG['max_tokens']} tokens", flush=True)
    run_model(prov, items, out, model_name=args.model, benchmark="simulate",
              version=meta.version, workers=args.workers)

    runs = json.loads(out.read_text())["runs"]
    st = defaultdict(lambda: {"n": 0, "pass": 0, "pr": 0, "pn": 0, "trunc": 0, "err": 0})
    for iid, r in runs.items():
        it = by_id.get(iid)
        if not r or it is None:
            continue
        last = r[-1]
        s = st[it.bundle_id]
        s["n"] += 1
        if last.get("stop_reason") == "max_tokens":
            s["trunc"] += 1
        if last.get("error") or last.get("content") is None:
            s["err"] += 1
            continue
        res = score_response(it, last["content"], last.get("stop_reason"), last.get("error"))
        s["pass"] += res.item_correct
        s["pr"] += sum(1 for p in res.parts if p.correct)
        s["pn"] += len(res.parts)

    print(f"\n=== Benchpress-Simulate {MANIFEST['version']} - {args.model} ===")
    print(f"{'task':10} {'exact':>10}  {'per-row':>8}  {'trunc':>5}  {'err':>3}")
    for b in meta.bundles:
        s = st[b]
        if not s["n"]:
            print(f"{b:10} (no data)")
            continue
        ex = s["pass"] / s["n"] * 100
        pr = s["pr"] / s["pn"] * 100 if s["pn"] else 0
        print(f"{b:10} {s['pass']:2}/{s['n']:2}={ex:3.0f}%  {pr:6.1f}%  {s['trunc']:5}  {s['err']:3}")


if __name__ == "__main__":
    main()
