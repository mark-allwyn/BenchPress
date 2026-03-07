#!/usr/bin/env python3
"""
llm-eval — personal LLM benchmark harness.

Results are stored per-model as persistent JSON files in results/.
Run new models over time and compare them against your historical data.

Usage:
    python run.py eval claude-sonnet-4                # Eval a model (with LLM judge scoring)
    python run.py eval claude-sonnet-4 --ids C01 L02  # Specific prompts
    python run.py eval claude-sonnet-4 --category coding --difficulty hard

    python run.py rejudge                              # Rejudge all models with current judge
    python run.py rejudge gpt-4o                      # Rejudge one model
    python run.py rejudge --force                     # Rejudge even if already scored

    python run.py deepeval                             # Score all models with DeepEval metrics
    python run.py deepeval gpt-4o                     # Score one model
    python run.py deepeval --ids C01 C02              # Score specific prompts
    python run.py deepeval --force                    # Re-score even if already scored

    python run.py compare                             # Compare all models
    python run.py compare claude-sonnet-4 gpt-4o      # Compare specific models
    python run.py compare --category coding           # Compare on coding only

    python run.py models                              # List all evaluated models
    python run.py prompts                             # List all prompts in eval set
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from scripts.providers import get_provider, sanitize_error
from scripts.checks import check_response
from scripts.judge import judge_response
from scripts.dashboard import generate_dashboard


RESULTS_DIR = "results"
EVAL_FILE = "evals/default.json"


# ── Data layer ──

def model_path(model_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{model_name}.json")


def load_model_results(model_name: str) -> dict:
    path = model_path(model_name)
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {
        "model_name": model_name,
        "created": datetime.now().isoformat(),
        "runs": {},
    }


def save_model_results(model_name: str, data: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data["updated"] = datetime.now().isoformat()
    target = model_path(model_name)
    tmp = target + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        # Validate before replacing
        with open(tmp) as f:
            json.load(f)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def list_evaluated_models() -> list[str]:
    if not Path(RESULTS_DIR).exists():
        return []
    return sorted(Path(f).stem for f in Path(RESULTS_DIR).glob("*.json") if f.stem != "comparison")


def load_eval(eval_file: str = None) -> list[dict]:
    path = eval_file or EVAL_FILE
    with open(path) as f:
        return json.load(f)["prompts"]


def load_config(path: str = "config.yaml") -> dict:
    if not Path(path).exists():
        print(f"Config not found: {path}")
        print("Copy config.example.yaml to config.yaml and add your API keys.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def filter_prompts(prompts, ids=None, categories=None, difficulty=None):
    out = prompts
    if ids:
        out = [p for p in out if p["id"] in ids]
    if categories:
        cats = [c.lower() for c in categories]
        out = [p for p in out if p["category"].lower() in cats]
    if difficulty:
        diffs = [d.lower() for d in difficulty]
        out = [p for p in out if p["difficulty"].lower() in diffs]
    return out


def latest_run(model_data: dict, pid: str) -> dict:
    runs = model_data.get("runs", {}).get(pid, [])
    return runs[-1] if runs else {}


# ── eval command ──

def cmd_eval(args):
    config = load_config(args.config)
    model_name = args.model

    models_cfg = config.get("models", {})
    if model_name not in models_cfg:
        print(f"Model '{model_name}' not in config.yaml")
        print(f"Available: {', '.join(models_cfg.keys())}")
        sys.exit(1)

    model_cfg = models_cfg[model_name]
    eval_file = config.get("eval", {}).get("eval_file", EVAL_FILE)
    prompts = filter_prompts(load_eval(eval_file), args.ids, args.category, args.difficulty)
    if not prompts:
        print("No prompts match your filters.")
        sys.exit(1)

    model_data = load_model_results(model_name)
    already_done = set(model_data["runs"].keys())
    overlap = {p["id"] for p in prompts} & already_done

    if overlap and not args.rerun:
        print(f"⚠ {len(overlap)} prompts already have results: {', '.join(sorted(overlap))}")
        print(f"  Use --rerun to re-evaluate (appends new run, keeps history)")
        prompts = [p for p in prompts if p["id"] not in overlap]
        if not prompts:
            print("Nothing new to run.")
            return

    try:
        provider = get_provider(model_cfg)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    params = model_cfg.get("params", {})
    delay = config.get("eval", {}).get("delay_between_calls", 1.0)

    # Set up LLM judges
    judges_cfg = config.get("judges", [])
    judge_providers = {}
    for jcfg in judges_cfg:
        jname = jcfg.get("model")
        if not jname or jname not in models_cfg:
            print(f"  Warning: judge model '{jname}' not found in config, skipping")
            continue
        if jname == model_name:
            print(f"  Skipping judge {jname} (cannot self-judge)")
            continue
        try:
            jprov = get_provider(models_cfg[jname])
            judge_providers[jname] = {"provider": jprov, "params": jcfg.get("params", {})}
        except ValueError as e:
            print(f"  Warning: could not init judge provider '{jname}': {e}")

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name} ({model_cfg['model']})")
    if judge_providers:
        print(f"  Judges: {', '.join(judge_providers.keys())}")
    print(f"  Prompts: {len(prompts)}")
    print(f"{'='*60}\n")

    for i, pmeta in enumerate(prompts, 1):
        pid = pmeta["id"]
        print(f"  [{i}/{len(prompts)}] {pid} — {pmeta['subcategory']}...", end=" ", flush=True)

        t0 = time.time()
        try:
            content, usage = provider.complete(pmeta["prompt"], params)
            latency = time.time() - t0
            auto = check_response(pmeta, content)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "api_model": model_cfg["model"],
                "content": content,
                "latency_s": round(latency, 2),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "auto_checks": auto,
                "judge_scores": {},
                "judge_score_avg": None,
                "judge_count": 0,
            }

            flag_str = f" ⚠ {', '.join(auto['flags'])}" if auto["flags"] else ""
            print(f"✓ {latency:.1f}s, {usage.get('output_tokens', '?')} tok{flag_str}")

            if judge_providers:
                for jname, jinfo in judge_providers.items():
                    jr = judge_response(jinfo["provider"], jinfo["params"], pmeta, content, auto)
                    entry["judge_scores"][jname] = {
                        "score": jr["judge_score"],
                        "rationale": jr["judge_rationale"],
                        "judged_at": datetime.now().isoformat(),
                    }
                    score_str = f"{jr['judge_score']}/5" if jr["judge_score"] else "failed"
                    print(f"    Judge ({jname}): {score_str}")
                valid = [v["score"] for v in entry["judge_scores"].values() if v["score"] is not None]
                entry["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
                entry["judge_count"] = len(valid)

            # DeepEval scoring (inline during eval if enabled)
            deepeval_cfg = config.get("deepeval", {})
            if deepeval_cfg.get("enabled"):
                try:
                    from scripts.deepeval_scorer import score_with_deepeval
                    de = score_with_deepeval(pmeta, content, config)
                    entry["deepeval_scores"] = de["deepeval_scores"]
                    entry["deepeval_avg"] = de["deepeval_avg"]
                    if de["deepeval_avg"] is not None:
                        print(f"    DeepEval: {de['deepeval_avg']:.2f} ({', '.join(f'{k}={v:.2f}' for k, v in de['deepeval_scores'].items() if v is not None)})")
                    else:
                        print(f"    DeepEval: failed")
                except Exception as e2:
                    print(f"    DeepEval error: {e2}")

        except Exception as e:
            latency = time.time() - t0
            entry = {
                "timestamp": datetime.now().isoformat(),
                "api_model": model_cfg["model"],
                "content": "",
                "latency_s": round(latency, 2),
                "error": sanitize_error(str(e)),
                "auto_checks": {"flags": ["API_ERROR"], "auto_scores": {}, "passed": False},
                "judge_scores": {},
                "judge_score_avg": None,
                "judge_count": 0,
            }
            print(f"✗ Error: {sanitize_error(str(e))}")

        if pid not in model_data["runs"]:
            model_data["runs"][pid] = []
        model_data["runs"][pid].append(entry)
        try:
            save_model_results(model_name, model_data)
        except Exception as e:
            print(f"  ⚠ Save failed (will retry next prompt): {e}")

        if i < len(prompts):
            time.sleep(delay)

    flagged = sum(
        1 for p in prompts
        if latest_run(model_data, p["id"]).get("auto_checks", {}).get("flags")
    )
    judged = sum(
        1 for p in prompts
        if latest_run(model_data, p["id"]).get("judge_score_avg") is not None
    )
    print(f"\n  Done: {len(prompts)} prompts, {flagged} auto-flagged, {judged} judge-scored")
    print(f"  Results: {model_path(model_name)}")
    print(f"\n  Next step: python run.py compare")

    # Auto-regenerate dashboard
    path = generate_dashboard()
    if path:
        print(f"  Dashboard updated: {path}")


# ── compare command ──

def cmd_compare(args):
    config = load_config(args.config)
    eval_file = config.get("eval", {}).get("eval_file", EVAL_FILE)
    prompts = filter_prompts(load_eval(eval_file), args.ids, args.category, args.difficulty)
    prompts_by_id = {p["id"]: p for p in prompts}
    pids = [p["id"] for p in prompts]

    model_names = args.models if args.models else list_evaluated_models()
    if not model_names:
        print("No evaluated models found.")
        sys.exit(1)

    models = {}
    for name in model_names:
        data = load_model_results(name)
        if data["runs"]:
            models[name] = data

    if not models:
        print("No results to compare.")
        sys.exit(1)

    col_w = max(len(n) for n in models) + 2

    # Load composite weights
    comp_cfg = config.get("composite", {})
    judge_weight = comp_cfg.get("judge_weight", 0.5)
    deepeval_weight = comp_cfg.get("deepeval_weight", 0.5)

    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON — {len(pids)} prompts")
    print(f"{'='*80}\n")

    header = f"{'Model':<{col_w}} {'Composite':>10} {'Score':>9} {'Scored':>7} {'Flagged':>8} {'Latency':>8} {'Tokens':>8}"
    print(header)
    print("─" * len(header))

    leaderboard = []
    for name, data in models.items():
        scores, latencies, tokens = [], [], []
        de_avgs = []
        flagged = 0
        for pid in pids:
            run = latest_run(data, pid)
            if not run:
                continue
            if run.get("judge_score_avg") is not None:
                scores.append(run["judge_score_avg"])
            if run.get("auto_checks", {}).get("flags"):
                flagged += 1
            latencies.append(run.get("latency_s", 0))
            tokens.append(run.get("output_tokens", 0) or 0)
            de_avg = run.get("deepeval_avg")
            if de_avg is not None:
                de_avgs.append(de_avg)

        total = sum(1 for pid in pids if latest_run(data, pid))
        avg_s = sum(scores) / len(scores) if scores else 0
        avg_l = sum(latencies) / len(latencies) if latencies else 0
        avg_t = sum(tokens) / len(tokens) if tokens else 0
        deepeval_avg = sum(de_avgs) / len(de_avgs) if de_avgs else None

        # Composite score
        normalized_judge = (avg_s - 1) / 4 if scores else None
        if normalized_judge is not None and deepeval_avg is not None:
            composite = round(judge_weight * normalized_judge + deepeval_weight * deepeval_avg, 4)
        elif normalized_judge is not None:
            composite = round(normalized_judge, 4)
        elif deepeval_avg is not None:
            composite = round(deepeval_avg, 4)
        else:
            composite = None

        leaderboard.append((name, avg_s, len(scores), total, flagged, avg_l, avg_t, composite))

    leaderboard.sort(key=lambda x: (x[2] > 0, x[7] or 0), reverse=True)

    for name, avg_s, scored, total, flagged, avg_l, avg_t, composite in leaderboard:
        s = f"{avg_s:.2f}/5" if scored else "  —  "
        c = f"{composite:.2f}" if composite is not None else "  —  "
        print(f"{name:<{col_w}} {c:>10} {s:>9} {scored:>3}/{total:<3} {flagged:>8} {avg_l:>7.1f}s {avg_t:>7.0f}")

    # Category breakdown
    if not args.category:
        categories = sorted(set(p["category"] for p in prompts))
        print(f"\n{'─'*70}")
        print("BY CATEGORY\n")

        cw = max(10, *(len(n) + 1 for n in models))
        ch = f"{'Category':<22}"
        for name, *_ in leaderboard:
            ch += f" {name:>{cw}}"
        print(ch)
        print("─" * len(ch))

        for cat in categories:
            cat_pids = [p["id"] for p in prompts if p["category"] == cat]
            row = f"{cat:<22}"
            for name, *_ in leaderboard:
                data = models[name]
                sc = [
                    latest_run(data, pid).get("judge_score_avg")
                    for pid in cat_pids
                    if latest_run(data, pid) and latest_run(data, pid).get("judge_score_avg") is not None
                ]
                row += f" {(f'{sum(sc)/len(sc):.2f}' if sc else '—'):>{cw}}"
            print(row)

    # Flags
    print(f"\n{'─'*70}")
    print("NOTABLE FLAGS\n")
    any_flags = False
    for pid in pids:
        row_flags = {}
        for name in [n for n, *_ in leaderboard]:
            run = latest_run(models[name], pid)
            if run:
                fl = run.get("auto_checks", {}).get("flags", [])
                if fl:
                    row_flags[name] = fl
        if row_flags:
            any_flags = True
            meta = prompts_by_id.get(pid, {})
            print(f"  {pid} ({meta.get('subcategory', '?')}):")
            for name, fl in row_flags.items():
                print(f"    {name}: {', '.join(fl)}")

    if not any_flags:
        print("  None — all passed auto-checks ✓")

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "comparison.md")
        _save_comparison_md(path, leaderboard, models, prompts, prompts_by_id)
        print(f"\nReport saved: {path}")


def cmd_models(args):
    models = list_evaluated_models()
    if not models:
        print("No models evaluated yet.")
        return

    print(f"\nEvaluated models ({len(models)}):\n")
    for name in models:
        data = load_model_results(name)
        total = len(data["runs"])
        scored = sum(1 for rs in data["runs"].values() if rs and rs[-1].get("judge_score_avg") is not None)
        de_scored = sum(
            1 for rs in data["runs"].values()
            if rs and rs[-1].get("deepeval_scores") and any(v is not None for v in rs[-1]["deepeval_scores"].values())
        )
        updated = data.get("updated", data.get("created", "?"))[:10]
        print(f"  {name:<30} {total:>2} prompts, {scored:>2} judged, {de_scored:>2} deepeval  (updated: {updated})")


def cmd_prompts(args):
    prompts = filter_prompts(load_eval(), args.ids, args.category, args.difficulty)
    print(f"\nEval prompts ({len(prompts)}):\n")
    print(f"  {'ID':<6} {'Category':<24} {'Diff':<8} Prompt")
    print(f"  {'─'*80}")
    for p in prompts:
        short = p["prompt"][:50].replace("\n", " ") + "..."
        cat = f"{p['category']}/{p['subcategory']}"
        print(f"  {p['id']:<6} {cat:<24} {p['difficulty']:<8} {short}")


def _save_comparison_md(path, leaderboard, models, prompts, prompts_by_id):
    lines = [
        "# LLM Comparison Report",
        f"*Generated: {datetime.now().isoformat()}*\n",
        "## Leaderboard\n",
        "| Model | Composite | Avg Score | Scored | Flagged | Avg Latency | Avg Tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, avg_s, scored, total, flagged, avg_l, avg_t, composite in leaderboard:
        s = f"{avg_s:.2f}/5" if scored else "-"
        c = f"{composite:.2f}" if composite is not None else "-"
        lines.append(f"| {name} | {c} | {s} | {scored}/{total} | {flagged} | {avg_l:.1f}s | {avg_t:.0f} |")

    lines.append("\n## Per-Prompt Detail\n")
    for p in prompts:
        pid = p["id"]
        lines.append(f"### {pid} — {p['subcategory']} ({p['difficulty']})\n")
        for name, *_ in leaderboard:
            run = latest_run(models[name], pid)
            if not run:
                continue
            score = run.get("judge_score_avg", "—")
            fl = run.get("auto_checks", {}).get("flags", [])
            flag_str = f" ⚠ {', '.join(fl)}" if fl else ""
            judge_details = ""
            for jn, js in run.get("judge_scores", {}).items():
                if js.get("score") is not None:
                    judge_details += f" [{jn}: {js['score']}/5]"
            lines.append(f"**{name}**: score={score}{flag_str}{judge_details}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def cmd_rejudge(args):
    config = load_config(args.config)
    models_cfg = config.get("models", {})

    # Set up judge providers from judges array
    judges_cfg = config.get("judges", [])
    judge_providers = {}
    for jcfg in judges_cfg:
        jname = jcfg.get("model")
        if not jname or jname not in models_cfg:
            print(f"  Warning: judge model '{jname}' not found in config, skipping")
            continue
        # If --judge flag is set, only init that specific judge
        if args.judge and jname != args.judge:
            continue
        try:
            jprov = get_provider(models_cfg[jname])
            judge_providers[jname] = {"provider": jprov, "params": jcfg.get("params", {})}
        except ValueError as e:
            print(f"  Warning: could not init judge provider '{jname}': {e}")

    if not judge_providers:
        if args.judge:
            print(f"Judge '{args.judge}' not found in config judges list.")
        else:
            print("No valid judges configured in config.yaml")
        sys.exit(1)

    # Determine which models to rejudge
    if args.models:
        model_names = args.models
    else:
        model_names = list_evaluated_models()

    if not model_names:
        print("No models to rejudge.")
        return

    eval_file = config.get("eval", {}).get("eval_file", EVAL_FILE)
    prompts = load_eval(eval_file)
    prompts_by_id = {p["id"]: p for p in prompts}
    delay = config.get("eval", {}).get("delay_between_calls", 1.0)

    print(f"\n{'='*60}")
    print(f"  Rejudging with: {', '.join(judge_providers.keys())}")
    print(f"  Models: {len(model_names)}")
    print(f"  Force: {args.force}")
    print(f"{'='*60}\n")

    total_judged = 0
    total_skipped = 0
    total_errors = 0

    for model_name in model_names:
        # Filter out judges that match the model being evaluated (self-judge exclusion)
        applicable_judges = {jn: jp for jn, jp in judge_providers.items() if jn != model_name}
        if not applicable_judges:
            print(f"  Skipping {model_name} (all judges excluded due to self-judge)")
            continue

        model_data = load_model_results(model_name)
        if not model_data["runs"]:
            print(f"  Skipping {model_name} (no results)")
            continue

        pids = list(model_data["runs"].keys())
        changed = False
        judges_needed_by_pid = {}

        for pid in pids:
            runs = model_data.get("runs", {}).get(pid, [])
            if not runs:
                continue
            run = runs[-1]
            if run.get("error"):
                continue

            pmeta = prompts_by_id.get(pid)
            if not pmeta:
                continue

            # Ensure judge_scores dict exists on the latest run
            if "judge_scores" not in run:
                run["judge_scores"] = {}

            # Determine which judges need to score this prompt
            judges_needed = []
            for jname in applicable_judges:
                if not args.force and jname in run["judge_scores"] and run["judge_scores"][jname].get("score") is not None:
                    total_skipped += 1
                    continue
                judges_needed.append(jname)

            if not judges_needed:
                continue

            judges_needed_by_pid[pid] = judges_needed
            auto_checks = run.get("auto_checks", {"flags": [], "auto_scores": {}, "passed": True})

            for jname in judges_needed:
                jinfo = applicable_judges[jname]
                print(f"    {model_name}/{pid} judge={jname}...", end=" ", flush=True)

                try:
                    jr = judge_response(jinfo["provider"], jinfo["params"], pmeta, run["content"], auto_checks)

                    run["judge_scores"][jname] = {
                        "score": jr["judge_score"],
                        "rationale": jr["judge_rationale"],
                        "judged_at": datetime.now().isoformat(),
                    }

                    score_str = f"{jr['judge_score']}/5" if jr["judge_score"] else "failed"
                    print(f"{score_str}")
                    if jr["judge_score"] is not None:
                        total_judged += 1
                    else:
                        total_errors += 1
                    changed = True
                except Exception as e:
                    print(f"error: {e}")
                    total_errors += 1

                time.sleep(delay)

            # Recompute aggregates after all judges scored this prompt
            valid = [v["score"] for v in run["judge_scores"].values() if v["score"] is not None]
            run["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
            run["judge_count"] = len(valid)

        if changed:
            try:
                # Re-read file to avoid clobbering scores written by concurrent processes
                fresh_data = load_model_results(model_name)
                for pid in model_data["runs"]:
                    fresh_runs = fresh_data.get("runs", {}).get(pid, [])
                    if not fresh_runs:
                        continue
                    fresh_run = fresh_runs[-1]
                    if "judge_scores" not in fresh_run:
                        fresh_run["judge_scores"] = {}
                    # Merge our judge scores into the fresh data
                    source_run = model_data["runs"][pid][-1]
                    for jname, jdata in source_run.get("judge_scores", {}).items():
                        if jname in judges_needed_by_pid.get(pid, []):
                            fresh_run["judge_scores"][jname] = jdata
                    # Recompute aggregates
                    valid = [v["score"] for v in fresh_run["judge_scores"].values() if isinstance(v, dict) and v.get("score") is not None]
                    fresh_run["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
                    fresh_run["judge_count"] = len(valid)
                save_model_results(model_name, fresh_data)
            except Exception as e:
                print(f"    Save failed: {e}")

    print(f"\n  Done: {total_judged} judged, {total_skipped} skipped, {total_errors} errors")

    # Auto-regenerate dashboard
    path = generate_dashboard()
    if path:
        print(f"  Dashboard updated: {path}")


def cmd_deepeval(args):
    from scripts.deepeval_scorer import score_with_deepeval

    config = load_config(args.config)

    # Determine which models to score
    if args.models:
        model_names = args.models
    else:
        model_names = list_evaluated_models()

    if not model_names:
        print("No models to score.")
        return

    eval_file = config.get("eval", {}).get("eval_file", EVAL_FILE)
    prompts = load_eval(eval_file)
    prompts_by_id = {p["id"]: p for p in prompts}

    # Optional filtering
    if args.ids:
        prompts = filter_prompts(prompts, ids=args.ids)
        prompts_by_id = {p["id"]: p for p in prompts}
    filter_pids = set(prompts_by_id.keys()) if args.ids else None

    delay = config.get("eval", {}).get("delay_between_calls", 1.0)

    print(f"\n{'='*60}")
    print(f"  DeepEval Scoring")
    print(f"  Models: {len(model_names)}")
    print(f"  Force: {args.force}")
    print(f"{'='*60}\n")

    total_scored = 0
    total_skipped = 0
    total_errors = 0

    for model_name in model_names:
        model_data = load_model_results(model_name)
        if not model_data["runs"]:
            print(f"  Skipping {model_name} (no results)")
            continue

        pids = list(model_data["runs"].keys())
        if filter_pids:
            pids = [p for p in pids if p in filter_pids]

        to_score = []
        for pid in pids:
            run = latest_run(model_data, pid)
            if not run or run.get("error"):
                continue
            de_scores = run.get("deepeval_scores", {})
            if not args.force and de_scores and any(v is not None for v in de_scores.values()):
                total_skipped += 1
                continue
            to_score.append(pid)

        if not to_score:
            print(f"  {model_name}: all prompts already have DeepEval scores")
            continue

        print(f"  {model_name}: scoring {len(to_score)}/{len(pids)} prompts...")

        for i, pid in enumerate(to_score, 1):
            run = latest_run(model_data, pid)
            pmeta = prompts_by_id.get(pid)
            if not pmeta:
                print(f"    [{i}/{len(to_score)}] {pid} - prompt not found in eval set, skipping")
                continue

            print(f"    [{i}/{len(to_score)}] {pid}...", end=" ", flush=True)

            try:
                de = score_with_deepeval(pmeta, run["content"], config)

                # Append a new run entry with DeepEval scores, preserving everything else
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "api_model": run.get("api_model", ""),
                    "content": run["content"],
                    "latency_s": run.get("latency_s", 0),
                    "input_tokens": run.get("input_tokens"),
                    "output_tokens": run.get("output_tokens"),
                    "auto_checks": run.get("auto_checks", {"flags": [], "auto_scores": {}, "passed": True}),
                    "judge_scores": run.get("judge_scores", {}),
                    "judge_score_avg": run.get("judge_score_avg"),
                    "judge_count": run.get("judge_count", 0),
                    "deepeval_scores": de["deepeval_scores"],
                    "deepeval_avg": de["deepeval_avg"],
                }

                model_data["runs"][pid].append(entry)

                if de["deepeval_avg"] is not None:
                    parts = ", ".join(f"{k}={v:.2f}" for k, v in de["deepeval_scores"].items() if v is not None)
                    print(f"avg={de['deepeval_avg']:.2f} ({parts})")
                    total_scored += 1
                else:
                    print("failed")
                    total_errors += 1
            except Exception as e:
                print(f"error: {e}")
                total_errors += 1

            try:
                save_model_results(model_name, model_data)
            except Exception as e:
                print(f"    ⚠ Save failed (will retry next prompt): {e}")

            if i < len(to_score):
                time.sleep(delay)

    print(f"\n  Done: {total_scored} scored, {total_skipped} skipped, {total_errors} errors")

    # Auto-regenerate dashboard
    path = generate_dashboard()
    if path:
        print(f"  Dashboard updated: {path}")


def cmd_migrate_judges(args):
    """Migrate result files from single-judge to multi-judge format."""
    import shutil

    models = list_evaluated_models()
    if not models:
        print("No result files found.")
        return

    # Back up all result files before modifying
    backup_dir = os.path.join(RESULTS_DIR, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    for model in models:
        src = model_path(model)
        dst = os.path.join(backup_dir, f"{model}.json")
        shutil.copy2(src, dst)
    print(f"Backed up {len(models)} result files to {backup_dir}/")

    total_migrated = 0
    total_skipped = 0

    for model in models:
        data = load_model_results(model)
        changed = False

        for prompt_id, runs in data.get("runs", {}).items():
            for run in runs:
                # Skip if already migrated
                if "judge_scores" in run:
                    total_skipped += 1
                    continue

                old_score = run.get("judge_score")
                old_rationale = run.get("judge_rationale")
                old_model = run.get("judge_model")

                # Nothing to migrate if no old judge data
                if old_score is None and old_rationale is None and old_model is None:
                    total_skipped += 1
                    continue

                # Build new multi-judge structure
                judge_key = old_model or "unknown"
                run["judge_scores"] = {
                    judge_key: {
                        "score": old_score,
                        "rationale": old_rationale,
                        "judged_at": None,
                    }
                }
                run["judge_score_avg"] = float(old_score) if old_score is not None else None
                run["judge_count"] = 1 if old_score is not None else 0

                # Remove old fields
                run.pop("judge_score", None)
                run.pop("judge_rationale", None)
                run.pop("judge_model", None)

                changed = True
                total_migrated += 1

        if changed:
            save_model_results(model, data)
            print(f"  Migrated: {model}")

    print(f"\nDone: {total_migrated} entries migrated, {total_skipped} skipped")


def cmd_dashboard(args):
    path = generate_dashboard(args.output if hasattr(args, "output") else None)
    if path:
        print(f"Dashboard generated: {path}")
        if hasattr(args, "open") and args.open:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(path)}")
    else:
        print("No results to generate dashboard from.")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Personal LLM eval harness")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("eval", help="Run eval against a model")
    p.add_argument("model")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--category", nargs="+")
    p.add_argument("--difficulty", nargs="+")
    p.add_argument("--rerun", action="store_true", help="Re-run already evaluated prompts")

    p = sub.add_parser("compare", help="Compare models")
    p.add_argument("models", nargs="*")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--category", nargs="+")
    p.add_argument("--difficulty", nargs="+")
    p.add_argument("--save", action="store_true")

    p = sub.add_parser("models", help="List evaluated models")

    p = sub.add_parser("prompts", help="List eval prompts")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--category", nargs="+")
    p.add_argument("--difficulty", nargs="+")

    p = sub.add_parser("rejudge", help="Re-score existing responses with current judge")
    p.add_argument("models", nargs="*", help="Models to rejudge (default: all)")
    p.add_argument("--judge", default=None, help="Target a specific judge model (default: all configured judges)")
    p.add_argument("--force", action="store_true", help="Rejudge even if already scored by current judge")

    p = sub.add_parser("deepeval", help="Score stored responses with DeepEval metrics")
    p.add_argument("models", nargs="*", help="Models to score (default: all)")
    p.add_argument("--ids", nargs="+", help="Only score specific prompt IDs")
    p.add_argument("--force", action="store_true", help="Re-score even if already has DeepEval scores")

    p = sub.add_parser("migrate-judges", help="Migrate results from single-judge to multi-judge format")

    p = sub.add_parser("dashboard", help="Generate HTML dashboard")
    p.add_argument("--output", default=None, help="Output file path")
    p.add_argument("--open", action="store_true", help="Open in browser")

    args = parser.parse_args()
    cmds = {"eval": cmd_eval, "compare": cmd_compare, "models": cmd_models, "prompts": cmd_prompts, "rejudge": cmd_rejudge, "deepeval": cmd_deepeval, "migrate-judges": cmd_migrate_judges, "dashboard": cmd_dashboard}
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
