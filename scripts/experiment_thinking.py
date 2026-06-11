"""One-off experiment: re-run the causal benchmark on Claude models with
adaptive thinking enabled, to test whether the harness's no-thinking config
understates their scores. Writes to results/experiments/, never touches
results/*.json.

Usage: python scripts/experiment_thinking.py
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.checks import check_response

MODELS = ["claude-fable-5", "claude-opus-4-8"]
OUT_DIR = "results/experiments"
OUT_FILE = os.path.join(OUT_DIR, "thinking-adaptive.json")
MAX_TOKENS = 16000
WORKERS = 8

API_KEY = os.environ["ANTHROPIC_API_KEY"]
HEADERS = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

lock = threading.Lock()


def call(model: str, prompt: str) -> dict:
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=600) as client:
        resp = client.post("https://api.anthropic.com/v1/messages", headers=HEADERS, json=body)
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    d = resp.json()
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    usage = d.get("usage", {})
    return {
        "content": text,
        "stop_reason": d.get("stop_reason"),
        "stop_details": d.get("stop_details"),
        "output_tokens": usage.get("output_tokens"),
        "thinking_tokens": usage.get("output_tokens_details", {}).get("thinking_tokens"),
    }


def run_one(model: str, p: dict) -> tuple:
    try:
        r = call(model, p["prompt"])
    except Exception as e:
        r = {"error": str(e)}
    if not r.get("error"):
        r["auto_checks"] = check_response(p, r["content"])
    return model, p["id"], r


def main():
    prompts = json.load(open("evals/causal.json"))
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts", prompts)
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    if os.path.exists(OUT_FILE):
        results = json.load(open(OUT_FILE))

    jobs = [
        (m, p) for m in MODELS for p in prompts
        if results.get(m, {}).get(p["id"]) is None
        or results[m][p["id"]].get("error")
    ]
    print(f"{len(jobs)} calls to make")

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_one, m, p) for m, p in jobs]
        for fut in as_completed(futures):
            model, pid, r = fut.result()
            with lock:
                results.setdefault(model, {})[pid] = r
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    json.dump(results, open(OUT_FILE, "w"), indent=1)
                    print(f"{done}/{len(jobs)}", flush=True)
    json.dump(results, open(OUT_FILE, "w"), indent=1)

    # Summary
    for model in MODELS:
        runs = results.get(model, {})
        correct = refused = invalid = errors = 0
        variant = {}
        for p in prompts:
            r = runs.get(p["id"]) or {}
            if r.get("error"):
                errors += 1
                continue
            auto = r.get("auto_checks", {}).get("auto_scores", {})
            if auto.get("extracted_answer") is None:
                if r.get("stop_reason") == "refusal":
                    refused += 1
                else:
                    invalid += 1
                continue
            ok = auto.get("correct", 0) == 1
            correct += ok
            v = p.get("variant", "?")
            variant.setdefault(v, [0, 0])
            variant[v][1] += 1
            variant[v][0] += ok
        valid = len(prompts) - refused - invalid - errors
        acc = correct / valid if valid else 0
        print(f"\n{model}: score={correct}/100 accuracy={acc:.1%} "
              f"refused={refused} invalid={invalid} errors={errors}")
        for v, (c, t) in sorted(variant.items()):
            print(f"  {v}: {c}/{t} ({c/t:.0%})")


if __name__ == "__main__":
    main()
