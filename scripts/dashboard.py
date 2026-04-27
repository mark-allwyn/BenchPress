"""Generate a self-contained HTML dashboard from eval results."""

import html as html_mod
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

RESULTS_DIR = "results"
EVAL_FILE = "evals/general.json"
DOCS_DIR = "docs"
DASHBOARD_FILE = os.path.join(DOCS_DIR, "index.html")
CONFIG_FILE = "config.yaml"

# SEO base URL. Swap in a custom domain here + add docs/CNAME when moving off *.github.io.
GITHUB_PAGES_BASE = "https://mark-allwyn.github.io/BenchPress"


def _seo_head_html(page_slug: str, title: str, description: str) -> str:
    """Return meta/og/twitter/canonical/JSON-LD block for <head> injection.

    page_slug: e.g. "", "categories.html", "companies.html". Empty string = home.
    Adds Dataset structured data for benchmark leaderboard pages so they're eligible
    for Google's rich-results dataset listings.
    """
    url = f"{GITHUB_PAGES_BASE}/{page_slug}".rstrip('/')
    og_image = f"{GITHUB_PAGES_BASE}/og-card.png"
    esc_desc = html_mod.escape(description, quote=True)
    esc_title = html_mod.escape(title, quote=True)

    # Page-specific structured data: WebSite (default) + Dataset for leaderboard pages
    ld_blocks = [
        json.dumps({
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": "BenchPress",
            "url": GITHUB_PAGES_BASE,
            "description": description,
        })
    ]
    if page_slug in ("index.html", "generalist.html", "causal.html"):
        ds_name = ("BenchPress Generalist Benchmark" if page_slug == "generalist.html"
                   else "BenchPress Causal Reasoning Benchmark" if page_slug == "causal.html"
                   else "BenchPress LLM Evaluation Leaderboard")
        ld_blocks.append(json.dumps({
            "@context": "https://schema.org",
            "@type": "Dataset",
            "name": ds_name,
            "description": description,
            "url": url,
            "creator": {"@type": "Person", "name": "Mark Stent"},
            "keywords": "LLM, large language model, benchmark, leaderboard, evaluation, AI",
            "license": "https://opensource.org/licenses/MIT",
            "isAccessibleForFree": True,
        }))

    ld_html = "".join(
        f'<script type="application/ld+json">{b}</script>\n' for b in ld_blocks
    )

    return (
        f'<meta name="description" content="{esc_desc}">\n'
        f'<meta name="robots" content="index, follow, max-image-preview:large">\n'
        f'<link rel="canonical" href="{url}">\n'
        f'<link rel="icon" type="image/svg+xml" href="favicon.svg">\n'
        f'<link rel="apple-touch-icon" href="og-card.png">\n'
        f'<meta property="og:title" content="{esc_title}">\n'
        f'<meta property="og:description" content="{esc_desc}">\n'
        f'<meta property="og:image" content="{og_image}">\n'
        f'<meta property="og:image:width" content="1200">\n'
        f'<meta property="og:image:height" content="630">\n'
        f'<meta property="og:url" content="{url}">\n'
        f'<meta property="og:type" content="website">\n'
        f'<meta property="og:site_name" content="BenchPress">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{esc_title}">\n'
        f'<meta name="twitter:description" content="{esc_desc}">\n'
        f'<meta name="twitter:image" content="{og_image}">\n'
        f'{ld_html}'
    )


# Per-page SEO text. Swap here when copy changes.
SEO = {
    "index.html": (
        "BenchPress | LLM Leaderboard: Generalist + Causal Reasoning",
        "Compare 49 frontier and open-weight LLMs across two independent benchmarks. "
        "Generalist scores breadth across coding, reasoning, writing, and instruction-following. "
        "Causal scores narrow causal-inference reasoning. Multi-judge plus DeepEval scoring, with side-by-side scatter and per-company timeline.",
    ),
    "categories.html": (
        "BenchPress | LLM Performance by Category",
        "Per-category LLM rankings across 8 categories (coding, reasoning, writing, instruction-following, "
        "research, behavioural, meta-cognition, learning) with category heatmap and top-5 radar.",
    ),
    "companies.html": (
        "BenchPress | LLM Rankings by Company",
        "LLM rankings grouped by company (Anthropic, OpenAI, Google, Meta, xAI, Amazon, Alibaba, Mistral, Cohere, Moonshot, MiniMax, Zhipu) "
        "with composite scores, frontier-progress timeline, and category-strengths heatmap.",
    ),
    "judges.html": (
        "BenchPress | LLM-as-Judge Audit: Agreement and Bias",
        "Multi-judge consistency for the four BenchPress judges: distributions, pairwise agreement, "
        "category and difficulty bias, divergence with DeepEval, and biggest disagreements per prompt.",
    ),
    "methodology.html": (
        "BenchPress | Evaluation Methodology",
        "How BenchPress scores LLMs. Two benchmarks: Generalist (80 prompts, 8 categories, multi-judge plus DeepEval) "
        "and Causal Reasoning (100 multiple-choice questions, 20 concept bundles, 5 variants). Composite scoring, "
        "self-judging prevention, and reasoning-model handling explained.",
    ),
    "causal.html": (
        "BenchPress | Causal Reasoning Benchmark Leaderboard",
        "Specialised causal-inference benchmark: 100 multiple-choice questions across 20 concept bundles and 5 variants "
        "(base, trap, transfer, numeric, analyst). Tests DAG reasoning, confounding, colliders, M-bias, and Simpson's paradox. "
        "Per-variant accuracy, top-performer cards, and full leaderboard.",
    ),
    "generalist.html": (
        "BenchPress | Generalist LLM Benchmark Leaderboard",
        "Full breadth LLM leaderboard: 80 prompts across coding, reasoning, writing, instruction-following, "
        "behavioural, learning, meta-cognition, and research. Composite score blends multi-judge LLM scoring "
        "with DeepEval correctness, coherence, and instruction-following.",
    ),
}


def _slug_seo(slug: str) -> str:
    """Render the SEO head block for a page slug like 'index.html'."""
    title, desc = SEO[slug]
    return _seo_head_html(slug, title, desc)


def write_seo_assets(out_dir: str) -> None:
    """Write robots.txt, sitemap.xml, favicon.svg alongside the generated HTML."""
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    robots = (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {GITHUB_PAGES_BASE}/sitemap.xml\n"
    )
    with open(os.path.join(out_dir, "robots.txt"), "w") as f:
        f.write(robots)

    pages = ["index.html", "generalist.html", "causal.html",
             "companies.html", "categories.html", "judges.html",
             "methodology.html"]
    priorities = {"index.html": "1.0", "generalist.html": "0.95",
                  "causal.html": "0.9", "companies.html": "0.7",
                  "categories.html": "0.7", "judges.html": "0.6",
                  "methodology.html": "0.8"}
    urls_xml = ""
    for p in pages:
        loc = f"{GITHUB_PAGES_BASE}/{p}" if p != "index.html" else f"{GITHUB_PAGES_BASE}/"
        urls_xml += (
            f"  <url>\n"
            f"    <loc>{loc}</loc>\n"
            f"    <lastmod>{today}</lastmod>\n"
            f"    <changefreq>weekly</changefreq>\n"
            f"    <priority>{priorities[p]}</priority>\n"
            f"  </url>\n"
        )
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls_xml}"
        '</urlset>\n'
    )
    with open(os.path.join(out_dir, "sitemap.xml"), "w") as f:
        f.write(sitemap)

    # Minimal inline SVG favicon: dark rounded square with a bar-chart mark.
    favicon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#0f1117"/>'
        '<rect x="6" y="18" width="4" height="8" fill="#6c72ff"/>'
        '<rect x="12" y="12" width="4" height="14" fill="#4ecdc4"/>'
        '<rect x="18" y="8" width="4" height="18" fill="#22c55e"/>'
        '<rect x="24" y="14" width="4" height="12" fill="#f59e0b"/>'
        '</svg>'
    )
    with open(os.path.join(out_dir, "favicon.svg"), "w") as f:
        f.write(favicon)


def load_config():
    """Load config.yaml to get judge model name."""
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    return {}


def load_all_results():
    """Load all model result files. Skips comparison.json and *.pre-* historical backups."""
    models = {}
    for f in sorted(Path(RESULTS_DIR).glob("*.json")):
        if f.stem == "comparison" or ".pre-" in f.stem:
            continue
        try:
            with open(f) as fh:
                models[f.stem] = json.load(fh)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: skipping corrupt result file {f.name}: {e}")
    return models


def load_prompts(eval_file=None):
    path = eval_file or EVAL_FILE
    with open(path) as f:
        return json.load(f)["prompts"]


def latest_run(model_data, pid):
    runs = model_data.get("runs", {}).get(pid, [])
    return runs[-1] if runs else {}


def compute_stats(models, prompts, judge_models=None, composite_config=None, models_cfg=None):
    """Compute all stats needed for the dashboard."""
    judge_weight = (composite_config or {}).get("judge_weight", 0.5)
    deepeval_weight = (composite_config or {}).get("deepeval_weight", 0.5)
    models_cfg = models_cfg or {}
    pids = [p["id"] for p in prompts]
    categories = sorted(set(p["category"] for p in prompts))
    cat_pids = {c: [p["id"] for p in prompts if p["category"] == c] for c in categories}
    difficulties = ["easy", "medium", "hard"]
    diff_pids = {d: [p["id"] for p in prompts if p["difficulty"] == d] for d in difficulties}

    leaderboard = []
    for name, data in models.items():
        latencies, tokens, errors = [], [], 0
        flagged = 0
        de_scores_all = {"correctness": [], "coherence": [], "instruction_following": []}
        de_avgs = []
        runs_cache = {pid: latest_run(data, pid) for pid in pids}

        # Per-judge score breakdown (compute first - used for avg_score)
        judge_breakdown = {}
        judge_cat_breakdown = {cat: {} for cat in categories}
        judge_diff_breakdown = {d: {} for d in difficulties}
        pid_to_cat = {p["id"]: p["category"] for p in prompts}
        pid_to_diff = {p["id"]: p["difficulty"] for p in prompts}

        for pid in pids:
            run = runs_cache[pid]
            if not run:
                continue
            if run.get("error"):
                errors += 1
                continue
            if run.get("auto_checks", {}).get("flags"):
                flagged += 1
            latencies.append(run.get("latency_s", 0))
            tokens.append(run.get("output_tokens", 0) or 0)
            # DeepEval scores
            de = run.get("deepeval_scores", {})
            for metric_key in de_scores_all:
                val = de.get(metric_key)
                if val is not None:
                    de_scores_all[metric_key].append(val)
            de_avg = run.get("deepeval_avg")
            if de_avg is not None:
                de_avgs.append(de_avg)
            # Collect per-judge scores (global, per-category, per-difficulty)
            for jname, jdata in run.get("judge_scores", {}).items():
                if jdata.get("score") is not None:
                    sc = jdata["score"]
                    judge_breakdown.setdefault(jname, []).append(sc)
                    cat = pid_to_cat.get(pid)
                    if cat:
                        judge_cat_breakdown[cat].setdefault(jname, []).append(sc)
                    diff = pid_to_diff.get(pid)
                    if diff:
                        judge_diff_breakdown[diff].setdefault(jname, []).append(sc)

        judge_averages = {}
        for jname, jscores in judge_breakdown.items():
            judge_averages[jname] = round(sum(jscores) / len(jscores), 2) if jscores else None

        # Count scorable prompts (non-error runs)
        scorable = sum(1 for pid in pids if runs_cache[pid] and not runs_cache[pid].get("error"))

        # Only include judges with complete coverage (scored every scorable prompt)
        complete_judges = {
            jname: javg for jname, javg in judge_averages.items()
            if javg is not None and len(judge_breakdown[jname]) >= scorable
        }

        # avg_score = mean of complete judges only (fair comparison)
        cj_values = list(complete_judges.values())
        avg_s = sum(cj_values) / len(cj_values) if cj_values else 0
        scored_count = scorable

        total = sum(1 for pid in pids if runs_cache[pid])
        avg_l = sum(latencies) / len(latencies) if latencies else 0
        avg_t = sum(tokens) / len(tokens) if tokens else 0
        median_l = sorted(latencies)[len(latencies) // 2] if latencies else 0

        # Judge agreement (std dev) - only from complete judges
        if len(cj_values) >= 2:
            mean_ja = sum(cj_values) / len(cj_values)
            judge_std_dev = round((sum((x - mean_ja) ** 2 for x in cj_values) / len(cj_values)) ** 0.5, 2)
        else:
            judge_std_dev = None

        # Category scores: mean of complete judges only per category
        cat_scores = {}
        cat_deepeval = {}
        cat_composite = {}
        cat_scorable = {cat: sum(1 for pid in cat_pids[cat] if runs_cache[pid] and not runs_cache[pid].get("error")) for cat in categories}
        for cat in categories:
            # Only include judges that scored every scorable prompt in this category
            cat_ja_vals = []
            for jname, jscores in judge_cat_breakdown[cat].items():
                if jscores and len(jscores) >= cat_scorable[cat]:
                    cat_ja_vals.append(sum(jscores) / len(jscores))
            cat_scores[cat] = round(sum(cat_ja_vals) / len(cat_ja_vals), 2) if cat_ja_vals else None
            # DeepEval per-category average
            cat_de = [
                runs_cache[pid].get("deepeval_avg")
                for pid in cat_pids[cat]
                if runs_cache[pid] and runs_cache[pid].get("deepeval_avg") is not None
            ]
            cat_deepeval[cat] = round(sum(cat_de) / len(cat_de), 2) if cat_de else None
            # Per-category composite
            cat_nj = (cat_scores[cat] - 1) / 4 if cat_scores[cat] is not None else None
            cat_da = cat_deepeval[cat]
            if cat_nj is not None and cat_da is not None:
                cat_composite[cat] = round(judge_weight * cat_nj + deepeval_weight * cat_da, 4)
            elif cat_nj is not None:
                cat_composite[cat] = round(cat_nj, 4)
            elif cat_da is not None:
                cat_composite[cat] = round(cat_da, 4)
            else:
                cat_composite[cat] = None

        # Difficulty scores: mean of complete judges only per difficulty
        diff_scores = {}
        diff_deepeval = {}
        diff_composite = {}
        diff_scorable = {d: sum(1 for pid in diff_pids[d] if runs_cache[pid] and not runs_cache[pid].get("error")) for d in difficulties}
        for d in difficulties:
            diff_ja_vals = []
            for jname, jscores in judge_diff_breakdown[d].items():
                if jscores and len(jscores) >= diff_scorable[d]:
                    diff_ja_vals.append(sum(jscores) / len(jscores))
            diff_scores[d] = round(sum(diff_ja_vals) / len(diff_ja_vals), 2) if diff_ja_vals else None
            d_de = [
                runs_cache[pid].get("deepeval_avg")
                for pid in diff_pids[d]
                if runs_cache[pid] and runs_cache[pid].get("deepeval_avg") is not None
            ]
            diff_deepeval[d] = round(sum(d_de) / len(d_de), 2) if d_de else None
            d_nj = (diff_scores[d] - 1) / 4 if diff_scores[d] is not None else None
            d_da = diff_deepeval[d]
            if d_nj is not None and d_da is not None:
                diff_composite[d] = round(judge_weight * d_nj + deepeval_weight * d_da, 4)
            elif d_nj is not None:
                diff_composite[d] = round(d_nj, 4)
            elif d_da is not None:
                diff_composite[d] = round(d_da, 4)
            else:
                diff_composite[d] = None

        # Judge vs DeepEval divergence (complete judges only)
        # For each prompt, compute mean of complete judges' scores, normalize, compare to deepeval
        divergences = []
        for pid in pids:
            run = runs_cache[pid]
            if run and not run.get("error"):
                da = run.get("deepeval_avg")
                if da is None:
                    continue
                cj_scores = []
                for jname in complete_judges:
                    jdata = run.get("judge_scores", {}).get(jname)
                    if jdata and jdata.get("score") is not None:
                        cj_scores.append(jdata["score"])
                if cj_scores:
                    js_mean = sum(cj_scores) / len(cj_scores)
                    divergences.append(abs((js_mean - 1) / 4 - da))
        avg_divergence = round(sum(divergences) / len(divergences), 4) if divergences else None

        # Score distribution from complete judges only (integer 1-5)
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for jname, jscores in judge_breakdown.items():
            if jname in complete_judges:
                for s in jscores:
                    bucket = max(1, min(5, round(s)))
                    dist[bucket] = dist.get(bucket, 0) + 1

        # Efficiency = score / log2(avg_tokens) - rewards high scores with fewer tokens
        if avg_s > 0 and avg_t > 1:
            efficiency = round(avg_s / math.log2(avg_t), 2)
        else:
            efficiency = 0

        # DeepEval averages
        deepeval_avg = round(sum(de_avgs) / len(de_avgs), 4) if de_avgs else None
        deepeval_metrics = {}
        for metric_key, vals in de_scores_all.items():
            deepeval_metrics[metric_key] = round(sum(vals) / len(vals), 4) if vals else None

        # Composite score: weighted average of normalized judge (0-1) and deepeval avg (0-1)
        normalized_judge = (avg_s - 1) / 4 if cj_values else None
        if normalized_judge is not None and deepeval_avg is not None:
            composite_score = round(judge_weight * normalized_judge + deepeval_weight * deepeval_avg, 4)
        elif normalized_judge is not None:
            composite_score = round(normalized_judge, 4)
        elif deepeval_avg is not None:
            composite_score = round(deepeval_avg, 4)
        else:
            composite_score = None

        # Count prompts with non-None deepeval scores
        de_scored = sum(
            1 for pid in pids
            if runs_cache[pid]
            and not runs_cache[pid].get("error")
            and runs_cache[pid].get("deepeval_scores")
            and any(v is not None for v in runs_cache[pid]["deepeval_scores"].values())
        )

        # Inject company and launch_date from config
        mcfg = models_cfg.get(name, {})
        company = mcfg.get("company", "Unknown")
        launch_date = mcfg.get("launch_date")

        leaderboard.append({
            "name": name,
            "company": company,
            "launch_date": launch_date,
            "last_updated": data.get("updated"),
            "avg_score": round(avg_s, 2),
            "scored": scored_count,
            "de_scored": de_scored,
            "total": total,
            "errors": errors,
            "flagged": flagged,
            "avg_latency": round(avg_l, 1),
            "median_latency": round(median_l, 1),
            "avg_tokens": round(avg_t, 0),
            "efficiency": efficiency,
            "cat_scores": cat_scores,
            "score_dist": dist,
            "deepeval_avg": deepeval_avg,
            "deepeval_metrics": deepeval_metrics,
            "cat_deepeval": cat_deepeval,
            "composite_score": composite_score,
            "cat_composite": cat_composite,
            "diff_scores": diff_scores,
            "diff_deepeval": diff_deepeval,
            "diff_composite": diff_composite,
            "avg_divergence": avg_divergence,
            "judge_averages": judge_averages,
            "judge_std_dev": judge_std_dev,
        })

    leaderboard.sort(key=lambda x: (x["scored"] > 0, x["composite_score"] or 0), reverse=True)

    # Per-prompt flags
    flags = []
    for pid in pids:
        p = next(p for p in prompts if p["id"] == pid)
        row = {}
        for name in models:
            run = latest_run(models[name], pid)
            if run:
                fl = [f for f in run.get("auto_checks", {}).get("flags", [])
                      if not f.startswith("API_ERROR") and f != "EMPTY_RESPONSE"]
                if fl:
                    row[name] = fl
        if row:
            flags.append({"id": pid, "subcategory": p["subcategory"], "models": row})

    companies = sorted(set(m.get("company", "Unknown") for m in leaderboard))

    # Prompt-level results (Feature 5)
    prompt_results = []
    for p in prompts:
        pr = {
            "id": p["id"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "difficulty": p["difficulty"],
            "prompt_text": p["prompt"][:200],
            "models": {},
        }
        for name, data in models.items():
            run = latest_run(data, p["id"])
            if run and not run.get("error"):
                pr["models"][name] = {
                    "judge_score": run.get("judge_score_avg"),
                    "judge_scores": run.get("judge_scores", {}),
                    "judge_count": run.get("judge_count", 0),
                    "deepeval_avg": run.get("deepeval_avg"),
                    "latency_s": round(run.get("latency_s", 0), 1),
                    "error": False,
                    "flags": run.get("auto_checks", {}).get("flags", []),
                }
            elif run and run.get("error"):
                pr["models"][name] = {
                    "judge_score": None,
                    "deepeval_avg": None,
                    "latency_s": 0,
                    "error": True,
                    "flags": [],
                }
        prompt_results.append(pr)

    # --- Per-judge global aggregations ---
    # Collect all (judge, model, pid, score, deepeval_avg, category, difficulty) tuples
    prompt_lookup = {p["id"]: p for p in prompts}
    judge_all_scores = {}  # judge -> list of scores
    judge_cat_scores = {}  # judge -> cat -> list of scores
    judge_diff_scores = {}  # judge -> diff -> list of scores
    judge_model_scores = {}  # judge -> model -> list of scores
    judge_score_dists = {}  # judge -> {1:n, 2:n, ...}
    judge_deepeval_divs = {}  # judge -> list of abs divergences
    # For pairwise: prompt_key -> {judge: score}
    prompt_judge_map = {}  # (model, pid) -> {judge: score}

    for name, data in models.items():
        for pid in pids:
            run = latest_run(data, pid)
            if not run or run.get("error"):
                continue
            p_info = prompt_lookup.get(pid, {})
            cat = p_info.get("category", "")
            diff = p_info.get("difficulty", "")
            de_avg = run.get("deepeval_avg")

            for jname, jdata in run.get("judge_scores", {}).items():
                sc = jdata.get("score")
                if sc is None:
                    continue

                # Global
                judge_all_scores.setdefault(jname, []).append(sc)

                # By category
                judge_cat_scores.setdefault(jname, {}).setdefault(cat, []).append(sc)

                # By difficulty
                judge_diff_scores.setdefault(jname, {}).setdefault(diff, []).append(sc)

                # By model
                judge_model_scores.setdefault(jname, {}).setdefault(name, []).append(sc)

                # Score distribution
                if jname not in judge_score_dists:
                    judge_score_dists[jname] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                judge_score_dists[jname][sc] = judge_score_dists[jname].get(sc, 0) + 1

                # DeepEval divergence per judge
                if de_avg is not None:
                    norm_sc = (sc - 1) / 4
                    judge_deepeval_divs.setdefault(jname, []).append(abs(norm_sc - de_avg))

                # Pairwise map
                key = (name, pid)
                prompt_judge_map.setdefault(key, {})[jname] = sc

    # judge_global: each judge's global average
    judge_global = {}
    for jname, scores in judge_all_scores.items():
        judge_global[jname] = round(sum(scores) / len(scores), 2)

    # judge_by_category
    judge_by_category = {}
    for jname, cats_map in judge_cat_scores.items():
        judge_by_category[jname] = {}
        for cat, scores in cats_map.items():
            judge_by_category[jname][cat] = round(sum(scores) / len(scores), 2)

    # judge_by_difficulty
    judge_by_difficulty = {}
    for jname, diffs_map in judge_diff_scores.items():
        judge_by_difficulty[jname] = {}
        for d, scores in diffs_map.items():
            judge_by_difficulty[jname][d] = round(sum(scores) / len(scores), 2)

    # judge_by_model
    judge_by_model = {}
    for jname, models_map in judge_model_scores.items():
        judge_by_model[jname] = {}
        for mname, scores in models_map.items():
            judge_by_model[jname][mname] = round(sum(scores) / len(scores), 2)

    # judge_pairwise: pairwise agreement between judges (matrix form)
    all_judges = sorted(judge_all_scores.keys())
    judge_pairwise = {}
    judge_pairwise_matrix = {}  # (ja, jb) -> {avg_diff, agree_pct, n}
    for i, ja in enumerate(all_judges):
        for jb in all_judges:
            if ja == jb:
                judge_pairwise_matrix[(ja, jb)] = {"avg_diff": 0, "agree_pct": 100, "n": 0, "self": True}
                continue
            diffs_list = []
            agree_count = 0
            for key, jscores in prompt_judge_map.items():
                if ja in jscores and jb in jscores:
                    diff_val = abs(jscores[ja] - jscores[jb])
                    diffs_list.append(diff_val)
                    if diff_val <= 1:
                        agree_count += 1
            if diffs_list:
                judge_pairwise_matrix[(ja, jb)] = {
                    "avg_diff": round(sum(diffs_list) / len(diffs_list), 2),
                    "agree_pct": round(100 * agree_count / len(diffs_list)),
                    "n": len(diffs_list),
                    "self": False,
                }
                if ja < jb:
                    pair_key = f"{ja} vs {jb}"
                    judge_pairwise[pair_key] = judge_pairwise_matrix[(ja, jb)]

    # judge_vs_deepeval
    judge_vs_deepeval = {}
    for jname, divs in judge_deepeval_divs.items():
        judge_vs_deepeval[jname] = {
            "avg_divergence": round(sum(divs) / len(divs), 4),
        }

    # Biggest disagreements: prompts where judges disagreed most
    biggest_disagreements = []
    for key, jscores in prompt_judge_map.items():
        if len(jscores) < 2:
            continue
        vals = list(jscores.values())
        spread = max(vals) - min(vals)
        if spread > 0:
            model_name, pid = key
            p_info = prompt_lookup.get(pid, {})
            biggest_disagreements.append({
                "prompt_id": pid,
                "model": model_name,
                "category": p_info.get("category", ""),
                "scores": jscores,
                "spread": spread,
            })
    biggest_disagreements.sort(key=lambda x: x["spread"], reverse=True)
    biggest_disagreements = biggest_disagreements[:30]

    return {
        "leaderboard": leaderboard,
        "categories": categories,
        "companies": companies,
        "flags": flags,
        "total_prompts": len(pids),
        "total_models": len(models),
        "judge_models": judge_models or [],
        "generated": datetime.now().isoformat(),
        "difficulties": difficulties,
        "prompt_results": prompt_results,
        "judge_global": judge_global,
        "judge_by_category": judge_by_category,
        "judge_by_difficulty": judge_by_difficulty,
        "judge_by_model": judge_by_model,
        "judge_pairwise": judge_pairwise,
        "judge_pairwise_matrix": {f"{ja}|{jb}": v for (ja, jb), v in judge_pairwise_matrix.items()},
        "judge_score_distributions": judge_score_dists,
        "judge_vs_deepeval": judge_vs_deepeval,
        "biggest_disagreements": biggest_disagreements,
    }


def generate_html(stats, causal_stats=None):
    """Generate the full HTML dashboard."""
    # Inject causal accuracy into leaderboard entries
    if causal_stats:
        causal_by_name = {m["name"]: m["accuracy"] for m in causal_stats["leaderboard"]}
        for m in stats["leaderboard"]:
            m["causal_accuracy"] = causal_by_name.get(m["name"])
    data_json = json.dumps(stats)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - LLM Evaluation Leaderboard</title>
{_slug_seo("index.html")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e7f0;
    --text2: #8b90a5;
    --accent: #6c72ff;
    --accent2: #4ecdc4;
    --green: #22c55e;
    --green-mid: #4ade80;
    --green-light: #86efac;
    --yellow: #eab308;
    --red: #ef4444;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 0;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #242836 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
  }}
  .header-inner {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }}
  .header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
  }}
  .header .byline {{
    font-size: 0.85rem;
    color: var(--text2);
    margin: 0.2rem 0 0;
  }}
  .header .meta {{
    font-size: 0.75rem;
    color: var(--text2);
    margin-top: 0.5rem;
  }}
  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 1.5rem 2.5rem 3rem;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
  }}
  .kpi .label {{
    font-size: 0.75rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }}
  .kpi .value {{
    font-size: 1.8rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .kpi .sub {{
    font-size: 0.8rem;
    color: var(--text2);
    margin-top: 0.25rem;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    align-items: stretch;
  }}
  .grid-full {{
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
  }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text);
  }}
  .card .chart-container {{
    flex: 1;
  }}
  .info-tip {{
    display: inline-block;
    position: relative;
    width: 16px;
    height: 16px;
    line-height: 16px;
    text-align: center;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text2);
    background: var(--surface2);
    border-radius: 50%;
    margin-left: 6px;
    cursor: default;
    vertical-align: middle;
  }}
  .info-tip:hover::after {{
    content: attr(data-info);
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 6px;
    background: var(--surface2);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 400;
    white-space: nowrap;
    z-index: 20;
    pointer-events: none;
    border: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 2px solid var(--border);
    color: var(--text2);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }}
  th.num {{ text-align: right; }}
  td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }}
  td.num {{ text-align: right; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .rank {{
    width: 2rem;
    height: 2rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
  }}
  .rank-1 {{ background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000; }}
  .rank-2 {{ background: linear-gradient(135deg, #c0cfe0, #8da4bf); color: #1e293b; }}
  .rank-3 {{ background: linear-gradient(135deg, #d97706, #b45309); color: #fff; }}
  .rank-n {{ background: var(--surface2); color: var(--text2); }}
  .score-bar {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .score-bar .bar {{
    flex: 1;
    height: 8px;
    background: var(--surface2);
    border-radius: 4px;
    overflow: hidden;
  }}
  .score-bar .bar .fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
  }}
  .score-bar .val {{
    font-weight: 700;
    min-width: 3rem;
    text-align: right;
  }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-error {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .badge-flag {{ background: rgba(234,179,8,0.15); color: var(--yellow); }}
  .badge-ok {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .chart-container {{
    position: relative;
    width: 100%;
    min-height: 320px;
    height: 320px;
  }}
  .flags-list {{
    max-height: 400px;
    overflow-y: auto;
  }}
  .flag-item {{
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
  }}
  .flag-item:last-child {{ border-bottom: none; }}
  .flag-id {{
    font-weight: 600;
    color: var(--accent);
    font-size: 0.85rem;
  }}
  .flag-sub {{
    color: var(--text2);
    font-size: 0.8rem;
  }}
  .flag-models {{
    margin-top: 0.3rem;
    font-size: 0.8rem;
    color: var(--text2);
  }}
  .flag-models span {{
    color: var(--yellow);
  }}
  .cat-table td.cat-name {{
    font-weight: 600;
    text-transform: capitalize;
  }}
  td[data-tip] {{
    position: relative;
    cursor: default;
  }}
  td[data-tip]:hover::after {{
    content: attr(data-tip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface2);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 400;
    white-space: nowrap;
    z-index: 10;
    pointer-events: none;
    border: 1px solid var(--border);
  }}
  .score-cell {{
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .score-5 {{ color: var(--green); }}
  .score-4 {{ color: var(--green-light); }}
  .score-3 {{ color: var(--yellow); }}
  .score-2 {{ color: var(--orange); }}
  .score-1 {{ color: var(--red); }}
  .company-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    flex-shrink: 0;
  }}
  .tabs {{
    display: flex;
    gap: 0.25rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }}
  .tab {{
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text2);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }}
  .tab:hover {{ color: var(--text); }}
  .tab.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .nav {{
    display: flex;
    gap: 0.25rem;
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.25rem;
  }}
  .nav-link {{
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text2);
    text-decoration: none;
    transition: all 0.2s;
  }}
  .nav-link:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
  .nav-link.active {{ color: var(--text); background: var(--accent); }}
  .company-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    flex-shrink: 0;
  }}
  .table-scroll {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    position: relative;
  }}
  .table-scroll::after {{
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 30px;
    pointer-events: none;
    background: linear-gradient(to left, var(--surface), transparent);
    opacity: 0;
    transition: opacity 0.2s;
  }}
  .table-scroll.has-overflow::after {{
    opacity: 1;
  }}
  th[data-sort] {{
    cursor: pointer;
    user-select: none;
    position: relative;
    padding-right: 1.2rem;
  }}
  th[data-sort]:hover {{
    color: var(--text);
  }}
  th[data-sort]::after {{
    content: '';
    position: absolute;
    right: 0.3rem;
    top: 50%;
    transform: translateY(-50%);
    border: 4px solid transparent;
    border-top-color: var(--text2);
    margin-top: 3px;
    opacity: 0.4;
  }}
  th[data-sort].asc::after {{
    border-top-color: var(--accent);
    opacity: 1;
  }}
  th[data-sort].desc::after {{
    border: 4px solid transparent;
    border-bottom-color: var(--accent);
    margin-top: -5px;
    opacity: 1;
  }}
  @media (max-width: 1100px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 900px) {{
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .container {{ padding: 1rem; }}
    .header {{ padding: 1rem; }}
    .header-top {{ flex-direction: column; align-items: flex-start; gap: 0.5rem; }}
    .col-detail {{ display: none; }}
    .show-all-cols .col-detail {{ display: table-cell; }}
    .col-toggle {{ display: inline-block; }}
  }}
  .col-toggle {{
    display: none;
    padding: 0.35rem 0.75rem;
    background: var(--surface2);
    color: var(--text2);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.75rem;
    cursor: pointer;
  }}
  .col-toggle:hover {{ color: var(--text); }}
  @media (max-width: 600px) {{
    .kpi-row {{ grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
    .kpi {{ padding: 1rem; }}
    .kpi .value {{ font-size: 1.4rem; }}
    .container {{ padding: 0.75rem; }}
    .header {{ padding: 1rem 0.75rem; }}
    .header h1 {{ font-size: 1.2rem; }}
    .card {{ padding: 1rem; }}
    .card h2 {{ font-size: 0.9rem; }}
    table {{ font-size: 0.75rem; }}
    th, td {{ padding: 0.4rem 0.5rem; }}
    .score-bar .bar {{ display: none; }}
    .score-bar {{ justify-content: flex-end; }}
    .rank {{ width: 1.6rem; height: 1.6rem; font-size: 0.7rem; }}
    .chart-container {{ height: 260px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- LLM Evaluation Leaderboard</span></h1>
      {_nav_html("index.html", stats)}
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judges: {", ".join(stats["judge_models"])}' if stats.get("judge_models") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- KPIs -->
<div class="kpi-row">
  <div class="kpi">
    <div class="label">Top Model</div>
    <div class="value" style="font-size:1.3rem">{stats['leaderboard'][0]['name'] if stats['leaderboard'] else '-'}</div>
    <div class="sub">{f"{stats['leaderboard'][0]['composite_score']*100:.0f}" if stats['leaderboard'] and stats['leaderboard'][0].get('composite_score') is not None else '-'} general</div>
  </div>
  <div class="kpi">
    <div class="label">Models Evaluated</div>
    <div class="value">{stats['total_models']}</div>
    <div class="sub">{sum(m['scored'] for m in stats['leaderboard'])} total scored responses</div>
  </div>
  <div class="kpi">
    <div class="label">Most Efficient</div>
    <div class="value" style="color:var(--accent2)">{max((m['efficiency'] for m in stats['leaderboard']), default=0):.2f}</div>
    <div class="sub">{max(stats['leaderboard'], key=lambda m: m['efficiency'])['name'] if stats['leaderboard'] else '-'}</div>
  </div>
  <div class="kpi">
    <div class="label">Top Causal Model</div>
    <div class="value" style="font-size:1.3rem">{causal_stats['leaderboard'][0]['name'] if causal_stats and causal_stats.get('leaderboard') else '-'}</div>
    <div class="sub">{f"{causal_stats['leaderboard'][0]['accuracy']*100:.0f}" if causal_stats and causal_stats.get('leaderboard') else '-'} causal</div>
  </div>
</div>

<!-- How to read -->
<div class="card" style="background:var(--surface);border-left:3px solid var(--accent2)">
  <h3 style="font-size:1rem;margin-bottom:0.5rem">How to read this dashboard</h3>
  <p style="color:var(--text2);font-size:0.9rem;margin:0;line-height:1.6">
    Two independent benchmarks, both scored 0–100 so they're directly comparable.
    <strong style="color:var(--text)">General</strong> tests breadth: 80 prompts across 8 categories (coding, reasoning, writing, instruction-following, etc.), scored by 4 LLM judges plus DeepEval.
    <strong style="color:var(--text)">Causal</strong> tests narrow causal-inference reasoning: 100 multiple-choice questions across 20 concept bundles.
    A model can be strong on one and weak on the other; the scatter below shows the trade-off.
  </p>
</div>

<!-- General × Causal scatter -->
<div class="card">
  <div style="margin-bottom:1rem">
    <h2 style="margin-bottom:0.25rem">General × Causal <span class="info-tip" data-info="X-axis: general benchmark composite (0–100). Y-axis: causal benchmark accuracy (0–100). Both axes use the same scale; top-right is best on both. Axes zoom to fit the data cluster.">?</span></h2>
    <p style="color:var(--text2);font-size:0.85rem;margin:0">Each dot is a model, coloured by company. Top-right is best on both. Models without causal data are not shown.</p>
  </div>
  <div style="height:480px;position:relative">
    <canvas id="compositeCausalScatter"></canvas>
  </div>
</div>

<!-- Company Progress Over Time (HERO 2) -->
<div class="card">
  <div style="margin-bottom:1rem">
    <h2 style="margin-bottom:0.25rem">Company Progress Over Time <span class="info-tip" data-info="Best composite per company at each model launch date. Lines show running maximum, the frontier of what each company has shipped.">?</span></h2>
    <p style="color:var(--text2);font-size:0.85rem;margin:0">Each company's best published model over time. Composite (0 to 100).</p>
  </div>
  <div style="height:440px;position:relative">
    <canvas id="timelineChart"></canvas>
  </div>
</div>

<!-- Top-10 General + Top-10 Causal -->
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(420px, 1fr));gap:1rem;margin-bottom:1.5rem">
  <div class="card" style="margin-bottom:0">
    <div style="margin-bottom:1rem">
      <h2 style="margin-bottom:0.25rem">Top 10 Generalist <span class="info-tip" data-info="Generalist benchmark composite (0 to 100): blend of LLM-judge score and DeepEval.">?</span></h2>
      <p style="color:var(--text2);font-size:0.85rem;margin:0">Best 10 on the breadth benchmark.</p>
    </div>
    <div style="height:420px;position:relative">
      <canvas id="topCompositeBar"></canvas>
    </div>
  </div>
  <div class="card" style="margin-bottom:0">
    <div style="margin-bottom:1rem">
      <h2 style="margin-bottom:0.25rem">Top 10 Causal <span class="info-tip" data-info="Causal benchmark accuracy (0 to 100): 100 MC questions across 20 bundles × 5 variants.">?</span></h2>
      <p style="color:var(--text2);font-size:0.85rem;margin:0">Best 10 on causal-inference reasoning.</p>
    </div>
    <div style="height:420px;position:relative">
      <canvas id="topCausalBar"></canvas>
    </div>
  </div>
</div>

<!-- Link cards: where to dig in -->
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:1rem;margin-bottom:1.5rem">
  <a href="generalist.html" class="card" style="margin:0;text-decoration:none;color:var(--text);border-left:3px solid var(--accent2);transition:transform 0.15s">
    <h3 style="margin-bottom:0.4rem">Generalist leaderboard &rarr;</h3>
    <p style="color:var(--text2);font-size:0.85rem;margin:0;line-height:1.5">Full {stats['total_models']}-model leaderboard, DeepEval breakdown, difficulty curve, distributions, flags.</p>
  </a>
  <a href="causal.html" class="card" style="margin:0;text-decoration:none;color:var(--text);border-left:3px solid var(--accent2);transition:transform 0.15s">
    <h3 style="margin-bottom:0.4rem">Causal leaderboard &rarr;</h3>
    <p style="color:var(--text2);font-size:0.85rem;margin:0;line-height:1.5">Per-variant accuracy, bundle consistency heatmap, excluded models.</p>
  </a>
  <a href="companies.html" class="card" style="margin:0;text-decoration:none;color:var(--text);border-left:3px solid var(--accent2);transition:transform 0.15s">
    <h3 style="margin-bottom:0.4rem">Browse by company &rarr;</h3>
    <p style="color:var(--text2);font-size:0.85rem;margin:0;line-height:1.5">Per-company tables, category strengths heatmap, frontier history.</p>
  </a>
  <a href="categories.html" class="card" style="margin:0;text-decoration:none;color:var(--text);border-left:3px solid var(--accent2);transition:transform 0.15s">
    <h3 style="margin-bottom:0.4rem">Browse by category &rarr;</h3>
    <p style="color:var(--text2);font-size:0.85rem;margin:0;line-height:1.5">Coding, writing, reasoning, instruction-following, behavioural, and more.</p>
  </a>
</div>

</div>

<script type="module">
const DATA = await (await fetch('./data.json')).json();
const lb = DATA.leaderboard;
const cats = DATA.categories;

const COLORS = [
  '#6c72ff', '#4ecdc4', '#f97316', '#22c55e', '#ec4899',
  '#eab308', '#8b5cf6', '#06b6d4', '#ef4444', '#84cc16',
  '#f59e0b', '#14b8a6'
];

function compositeColor(s) {{
  if (s >= 0.95) return '#22c55e';
  if (s >= 0.90) return '#4ade80';
  if (s >= 0.85) return '#86efac';
  if (s >= 0.80) return '#eab308';
  if (s >= 0.70) return '#f97316';
  return '#ef4444';
}}

Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

// Scroll hint: detect overflow on .table-scroll containers
document.querySelectorAll('.table-scroll').forEach(el => {{
  function checkOverflow() {{
    el.classList.toggle('has-overflow', el.scrollWidth > el.clientWidth && el.scrollLeft < el.scrollWidth - el.clientWidth - 5);
  }}
  checkOverflow();
  el.addEventListener('scroll', checkOverflow);
  window.addEventListener('resize', checkOverflow);
}});

// scoreChart and agreementChart removed: scoreChart was redundant with the Top 10 bar.
// agreementChart already lives on judges.html. Their JS is gone with the markup.

// Charts inside <details> render at 0×0 if the details starts closed, so trigger
// a resize when the details first opens. Chart.js auto-redraws on the resize event.
document.querySelectorAll('details').forEach(d => {{
  d.addEventListener('toggle', () => {{
    if (d.open) window.dispatchEvent(new Event('resize'));
  }});
}});

// Top-10 horizontal bar charts: composite-only and causal-only.
// X-axis zooms to the visible range so small differences are legible.
function topBar(canvasId, getScore, scoreFmt, axisLabel, axisMax) {{
  const el = document.getElementById(canvasId);
  if (!el) return;
  const ranked = lb
    .map(m => ({{ name: m.name, score: getScore(m) }}))
    .filter(m => m.score != null)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);
  if (!ranked.length) return;
  // Zoom: floor at 90% of the lowest visible value (rounded to a nice step)
  const lo = ranked[ranked.length - 1].score;
  const hi = ranked[0].score;
  const span = hi - lo;
  const pad = Math.max(span * 0.15, axisMax * 0.02);
  const xMin = Math.max(0, lo - pad);
  const xMax = Math.min(axisMax, hi + pad);
  const view = ranked.slice().reverse(); // bottom-up
  new Chart(el, {{
    type: 'bar',
    data: {{
      labels: view.map(m => m.name),
      datasets: [{{
        data: view.map(m => m.score),
        backgroundColor: view.map(m => compositeColor(m.score / axisMax) + 'cc'),
        borderColor: view.map(m => compositeColor(m.score / axisMax)),
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => scoreFmt(ctx.raw) }} }},
      }},
      scales: {{
        x: {{ min: xMin, max: xMax, title: {{ display: true, text: axisLabel + ' (zoomed)', color: '#8b90a5' }} }},
        y: {{ ticks: {{ font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}}
topBar('topCompositeBar', m => m.composite_score != null ? m.composite_score * 100 : null, v => v.toFixed(0) + ' general', 'General (0–100)', 100);
topBar('topCausalBar',    m => m.causal_accuracy != null ? m.causal_accuracy * 100 : null, v => v.toFixed(0) + ' causal', 'Causal accuracy (0–100)', 100);

// Company Progress Over Time (homepage hero 2). Same chart as the companies page.
{_company_colors_js()}
(function () {{
  const canvas = document.getElementById('timelineChart');
  if (!canvas) return;
  const timeCompanies = {{}};
  lb.forEach(m => {{
    if (!m.launch_date || m.composite_score == null) return;
    const c = m.company || 'Unknown';
    if (!timeCompanies[c]) timeCompanies[c] = [];
    timeCompanies[c].push({{ date: m.launch_date, score: m.composite_score, name: m.name }});
  }});
  const datasets = [];
  Object.keys(timeCompanies).sort().forEach(company => {{
    const points = timeCompanies[company].sort((a, b) => a.date.localeCompare(b.date));
    let runMax = 0;
    const data = points.map(p => {{
      runMax = Math.max(runMax, p.score);
      return {{ x: p.date, y: runMax * 100, modelName: p.name, rawScore: p.score }};
    }});
    datasets.push({{
      label: company, data,
      borderColor: companyColor(company), backgroundColor: companyColor(company) + '33',
      borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, tension: 0.1, fill: false,
    }});
  }});
  let allScores = [];
  datasets.forEach(ds => ds.data.forEach(pt => allScores.push(pt.y)));
  const dataMin = allScores.length ? Math.min(...allScores) : 0;
  const dataMax = allScores.length ? Math.max(...allScores) : 100;
  const padding = (dataMax - dataMin) * 0.15 || 5;
  const yMin = Math.max(0, Math.floor((dataMin - padding) / 5) * 5);
  const yMax = Math.min(100, Math.ceil((dataMax + padding) / 5) * 5);
  new Chart(canvas, {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        x: {{ type: 'time', time: {{ unit: 'month', tooltipFormat: 'MMM yyyy' }}, title: {{ display: true, text: 'Launch Date', color: '#8b90a5' }} }},
        y: {{ min: yMin, max: yMax, ticks: {{ stepSize: 5 }}, title: {{ display: true, text: 'Composite (0 to 100)', color: '#8b90a5' }} }}
      }},
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.raw.modelName + ' (' + ctx.raw.y.toFixed(0) + ')' }} }}
      }}
    }}
  }});
}})();

// Composite × Causal scatter, only models with both signals
(function () {{
  const scatterEl = document.getElementById('compositeCausalScatter');
  if (!scatterEl) return;
  // Both axes on a 0–100 scale so a 5-point move means the same on each axis
  const points = lb
    .filter(m => m.composite_score != null && m.causal_accuracy != null)
    .map(m => ({{
      x: m.composite_score * 100,
      y: m.causal_accuracy * 100,
      name: m.name,
      company: m.company || 'Unknown',
    }}));
  const companies = [...new Set(points.map(p => p.company))].sort();
  const companyColor = Object.fromEntries(companies.map((c, i) => [c, COLORS[i % COLORS.length]]));
  const datasets = companies.map(c => ({{
    label: c,
    data: points.filter(p => p.company === c),
    backgroundColor: companyColor[c] + 'cc',
    borderColor: companyColor[c],
    borderWidth: 1.5,
    pointRadius: 6,
    pointHoverRadius: 9,
  }}));
  // Zoom to a square data range (same scale on both axes) so distance is meaningful
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xLo = Math.min(...xs), xHi = Math.max(...xs);
  const yLo = Math.min(...ys), yHi = Math.max(...ys);
  const span = Math.max(xHi - xLo, yHi - yLo);
  const xMid = (xLo + xHi) / 2, yMid = (yLo + yHi) / 2;
  const half = span / 2 + Math.max(span * 0.08, 2);
  const xMin = Math.max(0, xMid - half), xMax = Math.min(100, xMid + half);
  const yMin = Math.max(0, yMid - half), yMax = Math.min(100, yMid + half);
  new Chart(scatterEl, {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'right', labels: {{ boxWidth: 10, padding: 8, font: {{ size: 11 }} }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => `${{ctx.raw.name}} | general ${{ctx.raw.x.toFixed(0)}} | causal ${{ctx.raw.y.toFixed(0)}}`,
            title: () => '',
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'General benchmark (0–100)', color: '#8b90a5' }}, min: xMin, max: xMax }},
        y: {{ title: {{ display: true, text: 'Causal benchmark (0–100)', color: '#8b90a5' }}, min: yMin, max: yMax }}
      }}
    }}
  }});
}})();

// Efficiency chart (only renders if canvas present)
(function () {{
  const el = document.getElementById('efficiencyChart');
  if (!el) return;
  const effSorted = [...lb].sort((a, b) => b.efficiency - a.efficiency);
  const colour = e => e >= 0.50 ? '#22c55e' : e >= 0.45 ? '#4ade80' : e >= 0.40 ? '#86efac' : e >= 0.35 ? '#eab308' : '#f97316';
  new Chart(el, {{
    type: 'bar',
    data: {{
      labels: effSorted.map(m => m.name),
      datasets: [{{
        data: effSorted.map(m => m.efficiency),
        backgroundColor: effSorted.map(m => colour(m.efficiency) + 'cc'),
        borderColor: effSorted.map(m => colour(m.efficiency)),
        borderWidth: 1, borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ beginAtZero: true, title: {{ display: true, text: 'Efficiency', color: '#8b90a5' }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}})();

// Radar chart removed from homepage; lives on categories.html now.
const top5 = lb.slice(0, 5);
(function () {{
  const radarEl = document.getElementById('radarChart');
  if (!radarEl) return;
  new Chart(radarEl, {{
    type: 'radar',
    data: {{
      labels: cats.map(c => c.replace('_', ' ')),
      datasets: top5.map((m, i) => ({{
        label: m.name,
        data: cats.map(c => ((m.cat_composite && m.cat_composite[c]) || 0) * 100),
        borderColor: COLORS[i],
        backgroundColor: COLORS[i] + '22',
        pointBackgroundColor: COLORS[i],
        borderWidth: 2,
        pointRadius: 3,
      }}))
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        r: {{
          min: 0, max: 100, ticks: {{ stepSize: 20, display: false }},
          grid: {{ color: '#2e3345' }}, angleLines: {{ color: '#2e3345' }},
          pointLabels: {{ font: {{ size: 11 }}, color: '#e4e7f0' }}
        }}
      }},
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }}
    }}
  }});
}})();

// Score distribution stacked bar (only renders if canvas present)
(function () {{
  const el = document.getElementById('distChart');
  if (!el) return;
  new Chart(el, {{
    type: 'bar',
    data: {{
      labels: lb.map(m => m.name),
      datasets: [5, 4, 3, 2, 1].map((score, si) => ({{
        label: score + '/5',
        data: lb.map(m => m.score_dist[score] || 0),
        backgroundColor: ['#22c55e','#86efac','#eab308','#f97316','#ef4444'][si] + 'cc',
        borderRadius: 2,
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }},
      scales: {{ x: {{ stacked: true, ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}, y: {{ stacked: true, beginAtZero: true }} }}
    }}
  }});
}})();

// Sortable leaderboard table
(function() {{
  const table = document.getElementById('leaderboard-table');
  if (!table) return;
  const headers = table.querySelectorAll('th[data-sort]');
  const tbody = table.querySelector('tbody');

  headers.forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.sort;
      const type = th.dataset.type;
      const wasDesc = th.classList.contains('desc');
      const wasAsc = th.classList.contains('asc');

      // Clear all sort states
      headers.forEach(h => h.classList.remove('asc', 'desc'));

      // Toggle: desc->asc, asc->desc, default->desc for num, asc for str
      let dir;
      if (wasDesc) dir = 'asc';
      else if (wasAsc) dir = 'desc';
      else dir = type === 'str' ? 'asc' : 'desc';

      th.classList.add(dir);

      const rows = Array.from(tbody.querySelectorAll('tr.model-row'));
      rows.sort((a, b) => {{
        let va = a.dataset[key];
        let vb = b.dataset[key];
        if (type === 'num') {{
          va = parseFloat(va) || 0;
          vb = parseFloat(vb) || 0;
          return dir === 'asc' ? va - vb : vb - va;
        }} else {{
          return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        }}
      }});
      rows.forEach(r => {{
        tbody.appendChild(r);
        // Re-attach detail row after its parent
        const detail = tbody.querySelector('tr.judge-detail-row[data-parent="' + r.dataset.name + '"]');
        if (detail) tbody.appendChild(detail);
      }});
      setParams({{ sort: key, dir: dir }});
    }});
  }});

  // Click to expand judge detail rows
  tbody.addEventListener('click', function(e) {{
    const row = e.target.closest('tr.model-row');
    if (!row) return;
    const name = row.dataset.name;
    const detail = tbody.querySelector('tr.judge-detail-row[data-parent="' + name + '"]');
    if (detail) {{
      detail.style.display = detail.style.display === 'none' ? '' : 'none';
    }}
  }});
}})();

// URL param utilities
function getParams() {{ return Object.fromEntries(new URLSearchParams(location.search)); }}
function setParams(obj) {{
  const p = new URLSearchParams(location.search);
  Object.entries(obj).forEach(([k, v]) => {{
    if (v === '' || v == null) p.delete(k);
    else p.set(k, v);
  }});
  const qs = p.toString();
  history.replaceState(null, '', qs ? '?' + qs : location.pathname);
}}

// Company filter dropdown
(function() {{
  const sel = document.getElementById('company-filter');
  if (!sel) return;
  const companies = DATA.companies || [];
  companies.forEach(c => {{
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  }});

  function filterByCompany(val) {{
    const rows = document.querySelectorAll('#leaderboard-table tbody tr.model-row');
    rows.forEach(r => {{
      const show = !val || r.dataset.company === val;
      r.style.display = show ? '' : 'none';
      const detail = document.querySelector('tr.judge-detail-row[data-parent="' + r.dataset.name + '"]');
      if (detail) detail.style.display = 'none';
    }});
  }}

  sel.addEventListener('change', () => {{
    filterByCompany(sel.value);
    setParams({{ company: sel.value }});
  }});

  // Restore from URL
  const params = getParams();
  if (params.company) {{
    sel.value = params.company;
    filterByCompany(params.company);
  }}
}})();

// Restore sort from URL
(function() {{
  const params = getParams();
  if (params.sort) {{
    const th = document.querySelector('#leaderboard-table th[data-sort="' + params.sort + '"]');
    if (th) {{
      const dir = params.dir || 'desc';
      // Clear existing
      document.querySelectorAll('#leaderboard-table th[data-sort]').forEach(h => h.classList.remove('asc', 'desc'));
      th.classList.add(dir);
      const tbody = document.querySelector('#leaderboard-table tbody');
      const type = th.dataset.type;
      const rows = Array.from(tbody.querySelectorAll('tr.model-row'));
      rows.sort((a, b) => {{
        let va = a.dataset[params.sort];
        let vb = b.dataset[params.sort];
        if (type === 'num') {{
          va = parseFloat(va) || 0;
          vb = parseFloat(vb) || 0;
          return dir === 'asc' ? va - vb : vb - va;
        }} else {{
          return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        }}
      }});
      rows.forEach(r => {{
        tbody.appendChild(r);
        const detail = tbody.querySelector('tr.judge-detail-row[data-parent="' + r.dataset.name + '"]');
        if (detail) tbody.appendChild(detail);
      }});
    }}
  }}
}})();

// Difficulty grouped bar chart (top 5 models)
(function() {{
  const canvas = document.getElementById('difficultyChart');
  if (!canvas) return;
  const diffs = DATA.difficulties || ['easy', 'medium', 'hard'];
  const top5 = lb.slice(0, 5);
  const diffColors = {{ easy: '#22c55e', medium: '#eab308', hard: '#ef4444' }};
  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: top5.map(m => m.name),
      datasets: diffs.map(d => ({{
        label: d.charAt(0).toUpperCase() + d.slice(1),
        data: top5.map(m => ((m.diff_composite && m.diff_composite[d]) || 0) * 100),
        backgroundColor: diffColors[d] + 'cc',
        borderColor: diffColors[d],
        borderWidth: 1,
        borderRadius: 4,
      }}))
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
        }}
      }},
      scales: {{
        y: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}})();

// Judge vs DeepEval divergence bar chart
(function() {{
  const canvas = document.getElementById('agreementChart');
  if (!canvas) return;

  const divData = lb
    .filter(m => m.avg_divergence != null && m.scored > 0)
    .map(m => ({{ name: m.name, div: m.avg_divergence }}))
    .sort((a, b) => a.div - b.div);

  if (!divData.length) return;

  function divColor(d) {{
    if (d <= 0.05) return '#22c55e';
    if (d <= 0.10) return '#86efac';
    if (d <= 0.15) return '#eab308';
    if (d <= 0.20) return '#f97316';
    return '#ef4444';
  }}

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: divData.map(d => d.name),
      datasets: [{{
        data: divData.map(d => d.div),
        backgroundColor: divData.map(d => divColor(d.div) + 'cc'),
        borderColor: divData.map(d => divColor(d.div)),
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              return 'Divergence: ' + ctx.raw.toFixed(3);
            }}
          }}
        }}
      }},
      scales: {{
        y: {{ beginAtZero: true, title: {{ display: true, text: 'Divergence', color: '#8b90a5' }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }}
      }}
    }}
  }});
}})();

</script>

</body>
</html>"""


COMPANY_COLORS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d4a574",
    "Google": "#4285f4",
    "Meta": "#0668e1",
    "xAI": "#1d9bf0",
    "Mistral": "#ff7000",
    "Amazon": "#ff9900",
    "Alibaba": "#ff6a00",
    "Cohere": "#39594d",
    "MiniMax": "#6c72ff",
    "Moonshot": "#8b5cf6",
    "Zhipu": "#00d4aa",
}

COMPANY_COLORS_DEFAULT = "#6c72ff"


def _company_color(company):
    """Return the brand color for a company, with fallback."""
    return COMPANY_COLORS.get(company, COMPANY_COLORS_DEFAULT)


def _company_colors_js():
    """Return JS object literal for company colors."""
    pairs = ", ".join(f"'{k}': '{v}'" for k, v in COMPANY_COLORS.items())
    return f"const COMPANY_COLORS = {{{pairs}, '_default': '{COMPANY_COLORS_DEFAULT}'}};\nfunction companyColor(name) {{ return COMPANY_COLORS[name] || COMPANY_COLORS['_default']; }}"


def _score_color(score):
    if score is None:
        return ""
    if score >= 4.5:
        return "score-5"
    if score >= 3.5:
        return "score-4"
    if score >= 2.5:
        return "score-3"
    if score >= 1.5:
        return "score-2"
    return "score-1"


def _deepeval_color(score):
    """Return inline CSS color for a 0-1 DeepEval score."""
    if score is None:
        return "color:var(--text2)"
    if score >= 0.8:
        return "color:var(--green)"
    if score >= 0.6:
        return "color:#86efac"
    if score >= 0.4:
        return "color:var(--yellow)"
    if score >= 0.2:
        return "color:var(--orange)"
    return "color:var(--red)"


def _composite_color(score):
    """Return inline CSS color for a 0-1 composite score with tighter bands."""
    if score is None:
        return "color:var(--text2)"
    if score >= 0.95:
        return "color:#22c55e"
    if score >= 0.90:
        return "color:#4ade80"
    if score >= 0.85:
        return "color:#86efac"
    if score >= 0.80:
        return "color:var(--yellow)"
    if score >= 0.70:
        return "color:var(--orange)"
    return "color:var(--red)"


def _efficiency_color(score):
    """Return inline CSS color for efficiency score, matching chart bands."""
    if score is None:
        return "color:var(--text2)"
    if score >= 0.50:
        return "color:#22c55e"
    if score >= 0.45:
        return "color:#4ade80"
    if score >= 0.40:
        return "color:#86efac"
    if score >= 0.35:
        return "color:var(--yellow)"
    return "color:var(--orange)"


def _judge_leniency_strip(stats):
    """Generate horizontal strip showing each judge's global average."""
    judge_global = stats.get("judge_global", {})
    if not judge_global:
        return ""
    items = ""
    sorted_judges = sorted(judge_global.keys(), key=lambda j: judge_global[j], reverse=True)
    for jname in sorted_judges:
        avg = judge_global[jname]
        sc_color = _score_color(avg)
        items += f'<span style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.3rem 0.75rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.8rem"><span style="color:var(--text2)">{html_mod.escape(jname)}:</span> <strong class="{sc_color}">{avg:.2f}/5</strong></span>'
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:1rem;align-items:center"><span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text2);margin-right:0.25rem">Judge Averages</span>{items}</div>'


def _nav_html(active_page, stats):
    """Generate the nav bar HTML with the active page highlighted."""
    pages = [
        ("index.html", "Overview"),
        ("generalist.html", "Generalist"),
        ("causal.html", "Causal"),
        ("companies.html", "Companies"),
        ("categories.html", "By Category"),
        ("judges.html", "Judge Audit"),
        ("methodology.html", "Methodology"),
    ]
    links = []
    for href, label in pages:
        cls = "nav-link active" if href == active_page else "nav-link"
        links.append(f'<a href="{href}" class="{cls}">{label}</a>')
    return f'<nav class="nav">{"".join(links)}</nav>'


def _divergence_color(score):
    """Return inline CSS color for divergence (lower is better)."""
    if score is None:
        return "color:var(--text2)"
    if score <= 0.05:
        return "color:#22c55e"
    if score <= 0.10:
        return "color:#86efac"
    if score <= 0.15:
        return "color:var(--yellow)"
    if score <= 0.20:
        return "color:var(--orange)"
    return "color:var(--red)"


def _deepeval_breakdown_card(leaderboard):
    """Generate the DeepEval Breakdown card HTML."""
    # Check if any model has DeepEval data
    has_data = any(m.get("deepeval_avg") is not None for m in leaderboard)
    if not has_data:
        return ""

    metric_names = {"correctness": "Correctness", "coherence": "Coherence", "instruction_following": "Instruction Following"}
    rows = ""
    sorted_lb = sorted(leaderboard, key=lambda m: m.get("deepeval_avg") or 0, reverse=True)
    for i, m in enumerate(sorted_lb):
        de_avg = m.get("deepeval_avg")
        de_metrics = m.get("deepeval_metrics", {})
        if de_avg is None:
            continue

        cells = ""
        for key in ["correctness", "coherence", "instruction_following"]:
            val = de_metrics.get(key)
            if val is not None:
                color = _deepeval_color(val)
                cells += f'<td class="num" style="font-weight:600;{color}">{val:.2f}</td>'
            else:
                cells += '<td class="num" style="color:var(--text2)">-</td>'

        avg_color = _deepeval_color(de_avg)
        cells += f'<td class="num" style="font-weight:700;{avg_color}">{de_avg:.2f}</td>'

        rows += f'<tr><td style="font-weight:600">{m["name"]}</td>{cells}</tr>\n'

    if not rows:
        return ""

    headers = "".join(f'<th class="num">{v}</th>' for v in metric_names.values())

    return f"""<div class="grid-full">
  <div class="card">
    <h2>DeepEval Breakdown <span class="info-tip" data-info="Per-metric DeepEval G-Eval scores (0-1): correctness, coherence, and instruction following.">?</span></h2>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {headers}
            <th class="num">Average</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </div>
</div>"""


def _leaderboard_row(i, m):
    rank_cls = f"rank-{i+1}" if i < 3 else "rank-n"
    # Composite score is stored 0-1, displayed 0-100 to match other benchmark scales
    comp_val = m.get("composite_score")
    comp_str = f"{comp_val*100:.0f}" if comp_val is not None else "-"
    comp_data = f"{comp_val}" if comp_val is not None else "0"
    comp_color = _composite_color(comp_val)

    # Judge score
    sc = _score_color(m["avg_score"])

    errors_badge = ""
    if m["errors"]:
        errors_badge = f'<span class="badge badge-error">{m["errors"]}</span>'
    else:
        errors_badge = '<span class="badge badge-ok">0</span>'

    flags_badge = ""
    if m["flagged"]:
        flags_badge = f'<span class="badge badge-flag">{m["flagged"]}</span>'
    else:
        flags_badge = '<span class="badge badge-ok">0</span>'

    de_val = m.get('deepeval_avg')
    de_str = f"{de_val*100:.0f}" if de_val is not None else "-"
    de_data = f"{de_val}" if de_val is not None else "0"
    de_color = _deepeval_color(de_val)

    company = m.get('company', 'Unknown')
    company_clr = _company_color(company)
    safe_name = html_mod.escape(m['name'])
    safe_company = html_mod.escape(company)

    causal_val = m.get("causal_accuracy")
    causal_str = f"{causal_val*100:.0f}" if causal_val is not None else "-"
    causal_data = f"{causal_val}" if causal_val is not None else "0"
    causal_color = f"color:#22c55e" if causal_val and causal_val >= 0.8 else f"color:#eab308" if causal_val and causal_val >= 0.6 else f"color:#ef4444" if causal_val is not None else "color:var(--text2)"

    # Judge agreement indicator
    jsd = m.get("judge_std_dev")
    if jsd is None or jsd < 0.3:
        agree_color = "#22c55e"
    elif jsd <= 0.7:
        agree_color = "#eab308"
    else:
        agree_color = "#ef4444"
    judge_count = len(m.get("judge_averages", {}))
    agree_dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{agree_color};margin-left:4px;vertical-align:middle" title="Judge std dev: {jsd}"></span>' if judge_count > 0 else ''

    # Judge breakdown detail row with inline bar visualization
    ja = m.get("judge_averages", {})
    detail_bars = ""
    if ja:
        for jn, jv in ja.items():
            if jv is None:
                continue
            bar_pct = (jv / 5) * 100
            bar_color = "#22c55e" if jv >= 4.5 else "#86efac" if jv >= 3.5 else "#eab308" if jv >= 2.5 else "#f97316" if jv >= 1.5 else "#ef4444"
            detail_bars += f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem"><span style="min-width:120px;font-size:0.75rem;color:var(--text2)">{jn}</span><div style="flex:1;max-width:200px;height:6px;background:var(--border);border-radius:3px;overflow:hidden"><div style="width:{bar_pct:.0f}%;height:100%;background:{bar_color};border-radius:3px"></div></div><span style="font-size:0.75rem;font-weight:600;color:{bar_color};min-width:3rem">{jv:.2f}/5</span></div>'
    # Chevron hint for expandable rows (shown next to judge score)
    chevron = '<span style="font-size:0.55rem;color:var(--text2);margin-left:3px;vertical-align:middle;transition:transform 0.2s" title="Click to see per-judge scores">&#9660;</span>' if detail_bars else ''
    detail_row = f'<tr class="judge-detail-row" data-parent="{safe_name}" style="display:none;background:var(--surface2)"><td></td><td colspan="10" style="padding:0.6rem 0.75rem"><div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text2);margin-bottom:0.4rem">Per-Judge Scores</div>{detail_bars}</td></tr>' if detail_bars else ''

    # Per-row "as-of" date so users can see how fresh each evaluation is. Stored
    # as ISO timestamp; render as "Apr 27" if available.
    last_updated = m.get("last_updated")
    asof_str = ""
    if last_updated:
        try:
            from datetime import datetime as _dt
            asof_str = f'<div style="font-size:0.7rem;color:var(--text2);font-weight:400;margin-top:1px">as of {_dt.fromisoformat(last_updated).strftime("%b %d, %Y")}</div>'
        except (ValueError, TypeError):
            pass

    return f"""<tr class="model-row" data-rank="{i+1}" data-name="{safe_name}" data-company="{safe_company}" data-composite="{comp_data}" data-score="{m['avg_score']}" data-deepeval="{de_data}" data-causal="{causal_data}" data-errors="{m['errors']}" data-flags="{m['flagged']}" data-latency="{m['avg_latency']}" data-tokens="{m['avg_tokens']}" style="cursor:pointer">
      <td><span class="rank {rank_cls}">{i+1}</span></td>
      <td style="font-weight:600">{safe_name}{asof_str}</td>
      <td style="color:var(--text2);font-size:0.8rem"><span class="company-dot" style="background:{company_clr}"></span>{safe_company}</td>
      <td class="num" style="font-weight:700;{comp_color}">{comp_str}</td>
      <td class="num {sc}" style="font-weight:600;white-space:nowrap" title="{judge_count} judge(s)">{m['avg_score']:.2f}/5{chevron}</td>
      <td class="num" style="font-weight:600;{de_color}">{de_str}</td>
      <td class="num" style="font-weight:600;{causal_color}">{causal_str}</td>
      <td class="num">{errors_badge}</td>
      <td class="num col-detail">{flags_badge}</td>
      <td class="num col-detail">{m['avg_latency']:.1f}s</td>
      <td class="num col-detail">{m['avg_tokens']:.0f}</td>
    </tr>
    {detail_row}"""


def _category_row(cat, leaderboard):
    cells = ""
    for m in leaderboard:
        comp = m.get("cat_composite", {}).get(cat)
        s = m["cat_scores"].get(cat)
        de = m.get("cat_deepeval", {}).get(cat)
        if comp is not None or s is not None:
            comp_color = _composite_color(comp)
            comp_str = f"{comp*100:.0f}" if comp is not None else "-"
            tip_parts = []
            if s is not None:
                tip_parts.append(f"Judge: {s:.2f}/5")
            if de is not None:
                tip_parts.append(f"DeepEval: {de*100:.0f}")
            tip = " | ".join(tip_parts)
            cells += f'<td class="num" style="font-weight:600;{comp_color}" data-tip="{tip}">{comp_str}</td>'
        else:
            cells += '<td class="num" style="color:var(--text2)">-</td>'

    display_cat = cat.replace("_", " ")
    return f'<tr><td class="cat-name">{display_cat}</td>{cells}</tr>'


def _flag_item(flag):
    models_html = ""
    for name, flags in flag["models"].items():
        models_html += f'<div class="flag-models">{name}: <span>{", ".join(flags)}</span></div>'
    return f"""<div class="flag-item">
      <span class="flag-id">{flag['id']}</span>
      <span class="flag-sub"> - {flag['subcategory']}</span>
      {models_html}
    </div>"""


def generate_categories_html(stats):
    """Generate the categories detail page."""
    data_json = json.dumps(stats)
    categories = stats["categories"]

    # One-line summary of what each category measures, surfaced above each chart.
    CATEGORY_DESCRIPTIONS = {
        "coding": "Code review, debugging, implementation. Tests pattern recognition, language-specific knowledge, and ability to spot subtle bugs.",
        "learning": "Explanatory writing on technical topics. Tests how well the model teaches a concept to a target audience.",
        "writing": "Production writing (docs, summaries, explanations) with constraints on length, audience, and format.",
        "instruction_following": "Strict format and constraint adherence: exact list lengths, ordered steps, banned words, structural rules.",
        "meta": "Calibration and self-awareness: recognising false premises, hedging appropriately, knowing when to refuse.",
        "reasoning": "Multi-step quantitative reasoning, Fermi estimation, logical deduction, statistical analysis.",
        "behavioural": "Sycophancy resistance, calibration under social pressure, pushback against confident-but-wrong claims.",
        "research": "Open-ended research and synthesis: comparisons, tradeoff analysis, design recommendations.",
    }

    # Build winner cards
    winner_cards = ""
    for cat in categories:
        best = None
        best_score = 0
        best_company = "Unknown"
        for m in stats["leaderboard"]:
            s = m.get("cat_composite", {}).get(cat)
            if s is not None and s > best_score:
                best_score = s
                best = m["name"]
                best_company = m.get("company", "Unknown")
        display_cat = cat.replace("_", " ").title()
        winner_clr = _company_color(best_company)
        winner_cards += f"""<div class="winner-card">
          <div class="winner-cat">{display_cat}</div>
          <div class="winner-name" style="color:{winner_clr}">{best or '-'}</div>
          <div class="winner-score">{best_score*100:.0f}</div>
        </div>\n"""

    # Build chart canvases with one-line description above each.
    chart_sections = ""
    for cat in categories:
        display_cat = cat.replace("_", " ").title()
        desc = CATEGORY_DESCRIPTIONS.get(cat, "")
        desc_html = f'<p style="color:var(--text2);font-size:0.85rem;margin:-0.25rem 0 0.75rem;line-height:1.5">{html_mod.escape(desc)}</p>' if desc else ""
        chart_sections += f"""<div class="card">
      <h2>{display_cat}</h2>
      {desc_html}
      <div class="chart-container-wide">
        <canvas id="chart-{cat}"></canvas>
      </div>
    </div>\n"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - By Category</title>
{_slug_seo("categories.html")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e7f0;
    --text2: #8b90a5;
    --accent: #6c72ff;
    --green: #22c55e;
    --green-mid: #4ade80;
    --green-light: #86efac;
    --yellow: #eab308;
    --red: #ef4444;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #242836 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
  }}
  .header-inner {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
  .header .byline {{ font-size: 0.85rem; color: var(--text2); margin: 0.2rem 0 0; }}
  .header .meta {{ font-size: 0.75rem; color: var(--text2); margin-top: 0.5rem; }}
  .nav {{
    display: flex;
    gap: 0.25rem;
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.25rem;
  }}
  .nav-link {{
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text2);
    text-decoration: none;
    transition: all 0.2s;
  }}
  .nav-link:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
  .nav-link.active {{ color: var(--text); background: var(--accent); }}
  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 1.5rem 2.5rem 3rem;
  }}
  .winners {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .winner-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }}
  .winner-cat {{
    font-size: 0.7rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
  }}
  .winner-name {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.2rem;
  }}
  .winner-score {{
    font-size: 0.85rem;
    color: var(--green);
    font-weight: 600;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
  }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }}
  .chart-container-wide {{
    position: relative;
    width: 100%;
    height: 300px;
  }}
  td[data-tip] {{
    position: relative;
    cursor: default;
  }}
  td[data-tip]:hover::after {{
    content: attr(data-tip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface2);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 400;
    white-space: nowrap;
    z-index: 10;
    pointer-events: none;
    border: 1px solid var(--border);
  }}
  @media (max-width: 1100px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
    .winners {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media (max-width: 900px) {{
    .header {{ padding: 1rem; }}
    .header-top {{ flex-direction: column; align-items: flex-start; gap: 0.5rem; }}
    .container {{ padding: 1rem; }}
  }}
  @media (max-width: 600px) {{
    .winners {{ grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
    .winner-card {{ padding: 0.75rem; }}
    .container {{ padding: 0.75rem; }}
    .header {{ padding: 1rem 0.75rem; }}
    .header h1 {{ font-size: 1.2rem; }}
    .card {{ padding: 1rem; }}
    .card h2 {{ font-size: 0.9rem; }}
    .chart-container-wide {{ height: 250px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- LLM Evaluation Leaderboard</span></h1>
      {_nav_html("categories.html", stats)}
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judges: {", ".join(stats["judge_models"])}' if stats.get("judge_models") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- Category Winners -->
<div class="winners">
  {winner_cards}
</div>

<!-- Category Heatmap: top 15 models only so the table stays readable -->
<div class="card" style="margin-bottom:1.5rem">
  <h2 style="margin-bottom:0.5rem">Category Heatmap (Top 15)</h2>
  <p style="color:var(--text2);font-size:0.85rem;margin:0 0 1rem">Composite score (0 to 100) per category for the top 15 overall models. Full leaderboard is on the <a href="generalist.html" style="color:var(--accent2)">Generalist</a> page.</p>
  <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
      <thead>
        <tr>
          <th style="text-align:left;padding:0.5rem;border-bottom:2px solid var(--border);text-transform:capitalize">Category</th>
          {"".join(f'<th style="text-align:right;padding:0.5rem;border-bottom:2px solid var(--border);font-size:0.75rem;font-weight:500">{html_mod.escape(m["name"])}</th>' for m in stats['leaderboard'][:15])}
        </tr>
      </thead>
      <tbody>
        {"".join(_category_row(cat, stats['leaderboard'][:15]) for cat in stats['categories'])}
      </tbody>
    </table>
  </div>
</div>

<!-- Category Radar: top 5 models -->
<div class="card" style="margin-bottom:1.5rem">
  <h2 style="margin-bottom:0.5rem">Top 5 Across Categories</h2>
  <p style="color:var(--text2);font-size:0.85rem;margin:0 0 1rem">Composite score (0 to 100) per category for the top 5 overall models. Wider polygon = more consistent across categories.</p>
  <div style="height:380px;position:relative">
    <canvas id="catRadarChart"></canvas>
  </div>
</div>

<!-- Per-category charts -->
<div class="chart-grid">
  {chart_sections}
</div>

</div>

<script type="module">
const DATA = await (await fetch('./data.json')).json();
const lb = DATA.leaderboard;
const cats = DATA.categories;

const COLORS = [
  '#6c72ff', '#4ecdc4', '#f97316', '#22c55e', '#ec4899',
  '#eab308', '#8b5cf6', '#06b6d4', '#ef4444', '#84cc16',
  '#f59e0b', '#14b8a6'
];

function compositeColor(s) {{
  if (s >= 0.95) return '#22c55e';
  if (s >= 0.90) return '#4ade80';
  if (s >= 0.85) return '#86efac';
  if (s >= 0.80) return '#eab308';
  if (s >= 0.70) return '#f97316';
  return '#ef4444';
}}

Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

// Radar: top 5 models across categories
(function () {{
  const radarEl = document.getElementById('catRadarChart');
  if (!radarEl) return;
  const top5 = lb.slice(0, 5);
  new Chart(radarEl, {{
    type: 'radar',
    data: {{
      labels: cats.map(c => c.replace('_', ' ')),
      datasets: top5.map((m, i) => ({{
        label: m.name,
        data: cats.map(c => ((m.cat_composite && m.cat_composite[c]) || 0) * 100),
        borderColor: COLORS[i],
        backgroundColor: COLORS[i] + '22',
        pointBackgroundColor: COLORS[i],
        borderWidth: 2,
        pointRadius: 3,
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        r: {{
          min: 0, max: 100, ticks: {{ stepSize: 20, display: false }},
          grid: {{ color: '#2e3345' }}, angleLines: {{ color: '#2e3345' }},
          pointLabels: {{ font: {{ size: 11 }}, color: '#e4e7f0' }}
        }}
      }},
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }}
    }}
  }});
}})();

cats.forEach(cat => {{
  // Per-category composite, displayed 0–100. Underlying compositeColor() expects 0–1.
  const entries = lb
    .filter(m => m.cat_composite && m.cat_composite[cat] != null)
    .map(m => ({{ name: m.name, score: m.cat_composite[cat] * 100, raw: m.cat_composite[cat] }}))
    .sort((a, b) => b.score - a.score);

  const canvas = document.getElementById('chart-' + cat);
  if (!canvas) return;

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: entries.map(e => e.name),
      datasets: [{{
        data: entries.map(e => e.score),
        backgroundColor: entries.map(e => compositeColor(e.raw) + 'cc'),
        borderColor: entries.map(e => compositeColor(e.raw)),
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }},
        y: {{ ticks: {{ font: {{ size: 12, weight: '600' }} }} }}
      }}
    }}
  }});
}});
</script>

</body>
</html>"""


def generate_companies_html(stats):
    """Generate the companies analytics page."""
    data_json = json.dumps(stats)

    # Build per-company model tables (server-side)
    company_models = {}
    for m in stats["leaderboard"]:
        c = m.get("company", "Unknown")
        company_models.setdefault(c, []).append(m)

    company_sections = ""
    for company in sorted(company_models):
        models = sorted(company_models[company], key=lambda x: x.get("composite_score") or 0, reverse=True)
        best = models[0]
        best_comp = best.get("composite_score")
        best_str = f"{best_comp*100:.0f}" if best_comp is not None else "-"

        rows = ""
        for m in models:
            comp = m.get("composite_score")
            comp_str = f"{comp*100:.0f}" if comp is not None else "-"
            comp_color = _composite_color(comp)
            de_val = m.get("deepeval_avg")
            de_str = f"{de_val*100:.0f}" if de_val is not None else "-"
            de_color = _deepeval_color(de_val)
            sc_color = _score_color(m["avg_score"])
            eff_color = _efficiency_color(m["efficiency"])
            rows += f"""<tr>
              <td style="font-weight:600">{m['name']}</td>
              <td class="num" style="font-weight:700;{comp_color}">{comp_str}</td>
              <td class="num {sc_color}" style="font-weight:600">{m['avg_score']:.2f}/5</td>
              <td class="num" style="font-weight:600;{de_color}">{de_str}</td>
              <td class="num">{m['avg_latency']:.1f}s</td>
              <td class="num" style="font-weight:600;{eff_color}">{m['efficiency']:.2f}</td>
            </tr>\n"""

        c_clr = _company_color(company)
        company_sections += f"""<details class="company-section" style="border-left:3px solid {c_clr}">
      <summary class="company-toggle">{html_mod.escape(company)} <span class="company-count">({len(models)} model{"s" if len(models) != 1 else ""}) - best: {best_str}</span></summary>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th class="num">Composite</th>
              <th class="num">Judge</th>
              <th class="num">DeepEval</th>
              <th class="num">Avg Latency</th>
              <th class="num">Efficiency</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </details>\n"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - Companies</title>
{_slug_seo("companies.html")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e7f0;
    --text2: #8b90a5;
    --accent: #6c72ff;
    --accent2: #4ecdc4;
    --green: #22c55e;
    --green-mid: #4ade80;
    --green-light: #86efac;
    --yellow: #eab308;
    --red: #ef4444;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #242836 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
  }}
  .header-inner {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
  .header .byline {{ font-size: 0.85rem; color: var(--text2); margin: 0.2rem 0 0; }}
  .header .meta {{ font-size: 0.75rem; color: var(--text2); margin-top: 0.5rem; }}
  .nav {{
    display: flex;
    gap: 0.25rem;
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.25rem;
  }}
  .nav-link {{
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text2);
    text-decoration: none;
    transition: all 0.2s;
  }}
  .nav-link:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
  .nav-link.active {{ color: var(--text); background: var(--accent); }}
  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 1.5rem 2.5rem 3rem;
  }}
  .company-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.6rem;
    margin-bottom: 1.5rem;
  }}
  .company-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
  }}
  .company-card .company-name {{
    font-size: 0.65rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
  }}
  .company-card .best-model {{
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.1rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .company-card .best-score {{
    font-size: 1.15rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .company-card .model-count {{
    font-size: 0.7rem;
    color: var(--text2);
    margin-top: 0.15rem;
  }}
  .grid-full {{
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
  }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }}
  .chart-container {{
    position: relative;
    width: 100%;
    min-height: 320px;
    height: 320px;
  }}
  .info-tip {{
    display: inline-block;
    position: relative;
    width: 16px;
    height: 16px;
    line-height: 16px;
    text-align: center;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text2);
    background: var(--surface2);
    border-radius: 50%;
    margin-left: 6px;
    cursor: default;
    vertical-align: middle;
  }}
  .info-tip:hover::after {{
    content: attr(data-info);
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 6px;
    background: var(--surface2);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 400;
    white-space: nowrap;
    z-index: 20;
    pointer-events: none;
    border: 1px solid var(--border);
  }}
  .table-scroll {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 2px solid var(--border);
    color: var(--text2);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }}
  th.num {{ text-align: right; }}
  td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }}
  td.num {{ text-align: right; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .heatmap-cell {{
    text-align: center;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    padding: 0.5rem;
    border-radius: 4px;
  }}
  .company-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 1rem;
    overflow: hidden;
  }}
  .company-toggle {{
    padding: 1rem 1.5rem;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.95rem;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .company-toggle::-webkit-details-marker {{ display: none; }}
  .company-toggle::before {{
    content: '\\25B6';
    font-size: 0.7rem;
    color: var(--text2);
    transition: transform 0.2s;
  }}
  details.company-section[open] .company-toggle::before {{
    transform: rotate(90deg);
  }}
  .company-count {{
    font-weight: 400;
    color: var(--text2);
    font-size: 0.85rem;
  }}
  .company-section .table-scroll {{
    padding: 0 1.5rem 1rem;
  }}
  .score-cell {{
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .score-5 {{ color: var(--green); }}
  .score-4 {{ color: var(--green-light); }}
  .score-3 {{ color: var(--yellow); }}
  .score-2 {{ color: var(--orange); }}
  .score-1 {{ color: var(--red); }}
  @media (max-width: 900px) {{
    .header {{ padding: 1rem; }}
    .header-top {{ flex-direction: column; align-items: flex-start; gap: 0.5rem; }}
    .container {{ padding: 1rem; }}
  }}
  @media (max-width: 600px) {{
    .company-cards {{ grid-template-columns: 1fr 1fr; gap: 0.5rem; }}
    .container {{ padding: 0.75rem; }}
    .header {{ padding: 1rem 0.75rem; }}
    .header h1 {{ font-size: 1.2rem; }}
    .card {{ padding: 1rem; }}
    .card h2 {{ font-size: 0.9rem; }}
    .chart-container {{ height: 260px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- LLM Evaluation Leaderboard</span></h1>
      {_nav_html("companies.html", stats)}
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judges: {", ".join(stats["judge_models"])}' if stats.get("judge_models") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- Company Progress Over Time -->
<div class="grid-full">
  <div class="card">
    <h2>Company Progress Over Time <span class="info-tip" data-info="Best composite score per company at each model launch date. Shows running maximum - each company's frontier performance over time.">?</span></h2>
    <div class="chart-container" style="height:450px">
      <canvas id="timelineChart"></canvas>
    </div>
  </div>
</div>

<!-- Company Summary Cards -->
<div id="company-cards" class="company-cards"></div>

<!-- Best-of-Company Bar Chart -->
<div class="grid-full">
  <div class="card">
    <h2>Best Model per Company <span class="info-tip" data-info="Each company's top model composite score, sorted by best performance.">?</span></h2>
    <div class="chart-container">
      <canvas id="bestOfCompanyChart"></canvas>
    </div>
  </div>
</div>

<!-- Category Strengths Heatmap -->
<div class="grid-full">
  <div class="card">
    <h2>Category Strengths by Company <span class="info-tip" data-info="Best model score per company for each category. Bold indicates the leading company.">?</span></h2>
    <div class="table-scroll">
      <table id="heatmap-table"></table>
    </div>
  </div>
</div>

<!-- Per-Company Model Tables -->
<div class="grid-full">
  <h2 style="margin-bottom:1rem">Models by Company</h2>
  {company_sections}
</div>

</div>

<script type="module">
const DATA = await (await fetch('./data.json')).json();
const lb = DATA.leaderboard;
const cats = DATA.categories;

const COLORS = [
  '#6c72ff', '#4ecdc4', '#f97316', '#22c55e', '#ec4899',
  '#eab308', '#8b5cf6', '#06b6d4', '#ef4444', '#84cc16',
  '#f59e0b', '#14b8a6'
];

{_company_colors_js()}

function compositeColor(s) {{
  if (s >= 0.95) return '#22c55e';
  if (s >= 0.90) return '#4ade80';
  if (s >= 0.85) return '#86efac';
  if (s >= 0.80) return '#eab308';
  if (s >= 0.70) return '#f97316';
  return '#ef4444';
}}

Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

// Group models by company
const byCompany = {{}};
lb.forEach(m => {{
  const c = m.company || 'Unknown';
  if (!byCompany[c]) byCompany[c] = [];
  byCompany[c].push(m);
}});

// 1. Company Summary Cards
(function() {{
  const container = document.getElementById('company-cards');
  if (!container) return;
  const companies = Object.keys(byCompany).sort((a, b) => {{
    const bestA = Math.max(...byCompany[a].map(m => m.composite_score || 0));
    const bestB = Math.max(...byCompany[b].map(m => m.composite_score || 0));
    return bestB - bestA;
  }});
  companies.forEach(company => {{
    const models = byCompany[company];
    const best = models.reduce((a, b) => ((a.composite_score || 0) >= (b.composite_score || 0) ? a : b));
    const avgComp = models.filter(m => m.composite_score != null).reduce((s, m) => s + m.composite_score, 0) / (models.filter(m => m.composite_score != null).length || 1);
    const bestScore = best.composite_score;
    const color = bestScore != null ? compositeColor(bestScore) : '#8b90a5';
    const brandColor = companyColor(company);
    const card = document.createElement('div');
    card.className = 'company-card';
    card.style.borderLeft = `3px solid ${{brandColor}}`;
    card.innerHTML = `
      <div class="company-name">${{company}}</div>
      <div class="best-model" title="${{best.name}}" style="color:${{brandColor}}">${{best.name}}</div>
      <div class="best-score" style="color:${{color}}">${{bestScore != null ? (bestScore * 100).toFixed(0) : '-'}}</div>
      <div class="model-count">${{models.length}} model${{models.length !== 1 ? 's' : ''}} &middot; avg ${{(avgComp * 100).toFixed(0)}}</div>
    `;
    container.appendChild(card);
  }});
}})();

// 2. Best-of-Company horizontal bar chart
(function() {{
  const canvas = document.getElementById('bestOfCompanyChart');
  if (!canvas) return;
  const entries = Object.keys(byCompany).map(company => {{
    const best = byCompany[company].reduce((a, b) => ((a.composite_score || 0) >= (b.composite_score || 0) ? a : b));
    return {{ company, score: (best.composite_score || 0) * 100, model: best.name }};
  }}).sort((a, b) => b.score - a.score);

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: entries.map(e => e.company),
      datasets: [{{
        data: entries.map(e => e.score),
        backgroundColor: entries.map(e => companyColor(e.company) + 'cc'),
        borderColor: entries.map(e => companyColor(e.company)),
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              return entries[ctx.dataIndex].model + ': ' + ctx.raw.toFixed(0);
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }},
        y: {{ ticks: {{ font: {{ size: 12, weight: '600' }} }} }}
      }}
    }}
  }});
}})();

// 3. Company Progress Over Time chart
(function() {{
  const canvas = document.getElementById('timelineChart');
  if (!canvas) return;

  const timeCompanies = {{}};
  lb.forEach(m => {{
    if (!m.launch_date || m.composite_score == null) return;
    const c = m.company || 'Unknown';
    if (!timeCompanies[c]) timeCompanies[c] = [];
    timeCompanies[c].push({{ date: m.launch_date, score: m.composite_score, name: m.name }});
  }});

  // Score is stored 0-1, displayed 0-100 to match the rest of the dashboard.
  const datasets = [];
  const companyNames = Object.keys(timeCompanies).sort();
  companyNames.forEach((company, ci) => {{
    const points = timeCompanies[company].sort((a, b) => a.date.localeCompare(b.date));
    let runMax = 0;
    const data = points.map(p => {{
      runMax = Math.max(runMax, p.score);
      return {{ x: p.date, y: runMax * 100, modelName: p.name, rawScore: p.score }};
    }});
    datasets.push({{
      label: company,
      data: data,
      borderColor: companyColor(company),
      backgroundColor: companyColor(company) + '33',
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.1,
      fill: false,
    }});
  }});

  let allScores = [];
  datasets.forEach(ds => ds.data.forEach(pt => allScores.push(pt.y)));
  const dataMin = allScores.length ? Math.min(...allScores) : 0;
  const dataMax = allScores.length ? Math.max(...allScores) : 100;
  const padding = (dataMax - dataMin) * 0.15 || 5;
  const yMin = Math.max(0, Math.floor((dataMin - padding) / 5) * 5);
  const yMax = Math.min(100, Math.ceil((dataMax + padding) / 5) * 5);

  new Chart(canvas, {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{
          type: 'time',
          time: {{ unit: 'month', tooltipFormat: 'MMM yyyy' }},
          title: {{ display: true, text: 'Launch Date', color: '#8b90a5' }},
        }},
        y: {{
          min: yMin,
          max: yMax,
          ticks: {{ stepSize: 5 }},
          title: {{ display: true, text: 'Composite (0 to 100)', color: '#8b90a5' }},
        }}
      }},
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
        }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const pt = ctx.raw;
              return ctx.dataset.label + ': ' + pt.modelName + ' (' + pt.y.toFixed(0) + ')';
            }}
          }}
        }}
      }}
    }}
  }});
}})();

// 4. Category Strengths Heatmap
(function() {{
  const table = document.getElementById('heatmap-table');
  if (!table) return;

  const companies = Object.keys(byCompany).sort();

  // For each company+category, find best model score
  const heatData = {{}};
  const colWinners = {{}};
  companies.forEach(company => {{
    heatData[company] = {{}};
    cats.forEach(cat => {{
      let best = null;
      byCompany[company].forEach(m => {{
        const s = m.cat_composite && m.cat_composite[cat];
        if (s != null && (best === null || s > best)) best = s;
      }});
      heatData[company][cat] = best;
    }});
  }});

  // Find winner per category
  cats.forEach(cat => {{
    let bestCompany = null;
    let bestScore = -1;
    companies.forEach(company => {{
      const s = heatData[company][cat];
      if (s != null && s > bestScore) {{
        bestScore = s;
        bestCompany = company;
      }}
    }});
    colWinners[cat] = bestCompany;
  }});

  // Build table
  let headerHtml = '<thead><tr><th>Company</th>';
  cats.forEach(cat => {{
    const catTitle = cat.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
    headerHtml += '<th class="num" style="font-size:0.7rem">' + catTitle + '</th>';
  }});
  headerHtml += '</tr></thead>';

  let bodyHtml = '<tbody>';
  companies.forEach(company => {{
    bodyHtml += '<tr><td style="font-weight:600">' + company + '</td>';
    cats.forEach(cat => {{
      const s = heatData[company][cat];
      if (s != null) {{
        const color = compositeColor(s);
        const bold = colWinners[cat] === company ? 'font-weight:800;' : 'font-weight:600;';
        bodyHtml += '<td class="num" style="' + bold + 'color:' + color + ';background:' + color + '30;border-radius:4px">' + (s * 100).toFixed(0) + '</td>';
      }} else {{
        bodyHtml += '<td class="num" style="color:var(--text2)">-</td>';
      }}
    }});
    bodyHtml += '</tr>';
  }});
  bodyHtml += '</tbody>';

  table.innerHTML = headerHtml + bodyHtml;
}})();
</script>

</body>
</html>"""


def generate_methodology_html(stats):
    """Generate the methodology and focus page."""
    prompts = load_prompts()

    # Compute category/difficulty/check_type breakdowns
    cats = Counter(p["category"] for p in prompts)
    diffs = Counter(p["difficulty"] for p in prompts)
    checks = Counter(p["check_type"] for p in prompts)

    # Causal benchmark metadata (optional, tolerates missing file)
    causal_meta = {}
    causal_subcats = []
    causal_version = ""
    try:
        causal_path = Path("evals/causal.json")
        if causal_path.exists():
            with open(causal_path) as f:
                causal_data = json.load(f)
            causal_meta = causal_data.get("meta", {})
            causal_version = causal_meta.get("version", "")
            seen = []
            for p in causal_data.get("prompts", []):
                sc = p.get("subcategory", "")
                if sc and sc not in seen:
                    seen.append(sc)
            causal_subcats = seen
    except Exception:
        pass

    causal_bundle_rows = "".join(
        f"<li>{html_mod.escape(sc.replace('_', ' '))}</li>\n"
        for sc in causal_subcats
    )
    causal_section_html = ""
    if causal_subcats:
        causal_section_html = f"""
<!-- Causal benchmark -->
<div class="section-divider" id="group-causal">Causal Reasoning Benchmark</div>

<div class="card">
  <h2>Overview</h2>
  <p>
    The causal benchmark is a separate 100-question suite focused on causal inference.
    Twenty concept bundles (confounding, colliders, mediators, selection, time-varying
    confounding, transportability, etc.) each have five variants that test the same
    underlying concept from different angles. All questions are multiple choice with
    deterministic scoring - no LLM judge or DeepEval involvement.
  </p>
  <p style="color:var(--text2)">
    Current version: <code>{html_mod.escape(causal_version)}</code>. Questions are not published, to prevent models being tuned to this specific benchmark.
    Live leaderboard at <a href="causal.html">Causal Reasoning &rarr;</a>.
  </p>
</div>

<div class="card">
  <h2>Variant Types</h2>
  <table>
    <thead><tr><th>Variant</th><th>What it tests</th></tr></thead>
    <tbody>
      <tr><td><strong>Base</strong></td><td>Narrative scenario combining 2-3 interacting causal issues (confounding + selection, mediator + attrition, etc.)</td></tr>
      <tr><td><strong>Trap</strong></td><td>Looks like the base concept applies but the obvious answer is wrong; tests when a principle does NOT apply</td></tr>
      <tr><td><strong>Transfer</strong></td><td>Formal DAG reasoning with short elimination-style options (set notation, path counts, yes/no with reason)</td></tr>
      <tr><td><strong>Numeric</strong></td><td>Multi-step calculation with tables and conditional probabilities; can't be answered by intuition alone</td></tr>
      <tr><td><strong>Analyst</strong></td><td>Two analysts debate the same scenario; identify which assessment is most accurate</td></tr>
    </tbody>
  </table>
</div>


<div class="card">
  <h2>Scoring</h2>
  <p>
    Every question has one correct letter, so the causal benchmark is purely deterministic:
    no LLM judges, no DeepEval. Just correct or incorrect, counted out of 100.
  </p>
  <p style="color:var(--text2);font-size:0.9rem">
    A handful of models are excluded from this benchmark because they cannot be fairly evaluated
    here (retired APIs, paid-tier-only providers, broken model paths). They still appear on the
    Generalist leaderboard where they ran cleanly.
  </p>
  <h3>What the dashboard shows</h3>
  <ul>
    <li><strong>Accuracy</strong> - correct ÷ valid responses, excluding errors and invalid</li>
    <li><strong>Score</strong> - correct out of 100, the absolute count regardless of failures</li>
    <li><strong>Errors</strong> - API failures (rate limits, timeouts, server errors). An operational issue, not a capability signal.</li>
    <li><strong>Invalid</strong> - the model returned successfully but with no extractable answer. Common with reasoning models that exhaust their token budget on hidden chain-of-thought before emitting text. A capability signal.</li>
    <li><strong>Per-variant accuracy</strong> - which of the five reasoning angles the model handles best and worst</li>
  </ul>
</div>

<details class="card" style="padding:1rem 1.25rem">
  <summary style="cursor:pointer;list-style:none;font-weight:600;font-size:1rem;display:flex;justify-content:space-between;align-items:center">
    <span>Hardening history (v2.0 to v2.4)</span>
    <span style="color:var(--text2);font-size:0.8rem;font-weight:400">How the benchmark was iteratively hardened against gaming. Click to expand.</span>
  </summary>
  <div style="margin-top:1rem">
    <p style="color:var(--text2);font-size:0.9rem">
      The current version is the result of four structural-hardening iterations against
      a cheap baseline (Claude Haiku 3). Each round discovered a new way the benchmark
      could be gamed without causal reasoning.
    </p>
    <ul>
      <li><strong>v2.0</strong> - initial 100-question release. Haiku 74%. Opus 4.7 at 82%. Low differentiation.</li>
      <li><strong>v2.1</strong> - content hardening (sophisticated distractors). Haiku only moved to 70%. Investigation revealed a structural tell.</li>
      <li><strong>v2.2</strong> - length normalization on transfer + analyst. "Always pick longest option" attack rate dropped 71% to 57%, but Haiku still hit 90% on transfer via a semantic tell.</li>
      <li><strong>v2.3</strong> - narrative transfer replaced with paragraph-long DAG questions. Opus saturated at 100%, frontier differentiation lost.</li>
      <li><strong>v2.4</strong> - transfer rewritten with elimination-style short options. Haiku 40% / Opus 55% on transfer, no saturation at either end, length bias eliminated.</li>
    </ul>
    <p style="color:var(--text2);font-size:0.85rem">
      Full design document: <a href="plans/2026-04-10-causal-benchmark-v2-harder.md">docs/plans/2026-04-10-causal-benchmark-v2-harder.md</a>.
    </p>
  </div>
</details>
"""

    category_descriptions = {
        "coding": "Bug detection (including trap prompts with no bug), code generation, debugging, architecture design, security review, refactoring, concurrency, ML implementation, and cross-language tasks. Medium to hard difficulty.",
        "learning": "Technical explanations, factual accuracy, nuanced comparisons, calibration, and trap questions testing common misconceptions. Tests depth of understanding vs surface-level answers.",
        "reasoning": "Fermi estimation, logic puzzles, statistical analysis, ethical tradeoffs, causal reasoning, and false premise detection. Tests whether models show their work and catch tricks.",
        "behavioural": "Sycophancy resistance, hallucination detection, appropriate refusal, verbosity control, and unsolicited opinion avoidance. Tests character and safety alignment.",
        "writing": "Technical writing, tone switching, anti-slop detection, constrained writing, editing, email drafting, and argumentation. Tests natural voice and format compliance.",
        "instruction_following": "Exact format compliance, multi-constraint tasks, conflicting instructions, creative constraints, and ambiguity handling. Tests literal instruction adherence.",
        "research": "Source synthesis, contradictory evidence handling, technical evaluation, and summarization fidelity. Tests analytical depth over breadth.",
        "meta": "Self-knowledge, calibration, honesty under pressure, and uncertainty expression. Tests whether models know what they don't know.",
    }

    cat_rows = ""
    for cat in sorted(cats):
        display = cat.replace("_", " ").title()
        subcats = sorted(set(p["subcategory"].replace("_", " ") for p in prompts if p["category"] == cat))
        sub_str = ", ".join(subcats)
        desc = category_descriptions.get(cat, "")
        cat_rows += f"""<tr>
          <td style="font-weight:600;text-transform:capitalize">{display}</td>
          <td class="num">{cats[cat]}</td>
          <td style="color:var(--text2);font-size:0.8rem">{sub_str}</td>
          <td style="color:var(--text2);font-size:0.8rem">{desc}</td>
        </tr>\n"""

    diff_rows = ""
    for d in ["easy", "medium", "hard"]:
        if d in diffs:
            diff_rows += f'<tr><td style="font-weight:600;text-transform:capitalize">{d}</td><td class="num">{diffs[d]}</td></tr>\n'

    # Group check types into categories
    automated_checks = []
    judge_only_checks = []
    noop_types = {"calibration", "reasoning", "format_check", "checklist", "analysis", "synthesis", "comparison", "behavioural"}
    for ct in sorted(checks):
        display = ct.replace("_", " ")
        if ct in noop_types:
            judge_only_checks.append((display, checks[ct]))
        else:
            automated_checks.append((display, checks[ct]))

    auto_rows = "".join(
        f'<tr><td>{name}</td><td class="num">{count}</td></tr>\n'
        for name, count in automated_checks
    )
    judge_rows = "".join(
        f'<tr><td>{name}</td><td class="num">{count}</td></tr>\n'
        for name, count in judge_only_checks
    )

    # Build questions section grouped by category
    diff_colors = {"easy": "var(--green)", "medium": "var(--yellow)", "hard": "var(--red)"}
    questions_sections = ""
    for cat in sorted(cats):
        display_cat = cat.replace("_", " ").title()
        cat_prompts = [p for p in prompts if p["category"] == cat]
        prompt_cards = ""
        for p in cat_prompts:
            pid = p["id"]
            subcat = p["subcategory"].replace("_", " ")
            diff = p["difficulty"]
            diff_color = diff_colors.get(diff, "var(--text2)")
            prompt_text = html_mod.escape(p["prompt"])
            ideal_text = html_mod.escape(p.get("ideal", ""))
            criteria = p.get("criteria", [])
            criteria_html = " ".join(
                f'<span class="criteria-tag">{html_mod.escape(c)}</span>'
                for c in criteria
            )
            check = p.get("check_type", "").replace("_", " ")

            prompt_cards += f"""<div class="prompt-card" data-category="{cat}" data-difficulty="{diff}" data-check="{p.get('check_type', '')}">
          <div class="prompt-header">
            <span class="prompt-id">{pid}</span>
            <span class="prompt-subcat">{subcat}</span>
            <span class="prompt-diff" style="color:{diff_color}">{diff}</span>
            <span class="prompt-check">{check}</span>
          </div>
          <div class="prompt-criteria">{criteria_html}</div>
        </div>\n"""

        questions_sections += f"""<details class="category-section" open>
      <summary class="category-toggle">{display_cat} <span class="category-count">{cats[cat]} prompts</span></summary>
      {prompt_cards}
    </details>\n"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - Methodology</title>
{_slug_seo("methodology.html")}
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e7f0;
    --text2: #8b90a5;
    --accent: #6c72ff;
    --accent2: #4ecdc4;
    --green: #22c55e;
    --green-mid: #4ade80;
    --green-light: #86efac;
    --yellow: #eab308;
    --red: #ef4444;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #242836 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
  }}
  .header-inner {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
  .header .byline {{ font-size: 0.85rem; color: var(--text2); margin: 0.2rem 0 0; }}
  .header .meta {{ font-size: 0.75rem; color: var(--text2); margin-top: 0.5rem; }}
  .nav {{
    display: flex;
    gap: 0.25rem;
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.25rem;
  }}
  .nav-link {{
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text2);
    text-decoration: none;
    transition: all 0.2s;
  }}
  .nav-link:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
  .nav-link.active {{ color: var(--text); background: var(--accent); }}
  .container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 2.5rem 3rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .card h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--text);
  }}
  .card h3 {{
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    color: var(--accent2);
  }}
  .card p {{
    color: var(--text2);
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
  }}
  .card ul {{
    color: var(--text2);
    font-size: 0.85rem;
    margin-left: 1.25rem;
    margin-bottom: 0.75rem;
  }}
  .card li {{
    margin-bottom: 0.3rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
  }}
  th {{
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid var(--border);
    color: var(--text2);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  th.num {{ text-align: right; }}
  td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .scoring-scale {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.25rem 0.75rem;
    font-size: 0.85rem;
  }}
  .scoring-scale .score {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
  .scoring-scale .desc {{ color: var(--text2); }}
  .score-5 {{ color: var(--green); }}
  .score-4 {{ color: var(--green-light); }}
  .score-3 {{ color: var(--yellow); }}
  .score-2 {{ color: var(--orange); }}
  .score-1 {{ color: var(--red); }}
  .highlight {{
    background: var(--surface2);
    border-radius: 6px;
    padding: 1rem;
    margin: 0.75rem 0;
    font-size: 0.85rem;
    color: var(--text2);
    border-left: 3px solid var(--accent);
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
  }}
  .kpi .value {{
    font-size: 1.8rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .kpi .label {{
    font-size: 0.75rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
  }}
  @media (max-width: 900px) {{
    .header {{ padding: 1rem; }}
    .header-top {{ flex-direction: column; align-items: flex-start; gap: 0.5rem; }}
    .container {{ padding: 1rem; }}
    .grid-2 {{ grid-template-columns: 1fr; }}
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media (max-width: 600px) {{
    .container {{ padding: 0.75rem; }}
    .header {{ padding: 1rem 0.75rem; }}
    .header h1 {{ font-size: 1.2rem; }}
    .card {{ padding: 1rem; }}
    .kpi-row {{ grid-template-columns: 1fr 1fr; }}
    .kpi .value {{ font-size: 1.4rem; }}
  }}
  .category-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 1rem;
    overflow: hidden;
  }}
  .category-toggle {{
    padding: 1rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}
  .category-toggle::-webkit-details-marker {{ display: none; }}
  .category-toggle::before {{
    content: '\\25B6';
    font-size: 0.7rem;
    color: var(--accent);
    transition: transform 0.2s;
  }}
  details[open] > .category-toggle::before {{
    transform: rotate(90deg);
  }}
  .category-count {{
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text2);
    background: var(--surface2);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
  }}
  .prompt-card {{
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border);
  }}
  .prompt-card:hover {{
    background: rgba(255,255,255,0.02);
  }}
  .prompt-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
  }}
  .prompt-id {{
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--accent);
    background: rgba(108,114,255,0.1);
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-family: monospace;
  }}
  .prompt-subcat {{
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text);
    text-transform: capitalize;
  }}
  .prompt-diff {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .prompt-check {{
    font-size: 0.7rem;
    color: var(--text2);
    background: var(--surface2);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    text-transform: capitalize;
  }}
  .prompt-text {{
    font-size: 0.85rem;
    color: var(--text);
    line-height: 1.6;
    white-space: pre-wrap;
    background: var(--bg);
    padding: 0.75rem 1rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    margin-bottom: 0.6rem;
    font-family: 'Inter', -apple-system, sans-serif;
  }}
  .prompt-ideal {{
    font-size: 0.8rem;
    color: var(--text2);
    line-height: 1.5;
    margin-bottom: 0.5rem;
  }}
  .prompt-ideal strong {{
    color: var(--accent2);
  }}
  .prompt-criteria {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }}
  .criteria-tag {{
    font-size: 0.7rem;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    background: rgba(78,205,196,0.1);
    color: var(--accent2);
    font-weight: 500;
    text-transform: capitalize;
  }}
  .section-divider {{
    font-size: 1.3rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
  }}
  .filter-toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
    margin-bottom: 1rem;
    padding: 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
  }}
  .filter-toolbar input[type="text"] {{
    flex: 1;
    min-width: 200px;
    padding: 0.5rem 0.75rem;
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.85rem;
    outline: none;
  }}
  .filter-toolbar input[type="text"]:focus {{
    border-color: var(--accent);
  }}
  .filter-toolbar select {{
    padding: 0.5rem 0.75rem;
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
  }}
  .filter-toolbar .filter-count {{
    font-size: 0.8rem;
    color: var(--text2);
    margin-left: auto;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- LLM Evaluation Leaderboard</span></h1>
      {_nav_html("methodology.html", stats)}
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judges: {", ".join(stats["judge_models"])}' if stats.get("judge_models") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- Section Jump Nav: grouped by benchmark / reference -->
<div style="position:sticky;top:0;z-index:10;background:var(--bg);padding:0.6rem 0;margin-bottom:1rem;border-bottom:1px solid var(--border);display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center">
  <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text2);margin-right:0.5rem">On this page</span>
  <a href="#group-generalist" style="padding:0.35rem 0.8rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.8rem;color:var(--text2);text-decoration:none">Generalist scoring</a>
  <a href="#group-causal" style="padding:0.35rem 0.8rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.8rem;color:var(--text2);text-decoration:none">Causal scoring</a>
  <a href="#group-reference" style="padding:0.35rem 0.8rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.8rem;color:var(--text2);text-decoration:none">Reference data</a>
</div>

<!-- Quick stats -->
<div class="kpi-row">
  <div class="kpi">
    <div class="value">{len(prompts)}</div>
    <div class="label">Prompts</div>
  </div>
  <div class="kpi">
    <div class="value">{len(cats)}</div>
    <div class="label">Categories</div>
  </div>
  <div class="kpi">
    <div class="value">{len(checks)}</div>
    <div class="label">Check Types</div>
  </div>
  <div class="kpi">
    <div class="value">{stats['total_models']}</div>
    <div class="label">Models Tested</div>
  </div>
</div>

<!-- TL;DR -->
<div class="card" id="section-tldr" style="border-left:3px solid var(--accent2)">
  <h2 style="margin-bottom:0.5rem">In short</h2>
  <ul style="margin-left:1.25rem;line-height:1.7;color:var(--text)">
    <li><strong>Two benchmarks run side by side.</strong> <a href="generalist.html" style="color:var(--accent2)">Generalist</a> (80 prompts, 8 categories) tests breadth. <a href="causal.html" style="color:var(--accent2)">Causal</a> (100 multiple-choice questions, 5 variants) tests narrow causal-inference reasoning. Both display 0 to 100. They are never blended.</li>
    <li><strong>Generalist is scored by three layers.</strong> Auto-checks flag mechanical failures (format, hallucination, sycophancy). Four LLM judges score 1 to 5. DeepEval rates correctness, coherence, and instruction-following 0 to 1. Composite blends judge and DeepEval.</li>
    <li><strong>Causal is deterministic.</strong> Multiple choice. No judges, no DeepEval. Accuracy = correct ÷ valid. Errors (API failures) and Invalid (empty/unextractable response) are reported separately.</li>
    <li><strong>Self-judging is prevented.</strong> A judge LLM never scores responses from its own family (gpt-4.1 does not judge gpt-4o, etc.). See the <a href="judges.html" style="color:var(--accent2)">Judge Audit</a> for divergence and agreement evidence.</li>
    <li><strong>Some models are excluded.</strong> Retired APIs, paid-tier-only providers, and broken model paths are excluded from causal so the leaderboard isn't padded with zeros. They still appear on the Generalist board where they ran cleanly.</li>
  </ul>
</div>

<!-- Group: Generalist Benchmark -->
<div class="section-divider" id="group-generalist">Generalist Benchmark</div>

<!-- Focus -->
<div class="card" id="section-focus">
  <h2>Focus</h2>
  <p>
    This evaluation measures what matters for practical, day-to-day use of LLMs as a working tool.
    It is not a general knowledge benchmark or a trivia test. The prompt set is designed around
    tasks a developer, researcher, or technical writer would actually ask an LLM to do, with
    emphasis on scenarios where models commonly fail or diverge.
  </p>
  <h3>What we test for</h3>
  <ul>
    <li><strong>Accuracy under pressure</strong> - trap questions, false premises, phantom bugs, and wrong claims that tempt sycophantic agreement</li>
    <li><strong>Honest calibration</strong> - does the model hedge when uncertain, refuse when appropriate, and acknowledge its own limitations?</li>
    <li><strong>Instruction following</strong> - exact format compliance, word count targets, constraint adherence, and banned word avoidance</li>
    <li><strong>Reasoning depth</strong> - multi-step problems, causal reasoning, estimation, and the ability to show work rather than guess</li>
    <li><strong>Practical coding</strong> - real debugging scenarios, architecture decisions, code review, and implementation - not leetcode</li>
    <li><strong>Writing quality</strong> - tone control, concision, editing skill, and the ability to adapt style to audience</li>
  </ul>
  <h3>What we deliberately avoid</h3>
  <ul>
    <li>Trivia and memorization (Wikipedia knowledge is cheap)</li>
    <li>Simple Q&A that any model can pass</li>
    <li>Prompts with only one valid answer format</li>
    <li>Benchmarks that reward verbosity over substance</li>
  </ul>
</div>

<!-- Pipeline -->
<div class="card" id="section-pipeline">
  <h2>Evaluation Pipeline</h2>
  <p>Each model runs through the same pipeline for every prompt:</p>
  <div class="highlight">
    Prompt sent to model &rarr; Response collected with latency/token counts &rarr;
    Automated checks run &rarr; LLM judge scores 1-5 with rationale &rarr;
    DeepEval G-Eval metrics (correctness, coherence, instruction following) &rarr;
    Composite score computed (weighted merge of judge + DeepEval) &rarr;
    Results persisted as JSON
  </div>
  <ul>
    <li>All models receive identical prompts with <code>temperature: 0</code> for reproducibility</li>
    <li>No system prompts are injected - models receive only the raw user prompt</li>
    <li>Each prompt has a defined ideal answer and scoring criteria that the judge evaluates against</li>
    <li>Results are append-only - re-running a model adds a new entry, preserving history</li>
  </ul>
</div>

<!-- Two-layer scoring breakdown - collapsible reference data -->
<details class="card" style="padding:1rem 1.25rem">
  <summary style="cursor:pointer;list-style:none;font-weight:600;font-size:1rem;display:flex;justify-content:space-between;align-items:center">
    <span>Check-type breakdown (Layers 1 and 2)</span>
    <span style="color:var(--text2);font-size:0.8rem;font-weight:400">Tables of all auto-check and judge-only check types. Click to expand.</span>
  </summary>
  <div class="grid-2" style="margin-top:1rem">
    <div>
      <h3 style="margin-bottom:0.5rem">Auto-Checks (Layer 1)</h3>
      <p style="color:var(--text2);font-size:0.85rem;margin-bottom:0.75rem">Deterministic, heuristic checks. Run instantly on every response and flag mechanical failures.</p>
      <table>
        <thead><tr><th>Check Type</th><th class="num">Prompts</th></tr></thead>
        <tbody>{auto_rows}</tbody>
      </table>
    </div>
    <div>
      <h3 style="margin-bottom:0.5rem">Judge-Only (Layer 2)</h3>
      <p style="color:var(--text2);font-size:0.85rem;margin-bottom:0.75rem">These check types have no automated heuristic. The LLM judge scores them entirely on quality and reasoning.</p>
      <table>
        <thead><tr><th>Check Type</th><th class="num">Prompts</th></tr></thead>
        <tbody>{judge_rows}</tbody>
      </table>
    </div>
  </div>
</details>

<!-- Judge scoring -->
<div class="card" id="section-scoring">
  <h2>Multi-Judge Scoring</h2>
  <p>
    Each model response is scored by multiple independent LLM judges, each rating it 1 to 5.{f' The current judges are {", ".join(f"<strong>{html_mod.escape(j)}</strong>" for j in stats.get("judge_models", []))}.' if stats.get("judge_models") else ''}
    Each judge receives the original prompt, the ideal answer, the scoring criteria, and any
    auto-check flags. It returns a score and a short rationale.
  </p>
  <h3>Averaging rules</h3>
  <ul>
    <li>A judge's scores only count toward the average if it has scored <strong>every scorable prompt</strong> for that model - partial coverage is excluded entirely</li>
    <li>The displayed judge score is the mean of each qualifying judge's global average (equal weight per judge)</li>
    <li>Self-judging is prevented - a judge model does not score its own responses (e.g. gpt-4.1 does not judge gpt-4.1)</li>
    <li>Click any row on the leaderboard to see per-judge score breakdowns</li>
  </ul>
  <div class="scoring-scale">
    <span class="score score-5">5</span><span class="desc">Excellent - fully addresses the prompt, accurate, well-structured, meets all criteria</span>
    <span class="score score-4">4</span><span class="desc">Good - mostly correct with minor gaps or style issues</span>
    <span class="score score-3">3</span><span class="desc">Adequate - partially addresses the prompt, some errors or missing elements</span>
    <span class="score score-2">2</span><span class="desc">Poor - significant errors, missing key requirements, or off-topic</span>
    <span class="score score-1">1</span><span class="desc">Failing - wrong, harmful, empty, or completely misses the point</span>
  </div>
  <h3>Judge guidelines</h3>
  <ul>
    <li>Hallucinated facts, fabricated references, and confident wrong answers are penalised</li>
    <li>Appropriate hedging, asking for clarification, and refusing harmful requests are rewarded</li>
    <li>Auto-check flag failures lower the score</li>
    <li>A 3 is average, 5 is genuinely excellent - the scale is strict but fair</li>
  </ul>
</div>

<!-- DeepEval scoring -->
<div class="card" id="section-deepeval">
  <h2>DeepEval G-Eval Scoring (Layer 3)</h2>
  <p>
    In addition to the multi-judge scores, each response is scored by
    <a href="https://github.com/confident-ai/deepeval" style="color:var(--accent)">DeepEval</a>
    using G-Eval metrics - research-backed LLM evaluation criteria that provide
    multi-dimensional scoring on a 0-1 scale.
  </p>
  <h3>Metrics</h3>
  <div class="scoring-scale">
    <span class="score score-5">Correctness</span><span class="desc">Is the response factually correct compared to the expected output? Penalises contradictions, omissions, and hallucinations.</span>
    <span class="score score-4">Coherence</span><span class="desc">Does the response have clear logical flow, good structure, and present ideas without contradictions?</span>
    <span class="score score-3">Instruction Following</span><span class="desc">Does the response address all parts of the prompt and adhere to format, length, and constraint requirements?</span>
  </div>
  <h3>How it works</h3>
  <ul>
    <li>Each metric uses a chain-of-thought evaluation via the same judge model</li>
    <li>DeepEval's native scores are 0 to 1; the dashboard displays them as 0 to 100 for parity with the other panels</li>
    <li>DeepEval supplements the LLM judge rather than replacing it. Both signals appear separately on the leaderboard so you can see when they agree.</li>
  </ul>
</div>

<!-- Composite score -->
<div class="card" id="section-composite">
  <h2>Composite Score (Generalist)</h2>
  <p>
    The composite is the headline number on the Generalist leaderboard. It blends two signals
    into one, displayed as 0 to 100. The dashboard computes everything internally on a
    0 to 1 scale, then multiplies by 100 for display.
  </p>
  <div class="highlight">
    composite = judge_weight &times; normalized_judge + deepeval_weight &times; deepeval_avg
  </div>
  <p style="color:var(--text2);font-size:0.9rem">
    where <code>normalized_judge = (avg_judge_score - 1) / 4</code> rescales the 1 to 5 judge
    average into 0 to 1, and <code>deepeval_avg</code> is the mean of correctness, coherence,
    and instruction-following metrics. Only judges with complete coverage (scored every
    prompt for that model) contribute to the average; partial-coverage judges are excluded
    entirely to avoid biased subsets.
  </p>
  <h3>Fallback behavior</h3>
  <ul>
    <li><strong>Both scores available</strong> - weighted average (default 50/50)</li>
    <li><strong>Only judge score</strong> - composite = normalized judge</li>
    <li><strong>Only DeepEval</strong> - composite = DeepEval average</li>
    <li><strong>Neither</strong> - no composite score</li>
  </ul>
  <p style="color:var(--text2);font-size:0.9rem;margin-top:0.5rem">
    The Causal benchmark uses a <strong>different scoring scheme</strong> (deterministic multiple
    choice, no judges, no DeepEval). The two scores are reported side by side on the leaderboard
    but never blended.
  </p>
</div>

<!-- Efficiency metric -->
<div class="card">
  <h2>Efficiency Metric</h2>
  <p>
    The efficiency score balances quality against verbosity:
    <code>efficiency = avg_score / log2(avg_tokens)</code>.
    This rewards models that achieve high scores without padding responses with unnecessary tokens.
    A concise, correct answer scores higher than an equally correct but bloated one.
  </p>
</div>

<!-- Reasoning model handling -->
<div class="card" id="section-reasoning">
  <h2>Reasoning Models</h2>
  <p>
    Reasoning-capable models (gpt-5.x, o3-mini, o4-mini, Gemini Pro reasoning) spend output
    tokens on hidden chain-of-thought before emitting any visible answer. When the response
    budget runs out before the model writes its final answer, the API returns success but
    with empty text. The dashboard counts these as <strong>Invalid</strong> rather than
    <strong>Errors</strong>, so a token-budget failure is distinguishable from an API failure.
    This matters when comparing models: an "invalid" rate on a reasoning model is a real
    capability signal, not a network problem.
  </p>
</div>

<!-- Group: Reference data -->
<div class="section-divider" id="group-reference">Reference Data</div>

<!-- Prompt breakdown -->
<div class="card" id="section-categories">
  <h2>Prompt Set Breakdown</h2>
  <table>
    <thead><tr><th>Category</th><th class="num">Prompts</th><th>Subcategories</th><th>What It Tests</th></tr></thead>
    <tbody>{cat_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>Difficulty Distribution</h2>
  <table>
    <thead><tr><th>Difficulty</th><th class="num">Prompts</th></tr></thead>
    <tbody>{diff_rows}</tbody>
  </table>
</div>

<div class="card" style="border-left:3px solid var(--yellow);margin-top:2rem">
  <h3 style="margin-top:0">Benchmark Integrity</h3>
  <p style="margin:0;color:var(--text2)">
    Exact prompts are not published to prevent models from being tuned to this specific benchmark.
    Categories, evaluation criteria, and scoring methodology are fully documented above.
    Each prompt is scored by automated checks where applicable, plus multi-judge LLM scoring for nuanced evaluation.
  </p>
</div>

{causal_section_html}

<!-- Generalist prompt list collapsed by default to keep the page browsable -->
<details class="card" id="section-prompts" style="padding:1rem 1.25rem">
  <summary style="cursor:pointer;list-style:none;font-weight:600;font-size:1rem;display:flex;justify-content:space-between;align-items:center">
    <span>Generalist prompt list ({len(prompts)})</span>
    <span style="color:var(--text2);font-size:0.8rem;font-weight:400">All {len(prompts)} Generalist prompts, searchable. Causal questions are not published. Click to expand.</span>
  </summary>
  <div style="margin-top:1rem">
    <div class="filter-toolbar">
      <input type="text" id="search-input" placeholder="Search prompts...">
      <select id="filter-category">
        <option value="">All Categories</option>
        {"".join(f'<option value="{c}">{c.replace("_", " ").title()}</option>' for c in sorted(cats))}
      </select>
      <select id="filter-difficulty">
        <option value="">All Difficulties</option>
        <option value="easy">Easy</option>
        <option value="medium">Medium</option>
        <option value="hard">Hard</option>
      </select>
      <select id="filter-check">
        <option value="">All Check Types</option>
        {"".join(f'<option value="{ct}">{ct.replace("_", " ").title()}</option>' for ct in sorted(checks))}
      </select>
      <span class="filter-count" id="filter-count">{len(prompts)} of {len(prompts)} shown</span>
    </div>

    {questions_sections}
  </div>
</details>

</div>

<script>
(function() {{
  const searchInput = document.getElementById('search-input');
  const catFilter = document.getElementById('filter-category');
  const diffFilter = document.getElementById('filter-difficulty');
  const checkFilter = document.getElementById('filter-check');
  const countDisplay = document.getElementById('filter-count');
  const totalPrompts = {len(prompts)};

  function applyFilters() {{
    const query = searchInput.value.toLowerCase();
    const cat = catFilter.value;
    const diff = diffFilter.value;
    const check = checkFilter.value;
    let shown = 0;

    document.querySelectorAll('.prompt-card').forEach(card => {{
      const matchCat = !cat || card.dataset.category === cat;
      const matchDiff = !diff || card.dataset.difficulty === diff;
      const matchCheck = !check || card.dataset.check === check;
      const matchText = !query || card.textContent.toLowerCase().includes(query);
      const visible = matchCat && matchDiff && matchCheck && matchText;
      card.style.display = visible ? '' : 'none';
      if (visible) shown++;
    }});

    // Hide category sections where all children are hidden
    document.querySelectorAll('.category-section').forEach(section => {{
      const visibleCards = section.querySelectorAll('.prompt-card:not([style*="display: none"])');
      section.style.display = visibleCards.length > 0 ? '' : 'none';
    }});

    countDisplay.textContent = shown + ' of ' + totalPrompts + ' shown';
  }}

  searchInput.addEventListener('input', applyFilters);
  catFilter.addEventListener('change', applyFilters);
  diffFilter.addEventListener('change', applyFilters);
  checkFilter.addEventListener('change', applyFilters);
}})();
</script>

</body>
</html>"""


def generate_judges_html(stats):
    """Generate the judges analysis page."""
    data_json = json.dumps(stats)

    judge_global = stats.get("judge_global", {})
    judge_by_category = stats.get("judge_by_category", {})
    judge_by_difficulty = stats.get("judge_by_difficulty", {})
    judge_by_model = stats.get("judge_by_model", {})
    judge_pairwise = stats.get("judge_pairwise", {})
    judge_pairwise_matrix_raw = stats.get("judge_pairwise_matrix", {})
    judge_pairwise_matrix = {tuple(k.split("|", 1)): v for k, v in judge_pairwise_matrix_raw.items()}
    judge_score_dists = stats.get("judge_score_distributions", {})
    judge_vs_deepeval = stats.get("judge_vs_deepeval", {})
    biggest_disagreements = stats.get("biggest_disagreements", [])
    all_judges = sorted(judge_global.keys())

    # No judges at all - show a placeholder
    if not all_judges:
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>BenchPress - Judges</title>
{_slug_seo("judges.html")}
<style>body{{font-family:sans-serif;background:#0f1117;color:#e4e7f0;padding:3rem;text-align:center}}
a{{color:#6c72ff}}</style></head>
<body><h1>No judge data available yet</h1><p>Run evaluations with multiple judges to see analysis here.</p>
<p><a href="index.html">Back to overview</a></p></body></html>"""

    # KPI cards
    judge_total_scored = {}
    for jname, dist in judge_score_dists.items():
        judge_total_scored[jname] = sum(dist.values())

    # Find strictest / most lenient
    sorted_judges = sorted(all_judges, key=lambda j: judge_global.get(j, 0))
    strictest = sorted_judges[0] if sorted_judges else "-"
    most_lenient = sorted_judges[-1] if sorted_judges else "-"

    max_scored = max(judge_total_scored.values()) if judge_total_scored else 0

    kpi_cards = ""
    for jname in all_judges:
        avg = judge_global.get(jname, 0)
        total = judge_total_scored.get(jname, 0)
        badge = ""
        if jname == strictest and len(all_judges) > 1:
            badge = '<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.7rem;font-weight:600;background:rgba(239,68,68,0.15);color:#ef4444;margin-left:0.5rem">Strictest</span>'
        elif jname == most_lenient and len(all_judges) > 1:
            badge = '<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.7rem;font-weight:600;background:rgba(34,197,94,0.15);color:#22c55e;margin-left:0.5rem">Most Lenient</span>'
        progress_badge = ""
        if max_scored > 0 and total < max_scored * 0.8:
            pct = round(100 * total / max_scored)
            progress_badge = f' <span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.7rem;font-weight:600;background:rgba(234,179,8,0.15);color:#eab308;margin-left:0.25rem">{pct}% complete</span>'
        kpi_cards += f"""<div class="kpi">
          <div class="label">{html_mod.escape(jname)}{badge}</div>
          <div class="value">{avg:.2f}<span style="font-size:0.9rem;color:var(--text2)">/5</span></div>
          <div class="sub">{total} prompts scored{progress_badge}</div>
        </div>\n"""

    # Pairwise agreement heatmap
    def _agree_heatmap_color(pct):
        if pct >= 90: return "rgba(34,197,94,0.25)"
        if pct >= 80: return "rgba(134,239,172,0.2)"
        if pct >= 70: return "rgba(234,179,8,0.2)"
        if pct >= 60: return "rgba(249,115,22,0.2)"
        return "rgba(239,68,68,0.2)"

    def _agree_text_color(pct):
        if pct >= 80: return "#22c55e"
        if pct >= 60: return "#eab308"
        return "#ef4444"

    heatmap_html = ""
    if len(all_judges) >= 2:
        # Build header row
        hdr_cells = '<th style="min-width:100px"></th>'
        for j in all_judges:
            hdr_cells += f'<th style="text-align:center;font-size:0.75rem;padding:0.5rem;color:var(--text2)">{html_mod.escape(j)}</th>'
        # Build body rows
        body_rows = ""
        for ja in all_judges:
            cells = f'<td style="font-weight:600;font-size:0.75rem;white-space:nowrap;padding:0.5rem 0.75rem">{html_mod.escape(ja)}</td>'
            for jb in all_judges:
                pdata = judge_pairwise_matrix.get((ja, jb))
                if pdata and pdata["self"]:
                    cells += '<td style="text-align:center;padding:0.5rem;background:var(--surface);color:var(--text2);font-size:0.7rem">-</td>'
                elif pdata:
                    bg = _agree_heatmap_color(pdata["agree_pct"])
                    tc = _agree_text_color(pdata["agree_pct"])
                    cells += f'<td style="text-align:center;padding:0.5rem;background:{bg}" title="Avg diff: {pdata["avg_diff"]:.2f} | {pdata["n"]} prompts compared"><div style="font-size:0.95rem;font-weight:700;color:{tc}">{pdata["agree_pct"]}%</div><div style="font-size:0.65rem;color:var(--text2);margin-top:2px">diff {pdata["avg_diff"]:.2f}</div></td>'
                else:
                    cells += '<td style="text-align:center;padding:0.5rem;color:var(--text2);font-size:0.7rem">n/a</td>'
            body_rows += f"<tr>{cells}</tr>\n"
        heatmap_html = f"""<table style="border-collapse:collapse;width:auto">
          <thead><tr>{hdr_cells}</tr></thead>
          <tbody>{body_rows}</tbody>
        </table>"""

    # Judge vs DeepEval rows
    jvd_rows = ""
    for jname in all_judges:
        jvd = judge_vs_deepeval.get(jname, {})
        div_val = jvd.get("avg_divergence")
        if div_val is not None:
            div_color = "#22c55e" if div_val <= 0.10 else "#eab308" if div_val <= 0.20 else "#ef4444"
            jvd_rows += f"""<tr>
              <td style="font-weight:600">{html_mod.escape(jname)}</td>
              <td class="num" style="font-weight:600;color:{div_color}">{div_val:.4f}</td>
            </tr>\n"""

    # Biggest disagreements table rows
    disagree_rows = ""
    for d in biggest_disagreements[:20]:
        score_cells = ""
        for jname in all_judges:
            sc = d["scores"].get(jname)
            if sc is not None:
                sc_color = _score_color(sc)
                score_cells += f'<td class="num {sc_color}" style="font-weight:600">{sc}/5</td>'
            else:
                score_cells += '<td class="num" style="color:var(--text2)">-</td>'
        spread_color = "#ef4444" if d["spread"] >= 3 else "#eab308" if d["spread"] >= 2 else "#f97316"
        disagree_rows += f"""<tr>
          <td style="font-weight:600;color:var(--accent)">{html_mod.escape(d['prompt_id'])}</td>
          <td>{html_mod.escape(d['model'])}</td>
          <td>{html_mod.escape(d['category'])}</td>
          {score_cells}
          <td class="num" style="font-weight:700;color:{spread_color}">{d['spread']}</td>
        </tr>\n"""

    judge_score_headers = "".join(f'<th class="num">{html_mod.escape(j)}</th>' for j in all_judges)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - Judge Analysis</title>
{_slug_seo("judges.html")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e7f0;
    --text2: #8b90a5;
    --accent: #6c72ff;
    --accent2: #4ecdc4;
    --green: #22c55e;
    --green-mid: #4ade80;
    --green-light: #86efac;
    --yellow: #eab308;
    --red: #ef4444;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #242836 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
  }}
  .header-inner {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
  .header .byline {{ font-size: 0.85rem; color: var(--text2); margin: 0.2rem 0 0; }}
  .header .meta {{ font-size: 0.75rem; color: var(--text2); margin-top: 0.5rem; }}
  .nav {{
    display: flex;
    gap: 0.25rem;
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.25rem;
  }}
  .nav-link {{
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text2);
    text-decoration: none;
    transition: all 0.2s;
  }}
  .nav-link:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
  .nav-link.active {{ color: var(--text); background: var(--accent); }}
  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 1.5rem 2.5rem 3rem;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
  }}
  .kpi .label {{
    font-size: 0.75rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }}
  .kpi .value {{
    font-size: 1.8rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .kpi .sub {{
    font-size: 0.8rem;
    color: var(--text2);
    margin-top: 0.25rem;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    align-items: stretch;
  }}
  .grid-full {{
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
  }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text);
  }}
  .card .chart-container {{
    flex: 1;
  }}
  .chart-container {{
    position: relative;
    width: 100%;
    min-height: 320px;
    height: 320px;
  }}
  .chart-container-wide {{
    position: relative;
    width: 100%;
    min-height: 400px;
    height: 400px;
  }}
  .info-tip {{
    display: inline-block;
    position: relative;
    width: 16px;
    height: 16px;
    line-height: 16px;
    text-align: center;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text2);
    background: var(--surface2);
    border-radius: 50%;
    margin-left: 6px;
    cursor: default;
    vertical-align: middle;
  }}
  .info-tip:hover::after {{
    content: attr(data-info);
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 6px;
    background: var(--surface2);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 400;
    white-space: nowrap;
    z-index: 20;
    pointer-events: none;
    border: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 2px solid var(--border);
    color: var(--text2);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }}
  th.num {{ text-align: right; }}
  td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }}
  td.num {{ text-align: right; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .table-scroll {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}
  .score-5 {{ color: var(--green); }}
  .score-4 {{ color: var(--green-light); }}
  .score-3 {{ color: var(--yellow); }}
  .score-2 {{ color: var(--orange); }}
  .score-1 {{ color: var(--red); }}
  .section-title {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text2);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  @media (max-width: 1100px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 600px) {{
    .kpi-row {{ grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
    .container {{ padding: 0.75rem; }}
    .header {{ padding: 1rem 0.75rem; }}
    .chart-container {{ height: 260px; min-height: 260px; }}
    .chart-container-wide {{ height: 300px; min-height: 300px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- Judge Analysis</span></h1>
      {_nav_html("judges.html", stats)}
    </div>
    <p class="byline">How do different judges score models?</p>
    <div class="meta">{len(all_judges)} judge(s) &middot; {stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- Judge KPI Strip -->
<div class="kpi-row">
  {kpi_cards}
</div>

<!-- Bias disclosure -->
<div class="card" style="border-left:3px solid var(--accent2);background:var(--surface)">
  <h3 style="font-size:1rem;margin-bottom:0.5rem">A note on judge bias</h3>
  <p style="color:var(--text2);font-size:0.9rem;margin:0;line-height:1.6">
    The four judges ({", ".join(html_mod.escape(j) for j in all_judges)}) are themselves LLMs from companies whose models also appear on the leaderboard. This is a real conflict-of-interest concern. Two safeguards: (1) <strong>self-judging is prevented</strong>: a judge never scores its own family's responses (e.g. gpt-4.1 does not judge gpt-4.1, gpt-4o, etc.). (2) The <strong>Judge Agreement</strong> matrix and <strong>Biggest Disagreements</strong> table below let you inspect where judges disagree and decide whether bias is bounded for your use case.
  </p>
</div>

<!-- Score Distributions -->
<div class="grid-full">
  <div class="card">
    <h2>Judge Score Distributions <span class="info-tip" data-info="How each judge distributes scores 1-5. Reveals if a judge skews harsh (more 1-2) or lenient (more 4-5).">?</span></h2>
    <div class="chart-container-wide">
      <canvas id="judgeDistChart"></canvas>
    </div>
  </div>
</div>

<!-- Comparison by Model + by Category -->
<div class="grid-2">
  <div class="card">
    <h2>By Model (Top 15) <span class="info-tip" data-info="Each judge's average score per model. Shows where judges agree and disagree.">?</span></h2>
    <div class="chart-container-wide">
      <canvas id="judgeByModelChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>By Category <span class="info-tip" data-info="Each judge's average score per category. Shows if a judge is harder on certain task types.">?</span></h2>
    <div class="chart-container-wide">
      <canvas id="judgeByCategoryChart"></canvas>
    </div>
  </div>
</div>

<!-- Comparison by Difficulty -->
<div class="grid-2">
  <div class="card">
    <h2>By Difficulty <span class="info-tip" data-info="Each judge's average score for easy, medium, and hard prompts.">?</span></h2>
    <div class="chart-container">
      <canvas id="judgeByDifficultyChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Judge vs DeepEval Divergence <span class="info-tip" data-info="Per judge: average absolute difference between the judge's normalised score (0-1) and DeepEval average across all prompts. Lower means more aligned with automated evaluation.">?</span></h2>
    <div class="chart-container">
      <canvas id="judgeVsDeepEvalChart"></canvas>
    </div>
  </div>
</div>

<!-- Pairwise Agreement Heatmap -->
{"" if not heatmap_html else f'''<div class="grid-full">
  <div class="card">
    <h2>Judge Agreement <span class="info-tip" data-info="Pairwise agreement between judges. Each cell shows the percentage of prompts where both judges scored within 1 point of each other, plus the average score difference. Hover for details.">?</span></h2>
    <div class="table-scroll">
      {heatmap_html}
    </div>
  </div>
</div>'''}

<!-- Biggest Disagreements -->
{"" if not disagree_rows else f'''<div class="grid-full">
  <div class="card">
    <h2>Biggest Disagreements <span class="info-tip" data-info="Prompts where judges disagreed the most, sorted by score spread.">?</span></h2>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Prompt</th>
            <th>Model</th>
            <th>Category</th>
            {judge_score_headers}
            <th class="num">Spread</th>
          </tr>
        </thead>
        <tbody>
          {disagree_rows}
        </tbody>
      </table>
    </div>
  </div>
</div>'''}

</div>

<script type="module">
const DATA = await (await fetch('./data.json')).json();
const COLORS = [
  '#6c72ff', '#4ecdc4', '#f97316', '#22c55e', '#ec4899',
  '#eab308', '#8b5cf6', '#06b6d4', '#ef4444', '#84cc16'
];
Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

const judges = Object.keys(DATA.judge_global || {{}}).sort();

// Score distribution grouped bar chart
(function() {{
  const canvas = document.getElementById('judgeDistChart');
  if (!canvas || !judges.length) return;
  const scores = [1, 2, 3, 4, 5];
  const scoreColors = ['#ef4444', '#f97316', '#eab308', '#86efac', '#22c55e'];
  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: scores.map(s => s + '/5'),
      datasets: judges.map((j, i) => ({{
        label: j,
        data: scores.map(s => (DATA.judge_score_distributions[j] || {{}})[s] || 0),
        backgroundColor: COLORS[i % COLORS.length] + 'cc',
        borderColor: COLORS[i % COLORS.length],
        borderWidth: 1,
        borderRadius: 4,
      }}))
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
        }}
      }},
      scales: {{
        x: {{ ticks: {{ font: {{ size: 12 }} }} }},
        y: {{ beginAtZero: true, title: {{ display: true, text: 'Count', color: '#8b90a5' }} }}
      }}
    }}
  }});
}})();

// By Model grouped bar chart (top 15)
(function() {{
  const canvas = document.getElementById('judgeByModelChart');
  if (!canvas || !judges.length) return;
  const jbm = DATA.judge_by_model || {{}};
  // Get all models from the first judge, sorted by leaderboard order
  const lbNames = DATA.leaderboard.map(m => m.name);
  const modelNames = lbNames.slice(0, 15);
  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: modelNames,
      datasets: judges.map((j, i) => ({{
        label: j,
        data: modelNames.map(m => (jbm[j] || {{}})[m] || 0),
        backgroundColor: COLORS[i % COLORS.length] + 'cc',
        borderColor: COLORS[i % COLORS.length],
        borderWidth: 1,
        borderRadius: 3,
      }}))
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
        }}
      }},
      scales: {{
        y: {{ min: 1, max: 5, ticks: {{ stepSize: 1 }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }}
      }}
    }}
  }});
}})();

// By Category grouped bar chart
(function() {{
  const canvas = document.getElementById('judgeByCategoryChart');
  if (!canvas || !judges.length) return;
  const jbc = DATA.judge_by_category || {{}};
  const cats = DATA.categories || [];
  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: cats.map(c => c.replace('_', ' ')),
      datasets: judges.map((j, i) => ({{
        label: j,
        data: cats.map(c => (jbc[j] || {{}})[c] || 0),
        backgroundColor: COLORS[i % COLORS.length] + 'cc',
        borderColor: COLORS[i % COLORS.length],
        borderWidth: 1,
        borderRadius: 4,
      }}))
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
        }}
      }},
      scales: {{
        y: {{ min: 1, max: 5, ticks: {{ stepSize: 1 }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}})();

// By Difficulty grouped bar chart
(function() {{
  const canvas = document.getElementById('judgeByDifficultyChart');
  if (!canvas || !judges.length) return;
  const jbd = DATA.judge_by_difficulty || {{}};
  const diffs = ['easy', 'medium', 'hard'];
  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: diffs.map(d => d.charAt(0).toUpperCase() + d.slice(1)),
      datasets: judges.map((j, i) => ({{
        label: j,
        data: diffs.map(d => (jbd[j] || {{}})[d] || 0),
        backgroundColor: COLORS[i % COLORS.length] + 'cc',
        borderColor: COLORS[i % COLORS.length],
        borderWidth: 1,
        borderRadius: 4,
      }}))
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
        }}
      }},
      scales: {{
        y: {{ min: 1, max: 5, ticks: {{ stepSize: 1 }} }},
        x: {{ ticks: {{ font: {{ size: 12 }} }} }}
      }}
    }}
  }});
}})();

// Judge vs DeepEval divergence
(function() {{
  const canvas = document.getElementById('judgeVsDeepEvalChart');
  if (!canvas || !judges.length) return;
  const jvd = DATA.judge_vs_deepeval || {{}};
  const divData = judges.map(j => ({{
    name: j,
    div: (jvd[j] || {{}}).avg_divergence || 0
  }})).sort((a, b) => a.div - b.div);

  function divColor(d) {{
    if (d <= 0.10) return '#22c55e';
    if (d <= 0.15) return '#86efac';
    if (d <= 0.20) return '#eab308';
    return '#ef4444';
  }}

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: divData.map(d => d.name),
      datasets: [{{
        data: divData.map(d => d.div),
        backgroundColor: divData.map(d => divColor(d.div) + 'cc'),
        borderColor: divData.map(d => divColor(d.div)),
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ beginAtZero: true, title: {{ display: true, text: 'Avg Divergence', color: '#8b90a5' }} }},
        x: {{ ticks: {{ font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}})();
</script>

</body>
</html>"""


def compute_causal_stats(models, prompts, models_cfg=None):
    """Compute stats for the causal reasoning benchmark with bundle/variant structure."""
    models_cfg = models_cfg or {}
    if not prompts:
        return None

    bundles = sorted(set(p.get("bundle_id", "") for p in prompts))
    variants = ["base", "trap", "transfer", "numeric", "analyst"]
    bundle_pids = {b: [p["id"] for p in prompts if p.get("bundle_id") == b] for b in bundles}
    variant_pids = {v: [p["id"] for p in prompts if p.get("variant") == v] for v in variants}

    leaderboard = []
    for name, data in models.items():
        correct = 0
        total_answered = 0
        invalid = 0
        errors = 0
        variant_correct = {v: 0 for v in variants}
        variant_total = {v: 0 for v in variants}
        bundle_scores = {}

        for p in prompts:
            pid = p["id"]
            runs = data.get("runs", {}).get(pid, [])
            run = runs[-1] if runs else {}
            if not run:
                continue
            if run.get("error"):
                errors += 1
                continue
            total_answered += 1
            auto = run.get("auto_checks", {}).get("auto_scores", {})
            extracted = auto.get("extracted_answer")
            if extracted is None:
                invalid += 1
                continue
            is_correct = auto.get("correct", 0) == 1
            if is_correct:
                correct += 1
            v = p.get("variant", "")
            if v in variant_total:
                variant_total[v] += 1
                if is_correct:
                    variant_correct[v] += 1
            bid = p.get("bundle_id", "")
            if bid:
                bundle_scores.setdefault(bid, {"correct": 0, "total": 0})
                bundle_scores[bid]["total"] += 1
                if is_correct:
                    bundle_scores[bid]["correct"] += 1

        valid = total_answered - invalid
        accuracy = round(correct / valid, 4) if valid else 0
        variant_accuracy = {}
        for v in variants:
            variant_accuracy[v] = round(variant_correct[v] / variant_total[v], 4) if variant_total[v] else None
        bundle_consistency = {}
        for b in bundles:
            bs = bundle_scores.get(b, {"correct": 0, "total": 0})
            bundle_consistency[b] = {"correct": bs["correct"], "total": bs["total"]}

        mcfg = models_cfg.get(name, {})
        leaderboard.append({
            "name": name,
            "company": mcfg.get("company", "Unknown"),
            "model_type": mcfg.get("type", "unknown"),
            "accuracy": accuracy,
            "correct": correct,
            "valid": valid,
            "total": total_answered,
            "invalid": invalid,
            "errors": errors,
            "variant_accuracy": variant_accuracy,
            "bundle_consistency": bundle_consistency,
        })

    leaderboard = [m for m in leaderboard if m["total"] > 0]
    leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)

    return {
        "leaderboard": leaderboard,
        "bundles": bundles,
        "variants": variants,
        "total_prompts": len(prompts),
        "total_models": len(leaderboard),
        "generated": datetime.now().isoformat(),
    }


def generate_causal_html(causal_stats, stats):
    """Generate the causal reasoning benchmark HTML page."""
    if not causal_stats or not causal_stats["leaderboard"]:
        return '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>BenchPress - Causal</title></head><body><h1>Causal Reasoning</h1><p>No results yet. Run: <code>python run.py eval &lt;model&gt; --benchmark causal</code></p></body></html>'

    lb = causal_stats["leaderboard"]
    variants = causal_stats["variants"]
    bundles = causal_stats["bundles"]
    data_json = json.dumps(causal_stats)

    # Leaderboard rows. Score now uses total_prompts as denominator so the fraction is comparable
    # across models even when a model has errors or invalid responses.
    total_prompts = causal_stats["total_prompts"]
    rows = ""
    for i, m in enumerate(lb):
        pct = round(m["accuracy"] * 100, 1)
        bar_color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 60 else "#ef4444"
        err_str = str(m.get("errors", 0)) if m.get("errors") else "0"
        inv_str = str(m["invalid"]) if m["invalid"] else "0"
        rows += f"""<tr>
          <td style="text-align:center">{i+1}</td>
          <td><strong>{html_mod.escape(m["name"])}</strong><br><span style="color:var(--text2);font-size:0.85em">{html_mod.escape(m["company"])}</span></td>
          <td style="text-align:center"><strong>{pct}%</strong><div style="background:var(--bg2);border-radius:4px;height:6px;margin-top:4px"><div style="background:{bar_color};height:100%;width:{pct}%;border-radius:4px"></div></div></td>
          <td style="text-align:center">{m["correct"]}/{total_prompts}</td>
          <td style="text-align:center;{'color:#ef4444' if m.get('errors') else 'color:var(--text2)'}">{err_str}</td>
          <td style="text-align:center;{'color:#eab308' if m['invalid'] else 'color:var(--text2)'}">{inv_str}</td>
        </tr>"""

    # Variant winner per type (top model on each variant) + variant heatmap (top 15)
    VARIANT_DESCRIPTIONS = {
        "base": "Narrative scenario combining 2-3 interacting causal issues. Tests recognising compound problems.",
        "trap": "Looks like the base concept applies but the obvious answer is wrong. Tests resistance to pattern matching.",
        "transfer": "Same reasoning translated into a formal DAG with short elimination-style options. Tests structural reasoning.",
        "numeric": "Multi-step calculation with tables and conditional probabilities. Cannot be answered by intuition alone.",
        "analyst": "Two analysts debate the scenario; identify which assessment is most accurate. Tests evaluating arguments.",
    }
    variant_winners = []
    for v in variants:
        best, best_acc = None, -1
        for m in lb:
            val = m["variant_accuracy"].get(v)
            if val is not None and val > best_acc:
                best_acc = val
                best = m["name"]
        variant_winners.append((v, best, best_acc))

    variant_winner_cards = ""
    for v, name, acc in variant_winners:
        pct = round(acc * 100) if acc >= 0 else 0
        color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 60 else "#ef4444"
        variant_winner_cards += f'''<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:0.9rem 1rem">
          <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text2);margin-bottom:0.3rem">{v.title()}</div>
          <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.15rem">{html_mod.escape(name) if name else '-'}</div>
          <div style="font-size:1.4rem;font-weight:700;color:{color}">{pct}%</div>
          <div style="font-size:0.75rem;color:var(--text2);margin-top:0.4rem;line-height:1.4">{VARIANT_DESCRIPTIONS[v]}</div>
        </div>'''

    # Variant heatmap (top 15 only): rows = top 15 models, cols = 5 variants
    var_lb = lb[:15]
    var_header = "".join(f'<th style="text-align:center;text-transform:capitalize;color:var(--text2);font-size:0.75rem">{v}</th>' for v in variants)
    var_rows = ""
    for m in var_lb:
        cells = ""
        for v in variants:
            val = m["variant_accuracy"].get(v)
            if val is not None:
                pct = round(val * 100)
                color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 60 else "#ef4444"
                cells += f'<td style="text-align:center;color:{color};background:{color}1a;font-weight:600">{pct}</td>'
            else:
                cells += '<td style="text-align:center;color:var(--text2)">-</td>'
        var_rows += f'<tr><td><strong>{html_mod.escape(m["name"])}</strong></td>{cells}</tr>'

    # Per-variant grouped bar chart data: top 10 models × 5 variants
    chart_lb = lb[:10]
    variant_chart_data = json.dumps({
        "labels": [m["name"] for m in chart_lb],
        "variants": variants,
        "datasets": [
            {
                "label": v.title(),
                "data": [round((m["variant_accuracy"].get(v) or 0) * 100) for m in chart_lb],
            }
            for v in variants
        ],
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BenchPress - Causal Reasoning</title>
{_slug_seo("causal.html")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ --bg: #0a0a0a; --bg2: #1a1a1a; --bg3: #2a2a2a; --text: #e5e5e5; --text2: #a3a3a3; --accent: #3b82f6; --accent2: #60a5fa; --border: #333; --radius: 8px; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
  .header {{ background:var(--bg2); border-bottom:1px solid var(--border); padding:1.5rem 2rem; }}
  .header-top {{ max-width:1400px; margin:0 auto; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; }}
  .nav {{ display:flex; gap:0.5rem; flex-wrap:wrap; }}
  .nav-link {{ color:var(--text2); text-decoration:none; padding:0.4rem 0.8rem; border-radius:var(--radius); font-size:0.9rem; transition:all 0.2s; }}
  .nav-link:hover {{ color:var(--text); background:var(--bg3); }}
  .nav-link.active {{ color:var(--accent2); background:var(--bg3); }}
  .container {{ max-width:1400px; margin:0 auto; padding:2rem; }}
  .card {{ background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:1.5rem; margin-bottom:1.5rem; }}
  .card h2 {{ margin-bottom:1rem; font-size:1.2rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
  th {{ background:var(--bg3); padding:0.6rem 0.8rem; text-align:left; font-weight:600; border-bottom:2px solid var(--border); }}
  td {{ padding:0.5rem 0.8rem; border-bottom:1px solid var(--border); }}
  tr:hover {{ background:var(--bg3); }}
  .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:1rem; margin-bottom:1.5rem; }}
  .stat-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:1rem; text-align:center; }}
  .stat-card .value {{ font-size:1.8rem; font-weight:700; color:var(--accent2); }}
  .stat-card .label {{ font-size:0.85rem; color:var(--text2); }}
</style>
</head>
<body>
<div class="header">
  <div class="header-top">
    <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- Causal Reasoning</span></h1>
    {_nav_html("causal.html", stats)}
  </div>
</div>
<div class="container">
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{causal_stats["total_prompts"]}</div><div class="label">Questions</div></div>
    <div class="stat-card"><div class="value">{causal_stats["total_models"]}</div><div class="label">Models Tested</div></div>
    <div class="stat-card"><div class="value">{round(lb[0]["accuracy"]*100, 1) if lb else 0}%</div><div class="label">Best Accuracy</div></div>
  </div>

  <div class="card" style="border-left:3px solid var(--accent2)">
    <h2 style="margin-bottom:0.5rem">About the causal benchmark</h2>
    <p style="color:var(--text2);font-size:0.9rem;margin-bottom:0.75rem;line-height:1.6">
      A standalone benchmark for causal-inference reasoning. <strong style="color:var(--text)">100 multiple-choice questions</strong> arranged as <strong style="color:var(--text)">20 concept bundles × 5 variants</strong>. Each bundle covers one causal-inference pitfall (e.g. confounding, M-bias, Simpson's paradox); the 5 variants stress-test the same concept different ways:
    </p>
    <ul style="color:var(--text2);font-size:0.88rem;margin-left:1.25rem;margin-bottom:0.75rem;line-height:1.7">
      <li><strong style="color:var(--text)">Base.</strong> Narrative scenario combining 2 to 3 interacting causal issues.</li>
      <li><strong style="color:var(--text)">Trap.</strong> Looks like the base concept applies but the obvious answer is wrong.</li>
      <li><strong style="color:var(--text)">Transfer.</strong> Same reasoning translated into a formal DAG with short elimination-style options.</li>
      <li><strong style="color:var(--text)">Numeric.</strong> Multi-step calculation with tables and conditional probabilities.</li>
      <li><strong style="color:var(--text)">Analyst.</strong> Two analysts debate the scenario; pick the most accurate assessment.</li>
    </ul>
    <p style="color:var(--text2);font-size:0.85rem;margin:0">
      Scoring is purely deterministic (multiple choice, no LLM judges needed). <strong style="color:var(--text)">Accuracy</strong> = correct ÷ valid responses. <strong style="color:var(--text)">Errors</strong> are API failures; <strong style="color:var(--text)">Invalid</strong> is the model returning an empty or unextractable response. Both are reported separately.
    </p>
  </div>

  <!-- Top performer per variant (cards explain what each variant tests) -->
  <div class="card">
    <h2 style="margin-bottom:0.5rem">Top performer per variant</h2>
    <p style="color:var(--text2);margin-bottom:1rem;font-size:0.85rem">Each variant tests the same causal-inference concept from a different angle. Best score on each is shown alongside what the variant tests.</p>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:0.75rem">
      {variant_winner_cards}
    </div>
  </div>

  <!-- Per-variant grouped bar chart for the top 10 models -->
  <div class="card">
    <h2 style="margin-bottom:0.5rem">Per-variant accuracy (Top 10)</h2>
    <p style="color:var(--text2);margin-bottom:1rem;font-size:0.85rem">For the top 10 overall models, accuracy on each of the five variants. Bars that drop sharply on Trap or Transfer reveal where a model relies on pattern-matching over structural reasoning.</p>
    <div style="height:420px;position:relative">
      <canvas id="variantChart"></canvas>
    </div>
  </div>

  <!-- Variant heatmap (top 15 models × 5 variants) -->
  <div class="card">
    <h2 style="margin-bottom:0.5rem">Variant heatmap (Top 15)</h2>
    <p style="color:var(--text2);margin-bottom:1rem;font-size:0.85rem">Composite shading per variant for the top 15 overall models. Green ≥80, yellow ≥60, red below.</p>
    <div style="overflow-x:auto">
      <table>
        <thead><tr><th style="color:var(--text2);font-size:0.75rem">Model</th>{var_header}</tr></thead>
        <tbody>{var_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Full leaderboard (deep-dive) -->
  <div class="card">
    <h2>Accuracy leaderboard</h2>
    <p style="color:var(--text2);margin-bottom:1rem;font-size:0.85rem">All models with causal data. Score is correct out of 100. Errors and Invalid columns separate API failures from empty/unextractable responses.</p>
    <table>
      <thead><tr><th style="text-align:center;width:40px">#</th><th>Model</th><th style="text-align:center" title="Correct ÷ valid responses (excludes errors and invalid)">Accuracy</th><th style="text-align:center" title="Correct out of 100 questions">Score</th><th style="text-align:center" title="API failures (rate limit, 4xx/5xx, timeout): operational issue, not model capability">Errors</th><th style="text-align:center" title="Model returned a response but no extractable answer (reasoning model burned all tokens, returned empty)">Invalid</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
<script>
const CAUSAL_DATA = {data_json};
const VARIANT_CHART = {variant_chart_data};
Chart.defaults.color = '#a3a3a3';
Chart.defaults.borderColor = '#333';
Chart.defaults.font.family = "-apple-system, sans-serif";

(function () {{
  const el = document.getElementById('variantChart');
  if (!el) return;
  const variantColors = {{ base:'#6c72ff', trap:'#ef4444', transfer:'#4ecdc4', numeric:'#eab308', analyst:'#f97316' }};
  new Chart(el, {{
    type: 'bar',
    data: {{
      labels: VARIANT_CHART.labels,
      datasets: VARIANT_CHART.datasets.map(ds => ({{
        label: ds.label,
        data: ds.data,
        backgroundColor: (variantColors[ds.label.toLowerCase()] || '#999') + 'cc',
        borderColor: variantColors[ds.label.toLowerCase()] || '#999',
        borderWidth: 1,
        borderRadius: 4,
      }})),
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }},
      scales: {{
        y: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }}, title: {{ display: true, text: 'Accuracy (0 to 100)' }} }},
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""


def generate_generalist_html(stats):
    """Generate the Generalist benchmark detail page. Owns the full leaderboard,
    DeepEval breakdown, difficulty curve, score distribution, efficiency chart,
    and auto-check flags. Mirrors the structure of generate_causal_html()."""
    data_json = json.dumps(stats)
    lb = stats["leaderboard"]
    total_models = stats["total_models"]
    total_prompts = stats["total_prompts"]
    cats = stats["categories"]

    leaderboard_rows = "".join(_leaderboard_row(i, m) for i, m in enumerate(lb))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - Generalist Benchmark</title>
{_slug_seo("generalist.html")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ --bg: #0f1117; --surface: #1a1d27; --surface2: #242836; --border: #2e3345; --text: #e4e7f0; --text2: #8b90a5; --accent: #6c72ff; --accent2: #4ecdc4; --green: #22c55e; --green-mid: #4ade80; --green-light: #86efac; --yellow: #eab308; --red: #ef4444; --orange: #f97316; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',-apple-system,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
  .header {{ background:linear-gradient(135deg,#1a1d27,#242836); border-bottom:1px solid var(--border); padding:1.5rem 2.5rem; }}
  .header-inner {{ max-width:1440px; margin:0 auto; }}
  .header-top {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; margin-bottom:0.25rem; }}
  .header h1 {{ font-size:1.5rem; font-weight:700; letter-spacing:-0.02em; }}
  .header .byline {{ font-size:0.85rem; color:var(--text2); margin:0.2rem 0 0; }}
  .header .meta {{ font-size:0.75rem; color:var(--text2); margin-top:0.5rem; }}
  .nav {{ display:flex; gap:0.25rem; flex-wrap:wrap; }}
  .nav-link {{ color:var(--text2); text-decoration:none; padding:0.45rem 0.85rem; border-radius:6px; font-size:0.85rem; transition:all 0.2s; }}
  .nav-link:hover {{ color:var(--text); background:var(--surface2); }}
  .nav-link.active {{ color:var(--text); background:var(--accent); }}
  .container {{ max-width:1440px; margin:0 auto; padding:1.5rem 1rem; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.5rem; margin-bottom:1.5rem; }}
  .card h2 {{ font-size:1.1rem; margin-bottom:1rem; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th {{ text-align:left; padding:0.6rem 0.5rem; border-bottom:2px solid var(--border); font-weight:600; cursor:pointer; user-select:none; color:var(--text2); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; }}
  th.num {{ text-align:right; }}
  th.desc::after {{ content:' ▼'; font-size:0.7em; }}
  th.asc::after {{ content:' ▲'; font-size:0.7em; }}
  td {{ padding:0.55rem 0.5rem; border-bottom:1px solid var(--border); }}
  td.num {{ text-align:right; }}
  tr.model-row:hover {{ background:var(--surface2); }}
  .rank {{ display:inline-block; width:24px; height:24px; line-height:24px; text-align:center; border-radius:50%; font-size:0.75rem; font-weight:600; background:var(--surface2); color:var(--text2); }}
  .rank-1 {{ background:#facc15; color:#0a0a0a; }}
  .rank-2 {{ background:#cbd5e1; color:#0a0a0a; }}
  .rank-3 {{ background:#fb923c; color:#0a0a0a; }}
  .company-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:0.4rem; vertical-align:middle; }}
  .badge {{ display:inline-block; padding:0.1rem 0.5rem; border-radius:10px; font-size:0.7rem; font-weight:600; }}
  .badge-error {{ background:#7f1d1d; color:#fca5a5; }}
  .badge-flag {{ background:#854d0e; color:#fde68a; }}
  .badge-ok {{ background:var(--surface2); color:var(--text2); }}
  .info-tip {{ display:inline-block; width:14px; height:14px; line-height:14px; text-align:center; border-radius:50%; background:var(--surface2); color:var(--text2); font-size:0.65rem; cursor:help; position:relative; font-weight:400; }}
  .info-tip:hover::after {{ content:attr(data-info); position:absolute; bottom:120%; left:50%; transform:translateX(-50%); background:#1a1d27; color:var(--text); padding:0.5rem 0.75rem; border:1px solid var(--border); border-radius:6px; width:280px; font-size:0.75rem; line-height:1.4; z-index:10; pointer-events:none; white-space:normal; text-transform:none; letter-spacing:0; font-weight:400; }}
  .table-scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
  .table-scroll .col-detail {{ display:none; }}
  .table-scroll.show-all-cols .col-detail {{ display:table-cell; }}
  .col-toggle {{ background:var(--surface2); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:0.4rem 0.75rem; font-size:0.75rem; cursor:pointer; }}
  .chart-container {{ position:relative; height:340px; }}
  details summary {{ list-style:none; }}
  details summary::-webkit-details-marker {{ display:none; }}
  .flags-list {{ max-height:480px; overflow-y:auto; padding-right:0.5rem; }}
  .flag-item {{ padding:0.6rem 0.75rem; border-bottom:1px solid var(--border); font-size:0.8rem; }}
  .flag-id {{ color:var(--accent2); font-weight:600; }}
  .flag-sub {{ color:var(--text2); font-size:0.75rem; }}
  .flag-models {{ margin-top:0.25rem; font-size:0.75rem; color:var(--text2); }}
  .flag-models span {{ color:var(--yellow); }}
  .grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:1rem; }}
  @media (max-width:640px) {{ .header {{ padding:1rem; }} .container {{ padding:1rem 0.75rem; }} .card {{ padding:1rem; }} }}
</style>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- Generalist</span></h1>
      {_nav_html("generalist.html", stats)}
    </div>
    <p class="byline">The breadth benchmark.</p>
    <div class="meta">{total_models} models · {total_prompts} prompts · {len(cats)} categories{f' · Judges: {", ".join(stats["judge_models"])}' if stats.get("judge_models") else ''} · Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>
<div class="container">

  <div class="card" style="border-left:3px solid var(--accent2)">
    <h2 style="margin-bottom:0.5rem">About the Generalist benchmark</h2>
    <p style="color:var(--text2);font-size:0.9rem;margin-bottom:0.75rem;line-height:1.6">
      <strong style="color:var(--text)">{total_prompts} prompts across {len(cats)} categories</strong> covering the everyday use of LLMs as a working tool: practical coding, writing, instruction following, reasoning, calibration, and behavioural pressure. Each response goes through three scoring layers:
    </p>
    <ul style="color:var(--text2);font-size:0.9rem;margin-left:1.25rem;line-height:1.7">
      <li><strong style="color:var(--text)">Auto-checks.</strong> Deterministic heuristics for format, hallucination patterns, sycophancy. Surface as flags.</li>
      <li><strong style="color:var(--text)">LLM judges.</strong> {len(stats.get("judge_models", []))} independent judges score every response 1 to 5, with self-judging prevented.</li>
      <li><strong style="color:var(--text)">DeepEval G-Eval.</strong> Correctness, coherence, instruction-following on 0 to 1.</li>
    </ul>
    <p style="color:var(--text2);font-size:0.85rem;margin:0.75rem 0 0">
      The composite blends judge and DeepEval scores into a single 0 to 100 number. <a href="methodology.html" style="color:var(--accent2)">Methodology &rarr;</a>
    </p>
  </div>

  <!-- Visual overview at the top: difficulty curve + score distribution -->
  <div class="grid-2">
    <div class="card" style="margin-bottom:0">
      <h2>Performance by Difficulty <span class="info-tip" data-info="Composite (0 to 100) for easy, medium, and hard prompts across the top 5 models. Highlights where strong models start to falter.">?</span></h2>
      <div class="chart-container"><canvas id="difficultyChart"></canvas></div>
    </div>
    <div class="card" style="margin-bottom:0">
      <h2>Score Distribution <span class="info-tip" data-info="LLM judge scores (1 to 5) per model. Greener stacks mean more 5/5 responses. Reveals which models cluster at the top vs spread across the scale.">?</span></h2>
      <div class="chart-container"><canvas id="distChart"></canvas></div>
    </div>
  </div>

  <!-- Efficiency: quality vs verbosity tradeoff -->
  <div class="card" style="margin-top:1rem">
    <h2>Efficiency <span class="info-tip" data-info="Quality per token: avg judge score (1 to 5) / log2(avg_tokens). Higher means better quality per output token, useful for cost-aware selection.">?</span></h2>
    <p style="color:var(--text2);font-size:0.85rem;margin:-0.5rem 0 1rem">A unitless metric. Higher means more quality per output token.</p>
    <div class="chart-container"><canvas id="efficiencyChart"></canvas></div>
  </div>

  <!-- Full leaderboard (the deep-dive table) -->
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;flex-wrap:wrap;gap:0.5rem">
      <h2 style="margin-bottom:0">Leaderboard <span class="info-tip" data-info="Ranked by composite score. Click headers to re-sort. Click a row to expand per-judge scores.">?</span></h2>
      <div style="display:flex;gap:0.5rem;align-items:center">
        <button class="col-toggle" id="col-toggle-btn" onclick="this.closest('.card').querySelector('.table-scroll').classList.toggle('show-all-cols');this.textContent=this.textContent==='Show all columns'?'Hide columns':'Show all columns'">Show all columns</button>
        <select id="company-filter" style="background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:0.4rem 0.75rem;font-size:0.8rem;cursor:pointer">
          <option value="">All Companies</option>
        </select>
      </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;margin-bottom:0.75rem;font-size:0.7rem;color:var(--text2)">
      <span style="text-transform:uppercase;letter-spacing:0.05em">Score colour bands</span>
      <span style="display:inline-flex;align-items:center;gap:0.3rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#22c55e"></span>≥95</span>
      <span style="display:inline-flex;align-items:center;gap:0.3rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#4ade80"></span>≥90</span>
      <span style="display:inline-flex;align-items:center;gap:0.3rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#86efac"></span>≥85</span>
      <span style="display:inline-flex;align-items:center;gap:0.3rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#eab308"></span>≥80</span>
      <span style="display:inline-flex;align-items:center;gap:0.3rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#f97316"></span>≥70</span>
      <span style="display:inline-flex;align-items:center;gap:0.3rem"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#ef4444"></span>&lt;70</span>
    </div>
    <div class="table-scroll">
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th style="width:3rem" data-sort="rank" data-type="num">#</th>
            <th data-sort="name" data-type="str">Model</th>
            <th data-sort="company" data-type="str">Company</th>
            <th data-sort="composite" data-type="num" class="desc"><span class="info-tip" data-info="Composite (0 to 100). Weighted blend of LLM-judge score and DeepEval.">General</span></th>
            <th data-sort="score" data-type="num"><span class="info-tip" data-info="Average across LLM judges, 1 to 5 scale.">Judge</span></th>
            <th class="num" data-sort="deepeval" data-type="num"><span class="info-tip" data-info="DeepEval G-Eval (correctness, coherence, instruction-following) averaged into 0 to 100.">DeepEval</span></th>
            <th class="num" data-sort="causal" data-type="num"><span class="info-tip" data-info="Causal benchmark accuracy (0 to 100). Independent of the general score.">Causal</span></th>
            <th class="num" data-sort="errors" data-type="num"><span class="info-tip" data-info="API failures (rate limits, 4xx/5xx, timeouts).">Errors</span></th>
            <th class="num col-detail" data-sort="flags" data-type="num"><span class="info-tip" data-info="Auto-check heuristic flags (sycophancy, hallucination, format violations).">Flags</span></th>
            <th class="num col-detail" data-sort="latency" data-type="num">Avg Latency</th>
            <th class="num col-detail" data-sort="tokens" data-type="num">Avg Tokens</th>
          </tr>
        </thead>
        <tbody>{leaderboard_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- DeepEval breakdown: secondary table data, lives below the leaderboard -->
  {_deepeval_breakdown_card(lb)}

  <details class="card" style="padding:0;margin-top:1rem">
    <summary style="padding:1.25rem 1.5rem;cursor:pointer;display:flex;justify-content:space-between;align-items:center;font-size:1rem;font-weight:600">
      <span>Auto-check flags ({len(stats['flags'])})</span>
      <span style="color:var(--text2);font-size:0.8rem;font-weight:400">Heuristic checks that flagged a model response. Click to expand.</span>
    </summary>
    <div style="padding:0 1.5rem 1.5rem">
      <div class="flags-list">
        {"".join(_flag_item(f) for f in stats['flags'][:30])}
        {f'<div style="padding:0.5rem;color:var(--text2);font-size:0.85rem">...and {len(stats["flags"])-30} more</div>' if len(stats['flags']) > 30 else ''}
      </div>
    </div>
  </details>

</div>

<script type="module">
const DATA = await (await fetch('./data.json')).json();
const lb = DATA.leaderboard;
const cats = DATA.categories;
const COLORS = ['#6c72ff','#4ecdc4','#f97316','#22c55e','#ec4899','#eab308','#8b5cf6','#06b6d4','#ef4444','#84cc16'];
function compositeColor(s) {{
  if (s >= 0.95) return '#22c55e'; if (s >= 0.90) return '#4ade80';
  if (s >= 0.85) return '#86efac'; if (s >= 0.80) return '#eab308';
  if (s >= 0.70) return '#f97316'; return '#ef4444';
}}
Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

// Sortable leaderboard
document.querySelectorAll('#leaderboard-table th[data-sort]').forEach(th => {{
  th.addEventListener('click', () => {{
    const tbody = document.querySelector('#leaderboard-table tbody');
    const rows = [...tbody.querySelectorAll('tr.model-row')];
    const detailRows = new Map();
    [...tbody.querySelectorAll('tr.judge-detail-row')].forEach(d => {{
      detailRows.set(d.dataset.parent, d);
    }});
    const key = th.dataset.sort, type = th.dataset.type;
    const desc = th.classList.contains('desc') ? false : true;
    document.querySelectorAll('#leaderboard-table th').forEach(t => t.classList.remove('asc','desc'));
    th.classList.add(desc ? 'desc' : 'asc');
    rows.sort((a, b) => {{
      let va = a.dataset[key] ?? '', vb = b.dataset[key] ?? '';
      if (type === 'num') {{ va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; return desc ? vb - va : va - vb; }}
      return desc ? vb.localeCompare(va) : va.localeCompare(vb);
    }});
    tbody.innerHTML = '';
    rows.forEach(r => {{
      tbody.appendChild(r);
      const d = detailRows.get(r.dataset.name);
      if (d) tbody.appendChild(d);
    }});
  }});
}});

// Click-to-expand judge detail rows
document.querySelectorAll('tr.model-row').forEach(r => {{
  r.addEventListener('click', () => {{
    const detail = document.querySelector(`tr.judge-detail-row[data-parent="${{r.dataset.name}}"]`);
    if (detail) detail.style.display = (detail.style.display === 'none' || !detail.style.display) ? 'table-row' : 'none';
  }});
}});

// Company filter
const companies = [...new Set(lb.map(m => m.company))].sort();
const filterEl = document.getElementById('company-filter');
companies.forEach(c => {{
  const opt = document.createElement('option');
  opt.value = c; opt.textContent = c;
  filterEl.appendChild(opt);
}});
filterEl.addEventListener('change', e => {{
  const v = e.target.value;
  document.querySelectorAll('tr.model-row').forEach(r => {{
    const show = !v || r.dataset.company === v;
    r.style.display = show ? '' : 'none';
    const d = document.querySelector(`tr.judge-detail-row[data-parent="${{r.dataset.name}}"]`);
    if (d) d.style.display = 'none';
  }});
}});

// Efficiency chart
new Chart(document.getElementById('efficiencyChart'), {{
  type: 'bar',
  data: {{
    labels: [...lb].sort((a,b) => b.efficiency - a.efficiency).map(m => m.name),
    datasets: [{{
      data: [...lb].sort((a,b) => b.efficiency - a.efficiency).map(m => m.efficiency),
      backgroundColor: '#4ecdc4cc', borderColor: '#4ecdc4', borderWidth: 1, borderRadius: 4,
    }}],
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Efficiency (avg judge / log2 tokens)' }} }}, x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }} }} }}
}});

// Difficulty chart
const top5 = lb.slice(0, 5);
const diffs = ['easy','medium','hard'];
const diffColors = {{ easy: '#22c55e', medium: '#eab308', hard: '#ef4444' }};
new Chart(document.getElementById('difficultyChart'), {{
  type: 'bar',
  data: {{
    labels: top5.map(m => m.name),
    datasets: diffs.map(d => ({{
      label: d.charAt(0).toUpperCase() + d.slice(1),
      data: top5.map(m => ((m.diff_composite && m.diff_composite[d]) || 0) * 100),
      backgroundColor: diffColors[d] + 'cc', borderColor: diffColors[d], borderWidth: 1, borderRadius: 4,
    }})),
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }}, scales: {{ y: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }}, x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }} }} }}
}});

// Score distribution chart
new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{
    labels: lb.map(m => m.name),
    datasets: [5,4,3,2,1].map((score, si) => ({{
      label: score + '/5',
      data: lb.map(m => m.score_dist[score] || 0),
      backgroundColor: ['#22c55e','#86efac','#eab308','#f97316','#ef4444'][si] + 'cc',
      borderRadius: 2,
    }})),
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }}, scales: {{ x: {{ stacked: true, ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}, y: {{ stacked: true, beginAtZero: true }} }} }}
}});

// Re-render any chart that lives inside a <details> on first open
document.querySelectorAll('details').forEach(d => {{
  d.addEventListener('toggle', () => {{ if (d.open) window.dispatchEvent(new Event('resize')); }});
}});
</script>
</body>
</html>"""


def generate_dashboard(output_path=None):
    """Main entry point - generate dashboard HTML files."""
    if output_path is None:
        output_path = DASHBOARD_FILE

    if not Path(RESULTS_DIR).exists():
        print("No results directory found.")
        return None

    models = load_all_results()
    if not models:
        print("No model results found.")
        return None

    config = load_config()
    judges_cfg = config.get("judges", [])
    judge_models = [j["model"] for j in judges_cfg]

    eval_cfg = config.get("eval", {})
    benchmarks = eval_cfg.get("benchmarks", {})
    general_file = benchmarks.get("general") if benchmarks else eval_cfg.get("eval_file")
    prompts = load_prompts(general_file)
    composite_config = config.get("composite", {})
    models_cfg = config.get("models", {})
    stats = compute_stats(models, prompts, judge_models=judge_models, composite_config=composite_config, models_cfg=models_cfg)

    # Causal benchmark (optional)
    causal_file = benchmarks.get("causal") if benchmarks else None
    causal_stats = None
    if causal_file and Path(causal_file).exists():
        causal_prompts = load_prompts(causal_file)
        skip_causal = set(
            (config.get("eval", {}).get("benchmark_scoring", {}).get("causal", {}) or {})
            .get("skip_models", []) or []
        )
        causal_models = {n: d for n, d in models.items() if n not in skip_causal}
        causal_stats = compute_causal_stats(causal_models, causal_prompts, models_cfg=models_cfg)
        if causal_stats is not None:
            # Annotate skipped models so the causal page can show them with reasons
            causal_stats["skipped"] = sorted([m for m in skip_causal if m in models])

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Inject causal accuracy into leaderboard entries before serializing data.json,
    # so all pages get a complete payload (otherwise causal_accuracy is missing for the
    # other generators, which all run before generate_html).
    if causal_stats:
        causal_by_name = {m["name"]: m["accuracy"] for m in causal_stats["leaderboard"]}
        for m in stats["leaderboard"]:
            m["causal_accuracy"] = causal_by_name.get(m["name"])

    # Shared dataset for the 5 leaderboard pages. Pages fetch this on load instead
    # of inlining ~6 MB into every HTML file. Cuts each page from 6.8 MB to ~50 KB.
    data_json_path = os.path.join(out_dir or ".", "data.json")
    with open(data_json_path, "w") as f:
        json.dump(stats, f)
    causal_data_path = os.path.join(out_dir or ".", "causal-data.json")
    with open(causal_data_path, "w") as f:
        json.dump(causal_stats or {}, f)

    # Main dashboard
    html = generate_html(stats, causal_stats=causal_stats)
    with open(output_path, "w") as f:
        f.write(html)

    # Categories page
    cat_path = os.path.join(out_dir or ".", "categories.html")
    cat_html = generate_categories_html(stats)
    with open(cat_path, "w") as f:
        f.write(cat_html)

    # Companies page
    companies_path = os.path.join(out_dir or ".", "companies.html")
    companies_html = generate_companies_html(stats)
    with open(companies_path, "w") as f:
        f.write(companies_html)

    # Methodology page
    meth_path = os.path.join(out_dir or ".", "methodology.html")
    meth_html = generate_methodology_html(stats)
    with open(meth_path, "w") as f:
        f.write(meth_html)

    # Judges page
    judges_path = os.path.join(out_dir or ".", "judges.html")
    judges_html = generate_judges_html(stats)
    with open(judges_path, "w") as f:
        f.write(judges_html)

    # Causal reasoning page
    causal_path = os.path.join(out_dir or ".", "causal.html")
    causal_html = generate_causal_html(causal_stats, stats)
    with open(causal_path, "w") as f:
        f.write(causal_html)

    # Generalist (general benchmark) detail page
    gen_path = os.path.join(out_dir or ".", "generalist.html")
    gen_html = generate_generalist_html(stats)
    with open(gen_path, "w") as f:
        f.write(gen_html)

    # Static SEO assets: robots.txt, sitemap.xml, favicon.svg
    write_seo_assets(out_dir or ".")

    return output_path


if __name__ == "__main__":
    path = generate_dashboard()
    if path:
        out_dir = os.path.dirname(path) or "."
        print(f"Dashboard generated: {path}")
        print(f"Companies page generated: {os.path.join(out_dir, 'companies.html')}")
        print(f"Categories page generated: {os.path.join(out_dir, 'categories.html')}")
        print(f"Methodology page generated: {os.path.join(out_dir, 'methodology.html')}")
        print(f"Judges page generated: {os.path.join(out_dir, 'judges.html')}")
