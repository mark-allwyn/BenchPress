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
EVAL_FILE = "evals/default.json"
DOCS_DIR = "docs"
DASHBOARD_FILE = os.path.join(DOCS_DIR, "dashboard.html")
CONFIG_FILE = "config.yaml"


def load_config():
    """Load config.yaml to get judge model name."""
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    return {}


def load_all_results():
    """Load all model result files."""
    models = {}
    for f in sorted(Path(RESULTS_DIR).glob("*.json")):
        if f.stem == "comparison":
            continue
        with open(f) as fh:
            models[f.stem] = json.load(fh)
    return models


def load_prompts():
    with open(EVAL_FILE) as f:
        return json.load(f)["prompts"]


def latest_run(model_data, pid):
    runs = model_data.get("runs", {}).get(pid, [])
    return runs[-1] if runs else {}


def compute_stats(models, prompts, judge_model=None, composite_config=None):
    """Compute all stats needed for the dashboard."""
    judge_weight = (composite_config or {}).get("judge_weight", 0.5)
    deepeval_weight = (composite_config or {}).get("deepeval_weight", 0.5)
    pids = [p["id"] for p in prompts]
    categories = sorted(set(p["category"] for p in prompts))
    cat_pids = {c: [p["id"] for p in prompts if p["category"] == c] for c in categories}

    leaderboard = []
    for name, data in models.items():
        scores, latencies, tokens, errors = [], [], [], 0
        flagged = 0
        de_scores_all = {"correctness": [], "coherence": [], "instruction_following": []}
        de_avgs = []
        for pid in pids:
            run = latest_run(data, pid)
            if not run:
                continue
            if run.get("error"):
                errors += 1
                continue
            if run.get("judge_score") is not None:
                scores.append(run["judge_score"])
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

        total = sum(1 for pid in pids if latest_run(data, pid))
        avg_s = sum(scores) / len(scores) if scores else 0
        avg_l = sum(latencies) / len(latencies) if latencies else 0
        avg_t = sum(tokens) / len(tokens) if tokens else 0
        median_l = sorted(latencies)[len(latencies) // 2] if latencies else 0

        # Category scores
        cat_scores = {}
        cat_deepeval = {}
        cat_composite = {}
        for cat in categories:
            cs = [
                latest_run(data, pid).get("judge_score")
                for pid in cat_pids[cat]
                if latest_run(data, pid) and latest_run(data, pid).get("judge_score") is not None
            ]
            cat_scores[cat] = round(sum(cs) / len(cs), 2) if cs else None
            # DeepEval per-category average
            cat_de = [
                latest_run(data, pid).get("deepeval_avg")
                for pid in cat_pids[cat]
                if latest_run(data, pid) and latest_run(data, pid).get("deepeval_avg") is not None
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

        # Score distribution
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for s in scores:
            dist[s] = dist.get(s, 0) + 1

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
        normalized_judge = (avg_s - 1) / 4 if scores else None
        if normalized_judge is not None and deepeval_avg is not None:
            composite_score = round(judge_weight * normalized_judge + deepeval_weight * deepeval_avg, 4)
        elif normalized_judge is not None:
            composite_score = round(normalized_judge, 4)
        elif deepeval_avg is not None:
            composite_score = round(deepeval_avg, 4)
        else:
            composite_score = None

        leaderboard.append({
            "name": name,
            "avg_score": round(avg_s, 2),
            "scored": len(scores),
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
                fl = run.get("auto_checks", {}).get("flags", [])
                if fl:
                    row[name] = fl
        if row:
            flags.append({"id": pid, "subcategory": p["subcategory"], "models": row})

    return {
        "leaderboard": leaderboard,
        "categories": categories,
        "flags": flags,
        "total_prompts": len(pids),
        "total_models": len(models),
        "judge_model": judge_model,
        "generated": datetime.now().isoformat(),
    }


def generate_html(stats):
    """Generate the full HTML dashboard."""
    data_json = json.dumps(stats)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenchPress - LLM Evaluation Leaderboard</title>
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
    color: var(--text1);
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
    color: var(--text1);
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
  .score-4 {{ color: #86efac; }}
  .score-3 {{ color: var(--yellow); }}
  .score-2 {{ color: var(--orange); }}
  .score-1 {{ color: var(--red); }}
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
  .table-scroll {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
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
  }}
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
      <nav class="nav">
        <a href="dashboard.html" class="nav-link active">Overview</a>
        <a href="categories.html" class="nav-link">By Category</a>
        <a href="methodology.html" class="nav-link">Methodology</a>
      </nav>
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judged by {stats["judge_model"]}' if stats.get("judge_model") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- KPIs -->
<div class="kpi-row">
  <div class="kpi">
    <div class="label">Top Model</div>
    <div class="value" style="font-size:1.3rem">{stats['leaderboard'][0]['name'] if stats['leaderboard'] else '-'}</div>
    <div class="sub">{f"{stats['leaderboard'][0]['composite_score']:.2f}" if stats['leaderboard'] and stats['leaderboard'][0].get('composite_score') is not None else '-'} composite</div>
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
    <div class="label">Total Flags</div>
    <div class="value">{sum(m['flagged'] for m in stats['leaderboard'])}</div>
    <div class="sub">across all models</div>
  </div>
</div>

<!-- Leaderboard + Score Chart -->
<div class="grid-full">
  <div class="card">
    <h2>Leaderboard <span class="info-tip" data-info="Ranked by composite score. Click column headers to re-sort.">?</span></h2>
    <div class="table-scroll">
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th style="width:3rem" data-sort="rank" data-type="num">#</th>
            <th data-sort="name" data-type="str">Model</th>
            <th data-sort="composite" data-type="num" class="desc">Composite</th>
            <th data-sort="score" data-type="num">Judge</th>
            <th class="num" data-sort="deepeval" data-type="num">DeepEval</th>
            <th class="num" data-sort="scored" data-type="num">Scored</th>
            <th class="num" data-sort="errors" data-type="num">Errors</th>
            <th class="num" data-sort="flags" data-type="num">Flags</th>
            <th class="num" data-sort="latency" data-type="num">Avg Latency</th>
            <th class="num" data-sort="tokens" data-type="num">Avg Tokens</th>
            <th class="num" data-sort="efficiency" data-type="num">Efficiency</th>
          </tr>
        </thead>
        <tbody>
          {"".join(_leaderboard_row(i, m) for i, m in enumerate(stats['leaderboard']))}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- DeepEval Breakdown -->
{_deepeval_breakdown_card(stats['leaderboard'])}

<!-- Charts row -->
<div class="grid-2">
  <div class="card">
    <h2>Composite Scores <span class="info-tip" data-info="Weighted average of normalised judge score (0-1) and DeepEval average.">?</span></h2>
    <div class="chart-container">
      <canvas id="scoreChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Efficiency <span class="info-tip" data-info="Quality per token: avg_score / log2(avg_tokens). Higher means better quality with fewer tokens.">?</span></h2>
    <div class="chart-container">
      <canvas id="efficiencyChart"></canvas>
    </div>
  </div>
</div>

<!-- Category breakdown (full width) -->
<div class="grid-full">
  <div class="card">
    <h2>Category Breakdown <span class="info-tip" data-info="Composite score per category. Hover cells for judge and DeepEval breakdown.">?</span></h2>
    <div class="table-scroll">
      <table class="cat-table">
        <thead>
          <tr>
            <th>Category</th>
            {"".join(f'<th class="num">{m["name"]}</th>' for m in stats['leaderboard'])}
          </tr>
        </thead>
        <tbody>
          {"".join(_category_row(cat, stats['leaderboard']) for cat in stats['categories'])}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Radar + Score Distribution -->
<div class="grid-2">
  <div class="card">
    <h2>Category Radar - Top 5 <span class="info-tip" data-info="Composite scores by category for the top 5 models. Wider coverage means more consistent performance.">?</span></h2>
    <div class="chart-container">
      <canvas id="radarChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Score Distribution <span class="info-tip" data-info="LLM judge scores (1-5) per model. More green (5/5) means higher quality responses.">?</span></h2>
    <div class="chart-container">
      <canvas id="distChart"></canvas>
    </div>
  </div>
</div>

<!-- Flags -->
<div class="grid-full">
  <div class="card">
    <h2>Auto-Check Flags ({len(stats['flags'])} prompts flagged) <span class="info-tip" data-info="Automated heuristic checks that flag potential issues like wrong answers, hallucinations, or format violations.">?</span></h2>
    <div class="flags-list">
      {"".join(_flag_item(f) for f in stats['flags'][:30])}
      {f'<div style="padding:0.5rem;color:var(--text2);font-size:0.85rem">...and {len(stats["flags"])-30} more</div>' if len(stats['flags']) > 30 else ''}
    </div>
  </div>
</div>

</div>

<script>
const DATA = {data_json};
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

// Composite score bar chart (0-1 scale)
new Chart(document.getElementById('scoreChart'), {{
  type: 'bar',
  data: {{
    labels: lb.map(m => m.name),
    datasets: [{{
      data: lb.map(m => m.composite_score || 0),
      backgroundColor: lb.map(m => compositeColor(m.composite_score || 0) + 'cc'),
      borderColor: lb.map(m => compositeColor(m.composite_score || 0)),
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ min: 0, max: 1, ticks: {{ stepSize: 0.2 }} }},
      x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}
    }}
  }}
}});

// Efficiency chart - sorted by efficiency descending
const effSorted = [...lb].sort((a, b) => b.efficiency - a.efficiency);
new Chart(document.getElementById('efficiencyChart'), {{
  type: 'bar',
  data: {{
    labels: effSorted.map(m => m.name),
    datasets: [{{
      data: effSorted.map(m => m.efficiency),
      backgroundColor: effSorted.map(m => {{
        if (m.efficiency >= 0.50) return '#22c55ecc';
        if (m.efficiency >= 0.45) return '#4ade80cc';
        if (m.efficiency >= 0.40) return '#86efaccc';
        if (m.efficiency >= 0.35) return '#eab308cc';
        return '#f97316cc';
      }}),
      borderColor: effSorted.map(m => {{
        if (m.efficiency >= 0.50) return '#22c55e';
        if (m.efficiency >= 0.45) return '#4ade80';
        if (m.efficiency >= 0.40) return '#86efac';
        if (m.efficiency >= 0.35) return '#eab308';
        return '#f97316';
      }}),
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ beginAtZero: true, title: {{ display: true, text: 'Efficiency', color: '#8b90a5' }} }},
      x: {{ ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }}
    }}
  }}
}});

// Radar chart (top 5 models) - uses composite per-category scores (0-1 scale)
const top5 = lb.slice(0, 5);
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: cats.map(c => c.replace('_', ' ')),
    datasets: top5.map((m, i) => ({{
      label: m.name,
      data: cats.map(c => (m.cat_composite && m.cat_composite[c]) || 0),
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
        min: 0,
        max: 1,
        ticks: {{ stepSize: 0.2, display: false }},
        grid: {{ color: '#2e3345' }},
        angleLines: {{ color: '#2e3345' }},
        pointLabels: {{ font: {{ size: 11 }}, color: '#e4e7f0' }}
      }}
    }},
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

// Score distribution stacked bar
new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{
    labels: lb.map(m => m.name),
    datasets: [5, 4, 3, 2, 1].map((score, si) => ({{
      label: score + '/5',
      data: lb.map(m => m.score_dist[score] || 0),
      backgroundColor: ['#22c55e', '#86efac', '#eab308', '#f97316', '#ef4444'][si] + 'cc',
      borderRadius: 2,
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
      x: {{ stacked: true, ticks: {{ maxRotation: 45, font: {{ size: 11 }} }} }},
      y: {{ stacked: true, beginAtZero: true }}
    }}
  }}
}});

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

      const rows = Array.from(tbody.querySelectorAll('tr'));
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
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}})();
</script>

</body>
</html>"""


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


def _deepeval_breakdown_card(leaderboard):
    """Generate the DeepEval Breakdown card HTML."""
    # Check if any model has DeepEval data
    has_data = any(m.get("deepeval_avg") is not None for m in leaderboard)
    if not has_data:
        return ""

    metric_names = {"correctness": "Correctness", "coherence": "Coherence", "instruction_following": "Instruction Following"}
    rows = ""
    for i, m in enumerate(leaderboard):
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
    # Composite score (0-1 scale)
    comp_val = m.get("composite_score")
    comp_str = f"{comp_val:.2f}" if comp_val is not None else "-"
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
    de_str = f"{de_val:.2f}" if de_val is not None else "-"
    de_data = f"{de_val}" if de_val is not None else "0"
    de_color = _deepeval_color(de_val)

    return f"""<tr data-rank="{i+1}" data-name="{m['name']}" data-composite="{comp_data}" data-score="{m['avg_score']}" data-deepeval="{de_data}" data-scored="{m['scored']}" data-errors="{m['errors']}" data-flags="{m['flagged']}" data-latency="{m['avg_latency']}" data-tokens="{m['avg_tokens']}" data-efficiency="{m['efficiency']}">
      <td><span class="rank {rank_cls}">{i+1}</span></td>
      <td style="font-weight:600">{m['name']}</td>
      <td class="num" style="font-weight:700;{comp_color}">{comp_str}</td>
      <td class="num {sc}" style="font-weight:600">{m['avg_score']:.2f}/5</td>
      <td class="num" style="font-weight:600;{de_color}">{de_str}</td>
      <td class="num">{m['scored']}/{m['total']}</td>
      <td class="num">{errors_badge}</td>
      <td class="num">{flags_badge}</td>
      <td class="num">{m['avg_latency']:.1f}s</td>
      <td class="num">{m['avg_tokens']:.0f}</td>
      <td class="num" style="font-weight:600;{_efficiency_color(m['efficiency'])}">{m['efficiency']:.2f}</td>
    </tr>"""


def _category_row(cat, leaderboard):
    cells = ""
    for m in leaderboard:
        comp = m.get("cat_composite", {}).get(cat)
        s = m["cat_scores"].get(cat)
        de = m.get("cat_deepeval", {}).get(cat)
        if comp is not None or s is not None:
            comp_color = _composite_color(comp)
            comp_str = f"{comp:.2f}" if comp is not None else "-"
            tip_parts = []
            if s is not None:
                tip_parts.append(f"Judge: {s:.2f}/5")
            if de is not None:
                tip_parts.append(f"DeepEval: {de:.2f}")
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

    # Build winner cards
    winner_cards = ""
    for cat in categories:
        best = None
        best_score = 0
        for m in stats["leaderboard"]:
            s = m.get("cat_composite", {}).get(cat)
            if s is not None and s > best_score:
                best_score = s
                best = m["name"]
        display_cat = cat.replace("_", " ").title()
        winner_cards += f"""<div class="winner-card">
          <div class="winner-cat">{display_cat}</div>
          <div class="winner-name">{best or '-'}</div>
          <div class="winner-score">{best_score:.2f}</div>
        </div>\n"""

    # Build chart canvases
    chart_sections = ""
    for cat in categories:
        display_cat = cat.replace("_", " ").title()
        chart_sections += f"""<div class="card">
      <h2>{display_cat}</h2>
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
      <nav class="nav">
        <a href="dashboard.html" class="nav-link">Overview</a>
        <a href="categories.html" class="nav-link active">By Category</a>
        <a href="methodology.html" class="nav-link">Methodology</a>
      </nav>
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judged by {stats["judge_model"]}' if stats.get("judge_model") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

<!-- Category Winners -->
<div class="winners">
  {winner_cards}
</div>

<!-- Per-category charts -->
<div class="chart-grid">
  {chart_sections}
</div>

</div>

<script>
const DATA = {data_json};
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

cats.forEach(cat => {{
  // Get models with composite scores for this category, sorted descending
  const entries = lb
    .filter(m => m.cat_composite && m.cat_composite[cat] != null)
    .map(m => ({{ name: m.name, score: m.cat_composite[cat] }}))
    .sort((a, b) => b.score - a.score);

  const canvas = document.getElementById('chart-' + cat);
  if (!canvas) return;

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: entries.map(e => e.name),
      datasets: [{{
        data: entries.map(e => e.score),
        backgroundColor: entries.map(e => compositeColor(e.score) + 'cc'),
        borderColor: entries.map(e => compositeColor(e.score)),
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
        x: {{ min: 0, max: 1, ticks: {{ stepSize: 0.2 }} }},
        y: {{ ticks: {{ font: {{ size: 12, weight: '600' }} }} }}
      }}
    }}
  }});
}});
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

    cat_rows = ""
    for cat in sorted(cats):
        display = cat.replace("_", " ").title()
        subcats = sorted(set(p["subcategory"].replace("_", " ") for p in prompts if p["category"] == cat))
        sub_str = ", ".join(subcats)
        cat_rows += f"""<tr>
          <td style="font-weight:600;text-transform:capitalize">{display}</td>
          <td class="num">{cats[cat]}</td>
          <td style="color:var(--text2);font-size:0.8rem">{sub_str}</td>
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

            prompt_cards += f"""<div class="prompt-card">
          <div class="prompt-header">
            <span class="prompt-id">{pid}</span>
            <span class="prompt-subcat">{subcat}</span>
            <span class="prompt-diff" style="color:{diff_color}">{diff}</span>
            <span class="prompt-check">{check}</span>
          </div>
          <div class="prompt-text">{prompt_text}</div>
          <div class="prompt-ideal"><strong>What we look for:</strong> {ideal_text}</div>
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
  .score-4 {{ color: #86efac; }}
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
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div class="header-top">
      <h1>BenchPress <span style="font-weight:400;color:var(--text2)">- LLM Evaluation Leaderboard</span></h1>
      <nav class="nav">
        <a href="dashboard.html" class="nav-link">Overview</a>
        <a href="categories.html" class="nav-link">By Category</a>
        <a href="methodology.html" class="nav-link active">Methodology</a>
      </nav>
    </div>
    <p class="byline">Opinionated in scope. Objective in execution.</p>
    <div class="meta">{stats['total_models']} models &middot; {stats['total_prompts']} prompts &middot; {len(stats['categories'])} categories{f' &middot; Judged by {stats["judge_model"]}' if stats.get("judge_model") else ''} &middot; Updated {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}</div>
  </div>
</div>

<div class="container">

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

<!-- Focus -->
<div class="card">
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
<div class="card">
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

<!-- Two-layer scoring -->
<div class="grid-2">
  <div class="card">
    <h2>Auto-Checks (Layer 1)</h2>
    <p>
      Deterministic, heuristic checks that run instantly on every response.
      These flag mechanical failures and feed into the judge as additional signal.
    </p>
    <table>
      <thead><tr><th>Check Type</th><th class="num">Prompts</th></tr></thead>
      <tbody>{auto_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Judge-Only (Layer 2)</h2>
    <p>
      These check types have no automated heuristic - the LLM judge scores them
      entirely on quality, reasoning, and adherence to criteria.
    </p>
    <table>
      <thead><tr><th>Check Type</th><th class="num">Prompts</th></tr></thead>
      <tbody>{judge_rows}</tbody>
    </table>
  </div>
</div>

<!-- Judge scoring -->
<div class="card">
  <h2>LLM Judge Scoring</h2>
  <p>
    A separate LLM (configured in <code>config.yaml</code>) scores every response on a 1-5 scale.
    The judge receives the original prompt, the ideal answer, the scoring criteria, and any
    auto-check flags. It returns a score and a short rationale.
  </p>
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
<div class="card">
  <h2>DeepEval G-Eval Scoring (Layer 3)</h2>
  <p>
    In addition to the single LLM judge score, each response is scored by
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
    <li>Scores are 0-1 floats (DeepEval's native scale), independent of the 1-5 judge score</li>
    <li>Both scoring systems coexist - DeepEval supplements rather than replaces the judge</li>
    <li>Can be run retroactively on existing results: <code>python run.py deepeval</code></li>
  </ul>
</div>

<!-- Composite score -->
<div class="card">
  <h2>Composite Score</h2>
  <p>
    The composite score merges the LLM Judge score and DeepEval average into a single
    0-1 metric for unified ranking. The judge score is first normalized from its 1-5 scale
    to 0-1 using <code>(judge_score - 1) / 4</code>, then combined with the DeepEval
    average via a configurable weighted average.
  </p>
  <div class="highlight">
    composite = judge_weight &times; normalized_judge + deepeval_weight &times; deepeval_avg
  </div>
  <h3>Fallback behavior</h3>
  <ul>
    <li><strong>Both scores available</strong> - weighted average (default: 50/50)</li>
    <li><strong>Only judge score</strong> - composite = normalized judge score</li>
    <li><strong>Only DeepEval score</strong> - composite = DeepEval average</li>
    <li><strong>Neither</strong> - no composite score</li>
  </ul>
  <p>
    Weights are configurable in <code>config.yaml</code> under the <code>composite:</code> section.
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

<!-- Prompt breakdown -->
<div class="card">
  <h2>Prompt Set Breakdown</h2>
  <table>
    <thead><tr><th>Category</th><th class="num">Prompts</th><th>Subcategories</th></tr></thead>
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

<!-- Questions -->
<div class="section-divider">All {len(prompts)} Questions</div>

{questions_sections}

</div>

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

    # Determine judge model and exclude from leaderboard
    config = load_config()
    judge_model = config.get("judge", {}).get("model")
    if judge_model and judge_model in models:
        del models[judge_model]

    prompts = load_prompts()
    composite_config = config.get("composite", {})
    stats = compute_stats(models, prompts, judge_model=judge_model, composite_config=composite_config)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Main dashboard
    html = generate_html(stats)
    with open(output_path, "w") as f:
        f.write(html)

    # Categories page
    cat_path = os.path.join(out_dir or ".", "categories.html")
    cat_html = generate_categories_html(stats)
    with open(cat_path, "w") as f:
        f.write(cat_html)

    # Methodology page
    meth_path = os.path.join(out_dir or ".", "methodology.html")
    meth_html = generate_methodology_html(stats)
    with open(meth_path, "w") as f:
        f.write(meth_html)

    # Index redirect (for GitHub Pages root URL)
    index_path = os.path.join(out_dir or ".", "index.html")
    with open(index_path, "w") as f:
        f.write('<!DOCTYPE html>\n<meta http-equiv="refresh" content="0;url=dashboard.html">\n')

    return output_path


if __name__ == "__main__":
    path = generate_dashboard()
    if path:
        out_dir = os.path.dirname(path) or "."
        print(f"Dashboard generated: {path}")
        print(f"Categories page generated: {os.path.join(out_dir, 'categories.html')}")
        print(f"Index redirect generated: {os.path.join(out_dir, 'index.html')}")
