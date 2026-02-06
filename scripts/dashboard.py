"""Generate a self-contained HTML dashboard from eval results."""

import json
import math
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = "results"
EVAL_FILE = "evals/default.json"
DASHBOARD_FILE = "dashboard.html"


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


def compute_stats(models, prompts):
    """Compute all stats needed for the dashboard."""
    pids = [p["id"] for p in prompts]
    categories = sorted(set(p["category"] for p in prompts))
    cat_pids = {c: [p["id"] for p in prompts if p["category"] == c] for c in categories}

    leaderboard = []
    for name, data in models.items():
        scores, latencies, tokens, errors = [], [], [], 0
        flagged = 0
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

        total = sum(1 for pid in pids if latest_run(data, pid))
        avg_s = sum(scores) / len(scores) if scores else 0
        avg_l = sum(latencies) / len(latencies) if latencies else 0
        avg_t = sum(tokens) / len(tokens) if tokens else 0
        median_l = sorted(latencies)[len(latencies) // 2] if latencies else 0

        # Category scores
        cat_scores = {}
        for cat in categories:
            cs = [
                latest_run(data, pid).get("judge_score")
                for pid in cat_pids[cat]
                if latest_run(data, pid) and latest_run(data, pid).get("judge_score") is not None
            ]
            cat_scores[cat] = round(sum(cs) / len(cs), 2) if cs else None

        # Score distribution
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for s in scores:
            dist[s] = dist.get(s, 0) + 1

        # Efficiency = score / log2(avg_tokens) - rewards high scores with fewer tokens
        if avg_s > 0 and avg_t > 1:
            efficiency = round(avg_s / math.log2(avg_t), 2)
        else:
            efficiency = 0

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
        })

    leaderboard.sort(key=lambda x: (x["scored"] > 0, x["avg_score"]), reverse=True)

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
<title>MarkDown - LLM Evaluation Leaderboard</title>
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
    padding: 2rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }}
  .header .meta {{
    font-size: 0.8rem;
    color: var(--text2);
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
    color: var(--text);
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
  .rank-2 {{ background: linear-gradient(135deg, #94a3b8, #64748b); color: #000; }}
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
    .header {{ padding: 1.5rem 1rem; flex-direction: column; gap: 0.5rem; }}
    .header .meta {{ text-align: left !important; }}
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
  <div>
    <h1>MarkDown - LLM Evaluation Leaderboard</h1>
    <div class="meta">{stats['total_prompts']} prompts across {len(stats['categories'])} categories</div>
  </div>
  <div style="display:flex;align-items:center;gap:1.5rem">
    <nav class="nav">
      <a href="dashboard.html" class="nav-link active">Overview</a>
      <a href="categories.html" class="nav-link">By Category</a>
    </nav>
    <div class="meta" style="text-align:right">
      {stats['total_models']} models evaluated<br>
      Updated: {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}
    </div>
  </div>
</div>

<div class="container">

<!-- KPIs -->
<div class="kpi-row">
  <div class="kpi">
    <div class="label">Top Model</div>
    <div class="value" style="font-size:1.3rem">{stats['leaderboard'][0]['name'] if stats['leaderboard'] else '-'}</div>
    <div class="sub">{stats['leaderboard'][0]['avg_score']}/5 avg score</div>
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
    <h2>Leaderboard</h2>
    <div class="table-scroll">
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th style="width:3rem" data-sort="rank" data-type="num">#</th>
            <th data-sort="name" data-type="str">Model</th>
            <th data-sort="score" data-type="num" class="desc">Score</th>
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

<!-- Charts row -->
<div class="grid-2">
  <div class="card">
    <h2>Overall Scores</h2>
    <div class="chart-container">
      <canvas id="scoreChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Efficiency (Score / log2 Tokens)</h2>
    <div class="chart-container">
      <canvas id="efficiencyChart"></canvas>
    </div>
  </div>
</div>

<!-- Category breakdown (full width) -->
<div class="grid-full">
  <div class="card">
    <h2>Category Breakdown</h2>
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
    <h2>Category Radar - Top 5</h2>
    <div class="chart-container">
      <canvas id="radarChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Score Distribution</h2>
    <div class="chart-container">
      <canvas id="distChart"></canvas>
    </div>
  </div>
</div>

<!-- Flags -->
<div class="grid-full">
  <div class="card">
    <h2>Auto-Check Flags ({len(stats['flags'])} prompts flagged)</h2>
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

Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

// Score bar chart
new Chart(document.getElementById('scoreChart'), {{
  type: 'bar',
  data: {{
    labels: lb.map(m => m.name),
    datasets: [{{
      data: lb.map(m => m.avg_score),
      backgroundColor: lb.map((_, i) => COLORS[i % COLORS.length] + 'cc'),
      borderColor: lb.map((_, i) => COLORS[i % COLORS.length]),
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ min: 0, max: 5, ticks: {{ stepSize: 1 }} }},
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
        if (m.efficiency >= 0.5) return '#4ecdc4cc';
        if (m.efficiency >= 0.4) return '#22c55ecc';
        if (m.efficiency >= 0.3) return '#eab308cc';
        return '#f97316cc';
      }}),
      borderColor: effSorted.map(m => {{
        if (m.efficiency >= 0.5) return '#4ecdc4';
        if (m.efficiency >= 0.4) return '#22c55e';
        if (m.efficiency >= 0.3) return '#eab308';
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

// Radar chart (top 5 models)
const top5 = lb.slice(0, 5);
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: cats.map(c => c.replace('_', ' ')),
    datasets: top5.map((m, i) => ({{
      label: m.name,
      data: cats.map(c => m.cat_scores[c] || 0),
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
        max: 5,
        ticks: {{ stepSize: 1, display: false }},
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


def _leaderboard_row(i, m):
    rank_cls = f"rank-{i+1}" if i < 3 else "rank-n"
    pct = (m["avg_score"] / 5) * 100
    sc = _score_color(m["avg_score"])
    color = ["#fbbf24", "#94a3b8", "#d97706"][i] if i < 3 else "#6c72ff"

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

    return f"""<tr data-rank="{i+1}" data-name="{m['name']}" data-score="{m['avg_score']}" data-scored="{m['scored']}" data-errors="{m['errors']}" data-flags="{m['flagged']}" data-latency="{m['avg_latency']}" data-tokens="{m['avg_tokens']}" data-efficiency="{m['efficiency']}">
      <td><span class="rank {rank_cls}">{i+1}</span></td>
      <td style="font-weight:600">{m['name']}</td>
      <td>
        <div class="score-bar">
          <div class="bar"><div class="fill" style="width:{pct}%;background:{color}"></div></div>
          <div class="val {sc}">{m['avg_score']:.2f}</div>
        </div>
      </td>
      <td class="num">{m['scored']}/{m['total']}</td>
      <td class="num">{errors_badge}</td>
      <td class="num">{flags_badge}</td>
      <td class="num">{m['avg_latency']:.1f}s</td>
      <td class="num">{m['avg_tokens']:.0f}</td>
      <td class="num" style="font-weight:600;color:var(--accent2)">{m['efficiency']:.2f}</td>
    </tr>"""


def _category_row(cat, leaderboard):
    cells = ""
    for m in leaderboard:
        s = m["cat_scores"].get(cat)
        if s is not None:
            cls = _score_color(s)
            cells += f'<td class="num score-cell {cls}">{s:.2f}</td>'
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
            s = m["cat_scores"].get(cat)
            if s is not None and s > best_score:
                best_score = s
                best = m["name"]
        display_cat = cat.replace("_", " ").title()
        winner_cards += f"""<div class="winner-card">
          <div class="winner-cat">{display_cat}</div>
          <div class="winner-name">{best or '-'}</div>
          <div class="winner-score">{best_score:.2f}/5</div>
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
<title>MarkDown - By Category</title>
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
    padding: 2rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
  .header .meta {{ font-size: 0.8rem; color: var(--text2); }}
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
    .header {{ padding: 1.5rem 1rem; flex-direction: column; gap: 0.5rem; }}
    .header .meta {{ text-align: left !important; }}
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
  <div>
    <h1>MarkDown - By Category</h1>
    <div class="meta">Best models per category</div>
  </div>
  <div style="display:flex;align-items:center;gap:1.5rem">
    <nav class="nav">
      <a href="dashboard.html" class="nav-link">Overview</a>
      <a href="categories.html" class="nav-link active">By Category</a>
    </nav>
    <div class="meta" style="text-align:right">
      {stats['total_models']} models evaluated<br>
      Updated: {datetime.fromisoformat(stats['generated']).strftime('%b %d, %Y %H:%M')}
    </div>
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

function scoreColor(s) {{
  if (s >= 4.5) return '#22c55e';
  if (s >= 3.5) return '#86efac';
  if (s >= 2.5) return '#eab308';
  if (s >= 1.5) return '#f97316';
  return '#ef4444';
}}

Chart.defaults.color = '#8b90a5';
Chart.defaults.borderColor = '#2e3345';
Chart.defaults.font.family = "'Inter', sans-serif";

cats.forEach(cat => {{
  // Get models with scores for this category, sorted descending
  const entries = lb
    .filter(m => m.cat_scores[cat] != null)
    .map(m => ({{ name: m.name, score: m.cat_scores[cat] }}))
    .sort((a, b) => b.score - a.score);

  const canvas = document.getElementById('chart-' + cat);
  if (!canvas) return;

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: entries.map(e => e.name),
      datasets: [{{
        data: entries.map(e => e.score),
        backgroundColor: entries.map(e => scoreColor(e.score) + 'cc'),
        borderColor: entries.map(e => scoreColor(e.score)),
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
        x: {{ min: 0, max: 5, ticks: {{ stepSize: 1 }} }},
        y: {{ ticks: {{ font: {{ size: 12, weight: '600' }} }} }}
      }}
    }}
  }});
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

    prompts = load_prompts()
    stats = compute_stats(models, prompts)

    # Main dashboard
    html = generate_html(stats)
    with open(output_path, "w") as f:
        f.write(html)

    # Categories page
    cat_path = os.path.join(os.path.dirname(output_path) or ".", "categories.html")
    cat_html = generate_categories_html(stats)
    with open(cat_path, "w") as f:
        f.write(cat_html)

    return output_path


if __name__ == "__main__":
    path = generate_dashboard()
    if path:
        print(f"Dashboard generated: {path}")
        print(f"Categories page generated: categories.html")
