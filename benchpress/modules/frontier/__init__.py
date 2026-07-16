"""The frontier tier: causal-inference tests where Opus 4.8 (+thinking, tools-off,
40k budget) still slips. Each test is an all-or-nothing battery of N sub-questions
scored conjunctively - a small residual per-question error rate (p) is amplified to
a low pass rate (p^N), which is immune to token truncation and tunable via N.

Why so few tests: an extensive probe (16+ judgment types) found a thinking frontier
model is at ceiling (~99-100%) on almost all exactly-gradable causal reasoning - any
task with a clean procedure it just executes, and counting is rescued by tokens. Only
COMPOSITE, multi-condition criteria resist: front-door validity (3 path conditions),
instrument validity at scale (2 conditions on a big graph), and multi-term arithmetic
(total-effect summation). Those are the three tests here.

One registered module ("frontier") emits items across the bundles; per-bundle stats
give a per-test score. Every gold value is computed exactly (graphs.py) - no judge.
MUST be run tools-off, thinking-on, at a fixed generous token budget (40k) so scores
reflect reasoning, never truncation.
"""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.registry import register_module
from benchpress.core.types import Item, ModuleMeta, Part
from benchpress.modules.frontier import graphs as g

VERSION = "2"
N_PER_BUNDLE = 8

# Per-test knobs. nodes/density shape the graph; queries = battery length N, tuned so
# Opus 4.8 (+thinking, 40k) lands ~40% (under the 55% target, with headroom). Measured
# per-question accuracy p -> N ~= ln(0.40)/ln(p). Calibrated via a 25-item confirm.
CONFIG = {
    "FRONTDOOR": {"nodes": 11, "density": 0.32, "queries": 24},  # p~97%
    "IV":        {"nodes": 20, "density": 0.32, "queries": 24},  # p~96%
    "SEM":       {"nodes": 14, "density": 0.34, "queries": 46},  # p~98% (needs enough X,Y pairs)
}

_YN = {"vocab": ["yes", "no"]}
_NUM = {"tol": 0.011}  # exact to 2 decimals


def _edges_str(edges):
    return ", ".join(f"{a}->{b}" for a, b in edges)


def _header(nodes, edges):
    return f"Variables: {', '.join(nodes)}.\nDirected edges: {_edges_str(edges)}.\n\n"


def _mixed(golds, n):
    """Reject degenerate yes/no batteries (all-yes / all-no would be guessable)."""
    floor = max(3, n // 5)
    yes = sum(golds)
    return floor <= yes <= n - floor


def _answer_format(n):
    return ("\n\nANSWER FORMAT - one line each:\n"
            + "\n".join(f"Q{i + 1}: <yes or no>" for i in range(n)))


def _yesno_item(bundle, seed, nodes, edges, preamble, qlines, golds, queries, tag):
    prompt = _header(nodes, edges) + preamble + "\n\n" + "\n".join(qlines) + _answer_format(len(qlines))
    parts = [Part(f"Q{i + 1}", "categorical", "yes" if gold else "no", _YN, [tag])
             for i, gold in enumerate(golds)]
    return Item(
        item_id=f"frontier-{bundle}-{seed:06d}", module="frontier", bundle_id=bundle,
        variant="battery", difficulty="hard",
        gen_params={"nodes": list(nodes), "edges": [list(e) for e in edges], "queries": queries},
        prompt=prompt, parts=parts, skill_tags=[tag],
    )


# ---- per-bundle builders: return an Item, or None if the draw is degenerate ----

def _frontdoor(seed):
    cfg = CONFIG["FRONTDOOR"]
    n = cfg["queries"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    rng = random.Random(seed * 7 + 1)
    dpairs = [(a, b) for a in nodes for b in nodes if a != b and nx.has_path(G, a, b)]
    if not dpairs:
        return None
    qs, golds, qp, tries = [], [], [], 0
    while len(qs) < n and tries < 8000:
        tries += 1
        x, y = rng.choice(dpairs)
        mids = [v for v in nodes if v not in (x, y)]
        onpath = [v for v in mids if nx.has_path(G, x, v) and nx.has_path(G, v, y)]
        pool = onpath if (onpath and rng.random() < 0.6) else mids
        Z = tuple(sorted(rng.sample(pool, rng.randint(1, min(3, len(pool))))))
        qs.append(f"Q{len(qs) + 1}: is Z = {{{', '.join(Z)}}} a valid front-door set for ({x} -> {y})?")
        golds.append(g.frontdoor_set_valid(G, x, y, Z))
        qp.append([x, y, list(Z)])
    if len(qs) < n or not _mixed(golds, n):
        return None
    preamble = (
        "A set Z satisfies the FRONT-DOOR criterion for the effect of X on Y iff: (1) Z intercepts "
        "every directed path from X to Y; (2) there is no unblocked back-door path from X to Z; and "
        "(3) every back-door path from Z to Y is blocked by X. Answer yes if valid, else no.")
    return _yesno_item("FRONTDOOR", seed, nodes, edges, preamble, qs, golds,
                       {"kind": "frontdoor", "q": qp}, "frontdoor")


def _iv(seed):
    cfg = CONFIG["IV"]
    n = cfg["queries"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    rng = random.Random(seed * 7 + 1)
    reach = {v: nx.descendants(G, v) for v in nodes}  # precompute (avoids O(n^3) has_path)
    triples = [(v, x, y) for x in nodes for y in reach[x] for v in nodes if v not in (x, y)]
    if not triples:
        return None
    qs, golds, qp, tries = [], [], [], 0
    while len(qs) < n and tries < 8000:
        tries += 1
        v, x, y = rng.choice(triples)
        qs.append(f"Q{len(qs) + 1}: is {v} a valid instrument for the effect of {x} on {y}?")
        golds.append(g.instrument_valid(G, v, x, y))
        qp.append([v, x, y])
    if len(qs) < n or not _mixed(golds, n):
        return None
    preamble = (
        "V is a valid INSTRUMENT for the effect of X on Y iff: (1) V is associated with X "
        "(V and X are d-connected), and (2) V affects Y only through X - that is, V is d-separated "
        "from Y in the graph obtained by deleting all edges leaving X. Answer yes if valid, else no.")
    return _yesno_item("IV", seed, nodes, edges, preamble, qs, golds,
                       {"kind": "iv", "q": qp}, "iv")


def _sem(seed):
    cfg = CONFIG["SEM"]
    n = cfg["queries"]
    nodes, edges, G, w = g.seeded_weighted_dag(seed, cfg["nodes"], cfg["density"])
    rng = random.Random(seed * 7 + 1)
    pairs = [(a, b) for a in nodes for b in nodes if a != b and nx.has_path(G, a, b)]
    rng.shuffle(pairs)
    qs, golds, qp = [], [], []
    for x, y in pairs:
        npaths = sum(1 for _ in nx.all_simple_paths(G, x, y))
        if not (2 <= npaths <= 30):
            continue
        qs.append(f"Q{len(qs) + 1}: total causal effect of {x} on {y}?")
        golds.append(round(g.total_effect(G, w, x, y), 2))
        qp.append([x, y])
        if len(qs) >= n:
            break
    if len(qs) < n:
        return None
    wl = ", ".join(f"{a}->{b} ({w[(a, b)]:+.1f})" for a, b in edges)
    prompt = (
        f"Variables: {', '.join(nodes)}.\n"
        f"Directed edges with linear structural coefficients: {wl}.\n\n"
        "In this linear structural model, the TOTAL causal effect of X on Y equals the sum, over "
        "every directed path from X to Y, of the product of the coefficients along that path. "
        "Compute it for each question, rounded to 2 decimal places.\n\n"
        + "\n".join(qs)
        + "\n\nANSWER FORMAT - one line each:\n"
        + "\n".join(f"Q{i + 1}: <number to 2 decimals>" for i in range(n)))
    parts = [Part(f"Q{i + 1}", "numeric_tolerance", float(gold), _NUM, ["sem"])
             for i, gold in enumerate(golds)]
    return Item(
        item_id=f"frontier-SEM-{seed:06d}", module="frontier", bundle_id="SEM",
        variant="battery", difficulty="hard",
        gen_params={"nodes": list(nodes), "edges": [list(e) for e in edges],
                    "weights": [[a, b, w[(a, b)]] for a, b in edges],
                    "queries": {"kind": "sem", "q": qp}},
        prompt=prompt, parts=parts, skill_tags=["sem"],
    )


_BUILDERS = {"FRONTDOOR": _frontdoor, "IV": _iv, "SEM": _sem}


@register_module("frontier")
def generate(seed: int, difficulty: str = "hard", n: int = N_PER_BUNDLE):
    """n = items per test. Default keeps test runs fast; the frozen set uses ~25."""
    items: list[Item] = []
    for bi, (bundle, builder) in enumerate(_BUILDERS.items()):
        made, attempt = 0, 0
        base = seed * 1_000_000 + bi * 10_000
        while made < n and attempt < 8000:
            item = builder(base + attempt)
            attempt += 1
            if item is not None:
                items.append(item)
                made += 1
    meta = ModuleMeta(
        name="frontier", version=VERSION, variants=["battery"],
        bundles=list(_BUILDERS.keys()),
        part_types=["categorical", "numeric_tolerance"],
    )
    return items, meta
