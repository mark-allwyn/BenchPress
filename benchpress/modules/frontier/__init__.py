"""The frontier tier: eight causal-structure tests where even frontier models
slip, because each forces exhaustive reasoning over a graph with no shortcut.

One registered module ("frontier") emits items across eight bundles; per-bundle
stats give a per-test score. Every gold value is computed by an exact algorithm
(graphs.py) - no judge. MUST be run tools-off.
"""

from __future__ import annotations

import random

from benchpress.core.registry import register_module
from benchpress.core.types import Item, ModuleMeta, Part
from benchpress.modules.frontier import graphs as g

VERSION = "1"
N_PER_BUNDLE = 8

# Difficulty knobs per test. nodes/density shape the graph; queries = battery length.
CONFIG = {
    # Hardened (calibration phase): pushed up to pull frontier scores down.
    # Official config = thinking ON. Only "exact count too large to enumerate even
    # carefully" survives thinking, so the suite is exhaustive-counting under load.
    "LINEXT":        {"nodes": 10, "density": 0.30, "gold_min": 1000, "gold_max": 50000},
    "OPENPATH":      {"nodes": 10, "density": 0.44, "gold_min": 25},
    "VSTRUCT":       {"nodes": 24, "density": 0.40, "gold_min": 80},
    "MINSEP_COUNT":  {"nodes": 10, "density": 0.32, "gold_min": 3},
    "DSEP":          {"nodes": 22, "density": 0.19, "queries": 50},
    # Dropped: COMPELLED + MINSEP_SIZE (thinking solves them), MEC (thinking reasons
    # out class size structurally even at scale: 88% with thinking).
}

_NUM = {"tol": 0.4}  # exact integer match
_YN = {"vocab": ["yes", "no"]}


def _edges_str(edges):
    return ", ".join(f"{a}->{b}" for a, b in edges)


def _count_item(bundle, seed, nodes, edges, prompt, gold, tags):
    return Item(
        item_id=f"frontier-{bundle}-{seed:06d}", module="frontier", bundle_id=bundle,
        variant="count", difficulty="hard",
        gen_params={"nodes": list(nodes), "edges": [list(e) for e in edges], "gold": gold},
        prompt=prompt, parts=[Part("ANSWER", "numeric_tolerance", float(gold), _NUM, tags)],
        skill_tags=tags,
    )


def _header(nodes, edges):
    # State the full variable set so isolated nodes are never hidden from the model.
    return (f"Variables: {', '.join(nodes)}.\nDirected edges: {_edges_str(edges)}.\n\n")


def _fmt_int(q):
    return q + "\n\nANSWER FORMAT - reply with exactly this line:\nANSWER: <whole number>"


# ---- per-bundle builders: return an Item, or None if the draw is trivial ----

def _dsep(seed):
    cfg = CONFIG["DSEP"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    rng = random.Random(seed * 7 + 1)
    qs, seen = [], set()
    tries = 0
    while len(qs) < cfg["queries"] and tries < 500:
        tries += 1
        x, y = rng.sample(nodes, 2)
        rest = [v for v in nodes if v not in (x, y)]
        cond = sorted(rng.sample(rest, rng.randint(1, 4)))
        key = (x, y, tuple(cond))
        if key in seen:
            continue
        seen.add(key)
        qs.append((x, y, cond, "yes" if g.d_separated(G, x, y, cond) else "no"))
    qlines = [f"Q{i+1}: are {x} and {y} d-separated given {{{', '.join(c)}}}?"
              for i, (x, y, c, _) in enumerate(qs)]
    prompt = (_header(nodes, edges) + "Answer each d-separation query yes or no:\n" + "\n".join(qlines)
              + "\n\nANSWER FORMAT - one line each:\n"
              + "\n".join(f"Q{i+1}: <yes or no>" for i in range(len(qs))))
    parts = [Part(f"Q{i+1}", "categorical", ans, _YN, ["dsep"]) for i, (_, _, _, ans) in enumerate(qs)]
    return Item(
        item_id=f"frontier-DSEP-{seed:06d}", module="frontier", bundle_id="DSEP",
        variant="battery", difficulty="hard",
        gen_params={"nodes": list(nodes), "edges": [list(e) for e in edges],
                    "queries": [[x, y, c, a] for x, y, c, a in qs]},
        prompt=prompt, parts=parts, skill_tags=["dsep"],
    )


def _linext(seed):
    cfg = CONFIG["LINEXT"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    gold = g.linear_extension_count(G, cap=cfg["gold_max"])
    if not (cfg["gold_min"] <= gold <= cfg["gold_max"]):
        return None
    q = "How many distinct valid topological orderings (linear extensions) does this DAG have?"
    return _count_item("LINEXT", seed, nodes, edges, _header(nodes, edges) + _fmt_int(q), gold, ["linext"])


def _vstruct(seed):
    cfg = CONFIG["VSTRUCT"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    gold = g.vstructure_count(edges, nodes)
    if gold < cfg["gold_min"]:
        return None
    q = ("A v-structure (collider) is a triple a->c<-b where a and b are NOT adjacent. "
         "How many v-structures are in this graph?")
    return _count_item("VSTRUCT", seed, nodes, edges, _header(nodes, edges) + _fmt_int(q), gold, ["vstruct"])


def _openpath(seed):
    cfg = CONFIG["OPENPATH"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    rng = random.Random(seed * 11 + 3)
    x, y = nodes[0], nodes[-1]
    cond = sorted(rng.sample([v for v in nodes if v not in (x, y)], 2))
    gold = g.open_path_count(G, x, y, cond)
    if gold < cfg["gold_min"]:
        return None
    q = (f"How many active (d-connecting) paths are there between {x} and {y} when "
         f"conditioning on {{{', '.join(cond)}}}?")
    item = _count_item("OPENPATH", seed, nodes, edges, _header(nodes, edges) + _fmt_int(q), gold, ["openpath"])
    item.gen_params["cond"] = cond
    return item


def _minsep_count(seed):
    cfg = CONFIG["MINSEP_COUNT"]
    nodes, edges, G = g.seeded_dag(seed, cfg["nodes"], cfg["density"])
    x, y = nodes[0], nodes[-1]
    if y in (set(G.successors(x)) | set(G.predecessors(x))):
        return None
    gold = len(g.minimal_separators(G, x, y))
    if gold < cfg["gold_min"]:
        return None
    q = (f"How many distinct MINIMAL sets of variables d-separate {x} and {y}? "
         "(minimal = no proper subset also d-separates)")
    return _count_item("MINSEP_COUNT", seed, nodes, edges, _header(nodes, edges) + _fmt_int(q), gold, ["minsep"])


_BUILDERS = {
    "DSEP": _dsep, "LINEXT": _linext, "VSTRUCT": _vstruct,
    "OPENPATH": _openpath, "MINSEP_COUNT": _minsep_count,
}


@register_module("frontier")
def generate(seed: int, difficulty: str = "hard"):
    items: list[Item] = []
    for bi, (bundle, builder) in enumerate(_BUILDERS.items()):
        made, attempt = 0, 0
        base = seed * 1_000_000 + bi * 10_000
        while made < N_PER_BUNDLE and attempt < 4000:
            item = builder(base + attempt)
            attempt += 1
            if item is not None:
                items.append(item)
                made += 1
    meta = ModuleMeta(
        name="frontier", version=VERSION, variants=["count", "battery"],
        bundles=list(_BUILDERS.keys()),
        part_types=["numeric_tolerance", "categorical"],
    )
    return items, meta
