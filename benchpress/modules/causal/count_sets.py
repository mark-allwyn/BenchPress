"""B17 count of minimal adjustment sets. A backdoor path blockable at either of
two nodes yields two distinct minimal adjustment sets. The model must count them
and give the smallest size. Recomputed via the networkx enumerator."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    # X<-A->B->Y backdoor: {A} (fork) or {B} (chain) each block it.
    raw = [("_a", "_x"), ("_a", "_b"), ("_b", "_y"), ("_x", "_y")]
    nodes = sorted({n for e in raw for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))
    edges = sorted([m[a], m[b]] for a, b in raw)
    x, y = m["_x"], m["_y"]
    G = nx.DiGraph([tuple(e) for e in edges])
    sets = dag.all_minimal_backdoor_sets(G, x, y)
    p = {"edges": edges, "x": x, "y": y}
    parts = [
        Part("NUM_MINIMAL_SETS", "numeric_tolerance", float(len(sets)), {"tol": 0.4}, ["backdoor"]),
        Part("SMALLEST_SET_SIZE", "numeric_tolerance",
             float(min(len(s) for s in sets)), {"tol": 0.4}, ["backdoor"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B17-{draw:04d}",
        module="causal", bundle_id="B17", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["backdoor", "adjustment"],
    )


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

We want the causal effect of {p['x']} on {p['y']}.

Determine:
1. how many distinct minimal backdoor adjustment sets exist for this effect (a whole number);
2. the size (number of variables) of the smallest such set (a whole number).

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
NUM_MINIMAL_SETS: <whole number>
SMALLEST_SET_SIZE: <whole number>

WORKED EXAMPLE (unrelated graph, format only):
NUM_MINIMAL_SETS: 1
SMALLEST_SET_SIZE: 2"""


def verify_count_sets(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    sets = dag.all_minimal_backdoor_sets(G, p["x"], p["y"])
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["NUM_MINIMAL_SETS"].expected) - len(sets)) > 0.4:
        return False
    if abs(float(parts["SMALLEST_SET_SIZE"].expected) - min(len(s) for s in sets)) > 0.4:
        return False
    return len(sets) >= 2  # teaching case: more than one minimal set
