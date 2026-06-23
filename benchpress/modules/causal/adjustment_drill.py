"""B14 adjustment-sufficiency drill. Given a DAG and three candidate adjustment
sets, judge each one sufficient (valid backdoor set) or not. Every verdict is
checked with networkx satisfies_backdoor."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    # Two confounders, a mediator, a collider.
    raw = [
        ("_c1", "_x"), ("_c1", "_y"), ("_c2", "_x"), ("_c2", "_y"),
        ("_x", "_y"), ("_x", "_m"), ("_m", "_y"), ("_x", "_k"), ("_y", "_k"),
    ]
    nodes = sorted({n for e in raw for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))
    edges = sorted([m[a], m[b]] for a, b in raw)
    x, y = m["_x"], m["_y"]
    G = nx.DiGraph([tuple(e) for e in edges])

    candidates = [
        sorted([m["_c1"], m["_c2"]]),   # sufficient
        [m["_c1"]],                      # insufficient (one backdoor open)
        [m["_m"]],                       # invalid (mediator / descendant)
    ]
    rng.shuffle(candidates)
    p = {"edges": edges, "x": x, "y": y, "candidates": candidates}

    parts = [
        Part(f"SET{i + 1}_SUFFICIENT", "categorical",
             "yes" if dag.satisfies_backdoor(G, x, y, set(c)) else "no",
             {"vocab": ["yes", "no"]}, ["backdoor"])
        for i, c in enumerate(candidates)
    ]
    return Item(
        item_id=f"causal-v{version}-B14-{draw:04d}",
        module="causal", bundle_id="B14", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["backdoor", "adjustment"],
    )


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    sets = "\n".join(
        f"Set {i + 1}: {{" + ", ".join(c) + "}}" for i, c in enumerate(p["candidates"])
    )
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

We want the causal effect of {p['x']} on {p['y']}. For each candidate set below, state whether adjusting for exactly that set is sufficient to identify the effect (a valid backdoor adjustment set): yes or no.

{sets}

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
SET1_SUFFICIENT: <yes or no>
SET2_SUFFICIENT: <yes or no>
SET3_SUFFICIENT: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
SET1_SUFFICIENT: yes
SET2_SUFFICIENT: no
SET3_SUFFICIENT: no"""


def verify_adjustment_drill(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    parts = {pt.part_id: pt for pt in item.parts}
    for i, c in enumerate(p["candidates"]):
        expect = "yes" if dag.satisfies_backdoor(G, p["x"], p["y"], set(c)) else "no"
        if str(parts[f"SET{i + 1}_SUFFICIENT"].expected).lower() != expect:
            return False
    return True
