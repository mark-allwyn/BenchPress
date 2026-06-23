"""B02 transfer variant: abstract DAG reasoning.

The model is shown a directed graph with abstract node labels and must give the
minimal backdoor adjustment set, a d-separation verdict, and identifiability.
The answer key is verified independently with networkx (dag.py).
"""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def _build(rng: random.Random) -> dict:
    k = rng.choice([2, 3])
    confs = [f"_c{i}" for i in range(k)]
    edges = []
    for c in confs:
        edges += [(c, "_x"), (c, "_y")]
    edges += [("_x", "_y"), ("_x", "_m"), ("_m", "_y"), ("_x", "_k"), ("_y", "_k")]

    nodes = confs + ["_x", "_y", "_m", "_k"]
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))

    g_edges = sorted([m[a], m[b]] for a, b in edges)
    x, y = m["_x"], m["_y"]
    confounders = sorted(m[c] for c in confs)

    a, b = m[confs[0]], m[confs[1]]
    cond = [x] if rng.random() < 0.5 else []
    G = nx.DiGraph([tuple(e) for e in g_edges])
    dsep = "yes" if nx.is_d_separator(G, {a}, {b}, set(cond)) else "no"

    return {
        "edges": g_edges, "x": x, "y": y, "confounders": confounders,
        "dsep_a": a, "dsep_b": b, "dsep_cond": cond, "dsep_answer": dsep,
    }


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    cond = "{" + ", ".join(p["dsep_cond"]) + "}"
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

We want the causal effect of {p['x']} on {p['y']}.

Determine:
1. the minimal set of variables to adjust for (backdoor adjustment set) to identify the effect of {p['x']} on {p['y']};
2. whether {p['dsep_a']} and {p['dsep_b']} are d-separated given {cond};
3. whether the causal effect is identifiable from the diagram.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
ADJUSTMENT_SET: {{comma-separated variable names, or {{}} for the empty set}}
D_SEPARATED: <yes or no>
IDENTIFIABLE: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
ADJUSTMENT_SET: {{V2, V4}}
D_SEPARATED: no
IDENTIFIABLE: yes"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _build(rng)
    parts = [
        Part("ADJUSTMENT_SET", "set_match", set(p["confounders"]), {}, ["dag", "backdoor"]),
        Part("D_SEPARATED", "categorical", p["dsep_answer"], {"vocab": ["yes", "no"]}, ["dag", "d_separation"]),
        Part("IDENTIFIABLE", "categorical", "yes", {"vocab": ["yes", "no"]}, ["identification"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B02-{draw:04d}",
        module="causal", bundle_id="B02", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["dag", "backdoor", "d_separation", "transfer"],
    )


def verify_transfer(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    x, y = p["x"], p["y"]
    parts = {pt.part_id: pt for pt in item.parts}

    gold_set = {str(s) for s in parts["ADJUSTMENT_SET"].expected}
    if set(dag.minimal_backdoor_set(G, x, y) or set()) != gold_set:
        return False
    if not dag.is_minimal_backdoor(G, x, y, gold_set):
        return False

    answer = "yes" if nx.is_d_separator(G, {p["dsep_a"]}, {p["dsep_b"]}, set(p["dsep_cond"])) else "no"
    if str(parts["D_SEPARATED"].expected).lower() != answer:
        return False
    if str(parts["IDENTIFIABLE"].expected).lower() != "yes":
        return False
    return True
