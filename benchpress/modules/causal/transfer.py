"""DAG bundles (abstract transfer variant).

Each structure builder returns a raw DAG (with placeholder node names), the
treatment/outcome, the unique minimal backdoor adjustment set by construction,
and a pair to ask a d-separation question about. Nodes are relabeled to abstract
V-names so the model must reason structurally. Answer keys are verified
independently with networkx (dag.py), so a miskey can never ship.

- B02 confounding: gold = the confounders (distractors: mediator, collider).
- B03 M-bias: Z is a collider between two latent causes; gold = adjust nothing
  (conditioning on Z would open a path - the classic trap).
"""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def _confounding(rng: random.Random) -> dict:
    k = rng.choice([2, 3])
    confs = [f"_c{i}" for i in range(k)]
    edges = []
    for c in confs:
        edges += [(c, "_x"), (c, "_y")]
    edges += [("_x", "_y"), ("_x", "_m"), ("_m", "_y"), ("_x", "_k"), ("_y", "_k")]
    return {"edges": edges, "x": "_x", "y": "_y", "gold": set(confs), "pair": (confs[0], confs[1])}


def _m_bias(rng: random.Random) -> dict:
    # U1->X, U1->Z, U2->Z, U2->Y, X->Y. Z is a collider on the X..Y backdoor path,
    # which is already blocked; adjusting Z opens it. Minimal set = {}.
    edges = [("_u1", "_x"), ("_u1", "_z"), ("_u2", "_z"), ("_u2", "_y"), ("_x", "_y")]
    return {"edges": edges, "x": "_x", "y": "_y", "gold": set(), "pair": ("_u1", "_u2")}


def _mediator(rng: random.Random) -> dict:
    # C confounds X,Y; M mediates X->M->Y. Total effect needs adjusting C only -
    # adjusting the mediator M (a descendant of X) is the trap. Minimal set = {C}.
    edges = [("_c", "_x"), ("_c", "_y"), ("_x", "_m"), ("_m", "_y"), ("_x", "_y")]
    return {"edges": edges, "x": "_x", "y": "_y", "gold": {"_c"}, "pair": ("_c", "_m")}


def _synthesis(rng: random.Random) -> dict:
    # Two confounders + mediator + collider + an instrument-like extra parent of X.
    # Gold backdoor set is still exactly the two confounders.
    edges = [
        ("_c1", "_x"), ("_c1", "_y"), ("_c2", "_x"), ("_c2", "_y"),
        ("_x", "_m"), ("_m", "_y"), ("_x", "_y"),
        ("_x", "_k"), ("_y", "_k"), ("_z", "_x"),
    ]
    return {"edges": edges, "x": "_x", "y": "_y", "gold": {"_c1", "_c2"}, "pair": ("_c1", "_z")}


STRUCTURES = {"B02": _confounding, "B03": _m_bias, "B05": _mediator, "B20": _synthesis}


def _relabel(rng: random.Random, raw: dict) -> dict:
    nodes = sorted({n for e in raw["edges"] for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))

    edges = sorted([m[a], m[b]] for a, b in raw["edges"])
    x, y = m[raw["x"]], m[raw["y"]]
    gold = sorted(m[g] for g in raw["gold"])
    a, b = m[raw["pair"][0]], m[raw["pair"][1]]
    cond = [x] if rng.random() < 0.5 else []
    G = nx.DiGraph([tuple(e) for e in edges])
    dsep = "yes" if nx.is_d_separator(G, {a}, {b}, set(cond)) else "no"
    return {
        "edges": edges, "x": x, "y": y, "confounders": gold,
        "dsep_a": a, "dsep_b": b, "dsep_cond": cond, "dsep_answer": dsep,
    }


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    cond = "{" + ", ".join(p["dsep_cond"]) + "}"
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

We want the causal effect of {p['x']} on {p['y']}.

Determine:
1. the minimal set of variables to adjust for (backdoor adjustment set) to identify the effect of {p['x']} on {p['y']}; if no adjustment is needed, answer with the empty set;
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


def make_item(rng: random.Random, bundle: str, draw: int, version: str) -> Item:
    p = _relabel(rng, STRUCTURES[bundle](rng))
    parts = [
        Part("ADJUSTMENT_SET", "set_match", set(p["confounders"]), {}, ["dag", "backdoor"]),
        Part("D_SEPARATED", "categorical", p["dsep_answer"], {"vocab": ["yes", "no"]}, ["dag", "d_separation"]),
        Part("IDENTIFIABLE", "categorical", "yes", {"vocab": ["yes", "no"]}, ["identification"]),
    ]
    return Item(
        item_id=f"causal-v{version}-{bundle}-{draw:04d}",
        module="causal", bundle_id=bundle, variant="transfer", difficulty="hard",
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
