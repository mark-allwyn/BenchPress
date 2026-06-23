"""B09 front-door. A latent confounder blocks backdoor adjustment, but a fully
mediating variable identifies the effect via the front-door criterion. Verified
with the networkx front-door predicate (dag.py)."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def _build() -> dict:
    # X -> M -> Y, latent U -> X, U -> Y. M is the unique front-door set.
    edges = [("_x", "_m"), ("_m", "_y"), ("_u", "_x"), ("_u", "_y")]
    return {"edges": edges, "x": "_x", "y": "_y", "latent": "_u", "mediator": "_m"}


def _relabel(rng: random.Random, raw: dict) -> dict:
    nodes = sorted({n for e in raw["edges"] for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))
    return {
        "edges": sorted([m[a], m[b]] for a, b in raw["edges"]),
        "x": m[raw["x"]], "y": m[raw["y"]],
        "latent": m[raw["latent"]], "mediator": m[raw["mediator"]],
        "observed": sorted(m[n] for n in nodes if n != raw["latent"]),
    }


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

Variable {p['latent']} is unobserved (latent); all other variables are observed. We want the causal effect of {p['x']} on {p['y']}.

Determine:
1. the set of observed variables that satisfies the front-door criterion for the effect of {p['x']} on {p['y']} (answer as a set);
2. whether the effect is identifiable by adjusting (backdoor) for observed variables alone (yes or no);
3. whether the effect is identifiable at all from this diagram (yes or no).

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
FRONT_DOOR_SET: {{comma-separated variable names}}
IDENTIFIABLE_BY_ADJUSTMENT: <yes or no>
IDENTIFIABLE: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
FRONT_DOOR_SET: {{V2}}
IDENTIFIABLE_BY_ADJUSTMENT: no
IDENTIFIABLE: yes"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _relabel(rng, _build())
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    by_adj = "yes" if dag.identifiable_by_adjustment(G, p["x"], p["y"], p["observed"]) else "no"
    parts = [
        Part("FRONT_DOOR_SET", "set_match", {p["mediator"]}, {}, ["front_door"]),
        Part("IDENTIFIABLE_BY_ADJUSTMENT", "categorical", by_adj, {"vocab": ["yes", "no"]}, ["identification"]),
        Part("IDENTIFIABLE", "categorical", "yes", {"vocab": ["yes", "no"]}, ["front_door", "identification"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B09-{draw:04d}",
        module="causal", bundle_id="B09", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["front_door", "identification"],
    )


def verify_frontdoor(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    x, y = p["x"], p["y"]
    parts = {pt.part_id: pt for pt in item.parts}

    gold = {str(s) for s in parts["FRONT_DOOR_SET"].expected}
    if not dag.is_front_door(G, gold, x, y):
        return False
    # uniqueness: no other single observed node is a front-door set
    singles = [n for n in p["observed"] if n not in (x, y) and {n} != gold]
    if any(dag.is_front_door(G, {n}, x, y) for n in singles):
        return False
    by_adj = "yes" if dag.identifiable_by_adjustment(G, x, y, p["observed"]) else "no"
    if str(parts["IDENTIFIABLE_BY_ADJUSTMENT"].expected).lower() != by_adj:
        return False
    if str(parts["IDENTIFIABLE"].expected).lower() != "yes":
        return False
    return True
