"""B07 instrumental variables. A latent confounder makes the effect
unidentifiable by adjustment, but an instrument identifies it. The model must
name the instrument and recognise that adjustment alone fails. Verified with the
networkx instrument/identifiability predicates (dag.py)."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def _build() -> dict:
    # Z instrument -> X; U latent confounder -> X, Y; X -> Y; W observed cause of Y.
    edges = [("_z", "_x"), ("_u", "_x"), ("_u", "_y"), ("_x", "_y"), ("_w", "_y")]
    return {"edges": edges, "x": "_x", "y": "_y", "latent": "_u", "instrument": "_z"}


def _relabel(rng: random.Random, raw: dict) -> dict:
    nodes = sorted({n for e in raw["edges"] for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))
    return {
        "edges": sorted([m[a], m[b]] for a, b in raw["edges"]),
        "x": m[raw["x"]], "y": m[raw["y"]],
        "latent": m[raw["latent"]], "instrument": m[raw["instrument"]],
        "observed": sorted(m[n] for n in nodes if n != raw["latent"]),
    }


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

Variable {p['latent']} is unobserved (latent); all other variables are observed. We want the causal effect of {p['x']} on {p['y']}.

Determine:
1. which single observed variable is a valid instrument for the effect of {p['x']} on {p['y']} (answer as a set, e.g. {{V1}});
2. whether the effect is identifiable by adjusting for observed variables alone (yes or no).

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
INSTRUMENT: {{a single variable name}}
IDENTIFIABLE_BY_ADJUSTMENT: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
INSTRUMENT: {{V3}}
IDENTIFIABLE_BY_ADJUSTMENT: no"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _relabel(rng, _build())
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    ident = "yes" if dag.identifiable_by_adjustment(G, p["x"], p["y"], p["observed"]) else "no"
    parts = [
        Part("INSTRUMENT", "set_match", {p["instrument"]}, {}, ["iv", "instrument"]),
        Part("IDENTIFIABLE_BY_ADJUSTMENT", "categorical", ident, {"vocab": ["yes", "no"]}, ["iv", "identification"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B07-{draw:04d}",
        module="causal", bundle_id="B07", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["iv", "instrument", "identification"],
    )


def verify_iv(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    x, y = p["x"], p["y"]
    parts = {pt.part_id: pt for pt in item.parts}

    gold = {str(s) for s in parts["INSTRUMENT"].expected}
    if not gold or not all(dag.is_instrument(G, z, x, y) for z in gold):
        return False
    others = [n for n in p["observed"] if n not in gold and n not in (x, y)]
    if any(dag.is_instrument(G, z, x, y) for z in others):  # instrument must be unique
        return False
    ident = "yes" if dag.identifiable_by_adjustment(G, x, y, p["observed"]) else "no"
    if str(parts["IDENTIFIABLE_BY_ADJUSTMENT"].expected).lower() != ident:
        return False
    return True
