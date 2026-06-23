"""B10 selection / collider bias. X and Y both cause a selection variable S and
are otherwise unrelated. They are independent unconditionally, but conditioning
on S (analysing only the selected sample) induces a spurious association - and
there is no causal effect at all. Verified with networkx d-separation."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part


def _build() -> dict:
    # X -> S <- Y. S is a collider (the selection variable). No X -> Y edge.
    edges = [("_x", "_s"), ("_y", "_s")]
    return {"edges": edges, "x": "_x", "y": "_y", "selection": "_s"}


def _relabel(rng: random.Random, raw: dict) -> dict:
    nodes = sorted({n for e in raw["edges"] for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))
    return {
        "edges": sorted([m[a], m[b]] for a, b in raw["edges"]),
        "x": m[raw["x"]], "y": m[raw["y"]], "selection": m[raw["selection"]],
    }


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

A study only observes units for which {p['selection']} occurred (the sample is selected on {p['selection']}).

Determine:
1. whether {p['x']} and {p['y']} are independent in the full population (ignoring selection) - yes or no;
2. whether {p['x']} and {p['y']} are independent within the selected sample (conditioning on {p['selection']}) - yes or no;
3. whether {p['x']} has any causal effect on {p['y']} - yes or no.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
INDEPENDENT_UNCONDITIONAL: <yes or no>
INDEPENDENT_GIVEN_SELECTION: <yes or no>
CAUSAL_EFFECT_EXISTS: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
INDEPENDENT_UNCONDITIONAL: yes
INDEPENDENT_GIVEN_SELECTION: no
CAUSAL_EFFECT_EXISTS: no"""


def _answers(p: dict) -> dict:
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    x, y, s = p["x"], p["y"], p["selection"]
    return {
        "uncond": "yes" if nx.is_d_separator(G, {x}, {y}, set()) else "no",
        "given_s": "yes" if nx.is_d_separator(G, {x}, {y}, {s}) else "no",
        "effect": "yes" if nx.has_path(G, x, y) else "no",
    }


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _relabel(rng, _build())
    a = _answers(p)
    parts = [
        Part("INDEPENDENT_UNCONDITIONAL", "categorical", a["uncond"], {"vocab": ["yes", "no"]}, ["selection"]),
        Part("INDEPENDENT_GIVEN_SELECTION", "categorical", a["given_s"], {"vocab": ["yes", "no"]}, ["selection", "collider"]),
        Part("CAUSAL_EFFECT_EXISTS", "categorical", a["effect"], {"vocab": ["yes", "no"]}, ["selection"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B10-{draw:04d}",
        module="causal", bundle_id="B10", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["selection", "collider"],
    )


def verify_selection(item: Item) -> bool:
    p = item.gen_params
    a = _answers(p)
    parts = {pt.part_id: pt for pt in item.parts}
    return (
        str(parts["INDEPENDENT_UNCONDITIONAL"].expected).lower() == a["uncond"]
        and str(parts["INDEPENDENT_GIVEN_SELECTION"].expected).lower() == a["given_s"]
        and str(parts["CAUSAL_EFFECT_EXISTS"].expected).lower() == a["effect"]
        # this bundle is the collider-bias case: independent only until you select
        and a["uncond"] == "yes" and a["given_s"] == "no" and a["effect"] == "no"
    )
