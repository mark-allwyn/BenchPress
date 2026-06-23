"""B18 intervention vs observation (numeric). Backdoor adjustment over a binary
confounder: the interventional effect is the stratum effects standardized by the
confounder's distribution. Recomputed in the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _draw(rng: random.Random) -> dict:
    c1 = round(rng.uniform(0.40, 0.70), 2)
    c0 = round(rng.uniform(0.20, 0.50), 2)
    return {
        "pZ": round(rng.uniform(0.30, 0.70), 2),
        "t1": round(c1 + rng.uniform(0.05, 0.20), 2), "c1": c1,
        "t0": round(c0 + rng.uniform(0.05, 0.20), 2), "c0": c0,
    }


def _adjusted(p: dict) -> tuple[float, str]:
    e1 = p["t1"] - p["c1"]
    e0 = p["t0"] - p["c0"]
    adjusted = round(p["pZ"] * e1 + (1 - p["pZ"]) * e0, 2)
    larger = "high" if e1 > e0 else "low"
    return adjusted, larger


def _render(p: dict) -> str:
    return f"""A binary factor Z splits a population: P(Z=high) = {p['pZ']:.2f}. Treatment is assigned differently across Z (so crude comparisons are confounded). Within each stratum the purchase rates are:

Z = high:  treated {p['t1']:.2f}, control {p['c1']:.2f}
Z = low:   treated {p['t0']:.2f}, control {p['c0']:.2f}

Determine:
1. the backdoor-adjusted (interventional) effect of treatment on purchase rate, standardizing over Z, to 2 decimals;
2. which stratum shows the larger treatment effect - high or low.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
ADJUSTED_EFFECT: <number to 2 decimals>
LARGER_EFFECT_STRATUM: <high or low>

WORKED EXAMPLE (unrelated numbers, format only):
ADJUSTED_EFFECT: 0.12
LARGER_EFFECT_STRATUM: high"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    adjusted, larger = _adjusted(p)
    parts = [
        Part("ADJUSTED_EFFECT", "numeric_tolerance", adjusted, {"tol": 0.02}, ["do_calculus"]),
        Part("LARGER_EFFECT_STRATUM", "categorical", larger, {"vocab": ["high", "low"]}, ["do_calculus"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B18-{draw:04d}",
        module="causal", bundle_id="B18", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["do_calculus", "standardization"],
    )


def verify_do_vs_observe(item: Item) -> bool:
    adjusted, larger = _adjusted(item.gen_params)
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["ADJUSTED_EFFECT"].expected) - adjusted) > 0.01:
        return False
    return str(parts["LARGER_EFFECT_STRATUM"].expected).lower() == larger
