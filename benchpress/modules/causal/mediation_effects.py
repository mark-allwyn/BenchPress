"""B13 total vs direct effect (numeric). In a linear model X affects Y directly
and through a mediator. The total effect is direct + indirect; controlling for
the mediator returns the direct effect. Both recomputed in the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _draw(rng: random.Random) -> dict:
    return {
        "a": round(rng.uniform(0.30, 0.70), 2),  # X -> M
        "b": round(rng.uniform(0.30, 0.70), 2),  # M -> Y
        "c": round(rng.uniform(0.20, 0.50), 2),  # X -> Y (direct)
    }


def _effects(p: dict) -> tuple[float, float]:
    total = round(p["c"] + p["a"] * p["b"], 2)
    return total, p["c"]


def _render(p: dict) -> str:
    return f"""In a standardized linear causal model, a treatment X affects an outcome Y in two ways:
- directly, with path coefficient X->Y = {p['c']:.2f};
- indirectly through a mediator M, with X->M = {p['a']:.2f} and M->Y = {p['b']:.2f}.

Determine:
1. the total causal effect of X on Y, to 2 decimals;
2. the direct effect of X on Y (the effect that remains when M is held fixed / adjusted for), to 2 decimals.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
TOTAL_EFFECT: <number to 2 decimals>
DIRECT_EFFECT: <number to 2 decimals>

WORKED EXAMPLE (unrelated numbers, format only):
TOTAL_EFFECT: 0.55
DIRECT_EFFECT: 0.30"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    total, direct = _effects(p)
    parts = [
        Part("TOTAL_EFFECT", "numeric_tolerance", total, {"tol": 0.02}, ["mediation"]),
        Part("DIRECT_EFFECT", "numeric_tolerance", direct, {"tol": 0.02}, ["mediation"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B13-{draw:04d}",
        module="causal", bundle_id="B13", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["mediation", "direct_effect", "total_effect"],
    )


def verify_mediation_effects(item: Item) -> bool:
    total, direct = _effects(item.gen_params)
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["TOTAL_EFFECT"].expected) - total) > 0.01:
        return False
    if abs(float(parts["DIRECT_EFFECT"].expected) - direct) > 0.01:
        return False
    # Must be a genuine mediation case (indirect path actually present).
    return abs(total - direct) > 0.01
