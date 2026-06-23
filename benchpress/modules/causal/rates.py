"""B08 rate difference / rate ratio (numeric) from a 2x2 table. Straight
arithmetic, recomputed from the counts in the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _draw(rng: random.Random) -> dict:
    n_e, n_u = rng.randint(80, 200), rng.randint(80, 200)
    p_e = round(rng.uniform(0.30, 0.70), 2)
    p_u = round(rng.uniform(0.10, 0.50), 2)
    return {"a": round(p_e * n_e), "n_e": n_e, "c": round(p_u * n_u), "n_u": n_u}


def _metrics(p: dict) -> tuple[float, float, str]:
    r_e = p["a"] / p["n_e"]
    r_u = p["c"] / p["n_u"]
    higher = "exposed" if r_e > r_u else "unexposed"
    return round(r_e - r_u, 2), round(r_e / r_u, 2), higher


def _render(p: dict) -> str:
    return f"""A marketing team sent a promotional email (exposed) to some customers and not to others (unexposed), then recorded who made a purchase.

- Exposed: {p['a']} of {p['n_e']} customers purchased.
- Unexposed: {p['c']} of {p['n_u']} customers purchased.

Determine:
1. the rate difference (exposed purchase rate minus unexposed rate), as a proportion to 2 decimals;
2. the rate ratio (exposed rate divided by unexposed rate), to 2 decimals;
3. which group had the higher purchase rate (exposed or unexposed).

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
RATE_DIFFERENCE: <number to 2 decimals>
RATE_RATIO: <number to 2 decimals>
HIGHER_GROUP: <exposed or unexposed>

WORKED EXAMPLE (unrelated numbers, format only):
RATE_DIFFERENCE: 0.10
RATE_RATIO: 1.25
HIGHER_GROUP: exposed"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    rd, rr, higher = _metrics(p)
    parts = [
        Part("RATE_DIFFERENCE", "numeric_tolerance", rd, {"tol": 0.02}, ["rates"]),
        Part("RATE_RATIO", "numeric_tolerance", rr, {"tol": 0.05}, ["rates"]),
        Part("HIGHER_GROUP", "categorical", higher, {"vocab": ["exposed", "unexposed"]}, ["rates"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B08-{draw:04d}",
        module="causal", bundle_id="B08", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["rates", "estimation"],
    )


def verify_rates(item: Item) -> bool:
    p = item.gen_params
    rd, rr, higher = _metrics(p)
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["RATE_DIFFERENCE"].expected) - rd) > 0.01:
        return False
    if abs(float(parts["RATE_RATIO"].expected) - rr) > 0.01:
        return False
    if str(parts["HIGHER_GROUP"].expected).lower() != higher:
        return False
    return True
