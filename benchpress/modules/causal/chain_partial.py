"""B19 chain partial correlation (numeric). In a chain X->M->Y with no direct
edge, X and Y are marginally correlated (product of path correlations) but
conditionally independent given M - the mediator screens them off. Recomputed in
the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _draw(rng: random.Random) -> dict:
    return {"r_xm": round(rng.uniform(0.40, 0.80), 2), "r_my": round(rng.uniform(0.40, 0.80), 2)}


def _vals(p: dict) -> tuple[float, float]:
    marginal = round(p["r_xm"] * p["r_my"], 2)
    partial = 0.0  # chain screened off by M
    return marginal, partial


def _render(p: dict) -> str:
    return f"""Three standardized variables form a chain X -> M -> Y. X affects Y only through M (no direct X->Y link). The path correlations are:
- corr(X, M) = {p['r_xm']:.2f}
- corr(M, Y) = {p['r_my']:.2f}

Determine:
1. the marginal correlation between X and Y, to 2 decimals;
2. the partial correlation between X and Y given M, to 2 decimals.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
MARGINAL_CORR: <number to 2 decimals>
PARTIAL_CORR_GIVEN_M: <number to 2 decimals>

WORKED EXAMPLE (unrelated numbers, format only):
MARGINAL_CORR: 0.30
PARTIAL_CORR_GIVEN_M: 0.00"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    marginal, partial = _vals(p)
    parts = [
        Part("MARGINAL_CORR", "numeric_tolerance", marginal, {"tol": 0.02}, ["mediation"]),
        Part("PARTIAL_CORR_GIVEN_M", "numeric_tolerance", partial, {"tol": 0.02}, ["mediation", "d_separation"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B19-{draw:04d}",
        module="causal", bundle_id="B19", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["mediation", "partial_correlation"],
    )


def verify_chain_partial(item: Item) -> bool:
    marginal, partial = _vals(item.gen_params)
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["MARGINAL_CORR"].expected) - marginal) > 0.01:
        return False
    return abs(float(parts["PARTIAL_CORR_GIVEN_M"].expected) - partial) <= 0.01
