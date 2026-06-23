"""B11 effect modification / interaction (numeric). A treatment helps one
subgroup much more than another, so a single average effect hides the
interaction. Subgroup effects and the interaction verdict are recomputed from
the counts in the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _draw(rng: random.Random) -> dict:
    for _ in range(2000):
        n_nt, n_nc = rng.randint(60, 150), rng.randint(60, 150)
        n_rt, n_rc = rng.randint(60, 150), rng.randint(60, 150)
        p_nc, p_rc = rng.uniform(0.30, 0.50), rng.uniform(0.30, 0.50)
        d_new = rng.uniform(0.15, 0.30)
        d_ret = rng.uniform(0.00, 0.08)
        p = {
            "s_nt": round((p_nc + d_new) * n_nt), "n_nt": n_nt,
            "s_nc": round(p_nc * n_nc), "n_nc": n_nc,
            "s_rt": round((p_rc + d_ret) * n_rt), "n_rt": n_rt,
            "s_rc": round(p_rc * n_rc), "n_rc": n_rc,
        }
        e_new, e_ret, _ = _effects(p)
        if abs(e_new - e_ret) > 0.05:  # a clear interaction
            return p
    raise RuntimeError("could not construct an interaction")


def _effects(p: dict) -> tuple[float, float, str]:
    e_new = round(p["s_nt"] / p["n_nt"] - p["s_nc"] / p["n_nc"], 2)
    e_ret = round(p["s_rt"] / p["n_rt"] - p["s_rc"] / p["n_rc"], 2)
    interaction = "yes" if abs(e_new - e_ret) > 0.05 else "no"
    return e_new, e_ret, interaction


def _render(p: dict) -> str:
    return f"""A retailer offered a discount (treatment) and measured purchase rates against no discount (control), split by customer type.

New customers:
- discount: {p['s_nt']} of {p['n_nt']} purchased
- no discount: {p['s_nc']} of {p['n_nc']} purchased
Returning customers:
- discount: {p['s_rt']} of {p['n_rt']} purchased
- no discount: {p['s_rc']} of {p['n_rc']} purchased

Determine:
1. the effect of the discount (purchase-rate difference, treatment minus control) for new customers, to 2 decimals;
2. the same effect for returning customers, to 2 decimals;
3. whether the discount's effect differs between the two customer types (is there effect modification?) - yes or no.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
EFFECT_NEW: <number to 2 decimals>
EFFECT_RETURNING: <number to 2 decimals>
INTERACTION_PRESENT: <yes or no>

WORKED EXAMPLE (unrelated numbers, format only):
EFFECT_NEW: 0.20
EFFECT_RETURNING: 0.05
INTERACTION_PRESENT: yes"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    e_new, e_ret, interaction = _effects(p)
    parts = [
        Part("EFFECT_NEW", "numeric_tolerance", e_new, {"tol": 0.02}, ["interaction"]),
        Part("EFFECT_RETURNING", "numeric_tolerance", e_ret, {"tol": 0.02}, ["interaction"]),
        Part("INTERACTION_PRESENT", "categorical", interaction, {"vocab": ["yes", "no"]}, ["interaction"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B11-{draw:04d}",
        module="causal", bundle_id="B11", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["interaction", "effect_modification"],
    )


def verify_effect_mod(item: Item) -> bool:
    p = item.gen_params
    e_new, e_ret, interaction = _effects(p)
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["EFFECT_NEW"].expected) - e_new) > 0.01:
        return False
    if abs(float(parts["EFFECT_RETURNING"].expected) - e_ret) > 0.01:
        return False
    if str(parts["INTERACTION_PRESENT"].expected).lower() != interaction or interaction != "yes":
        return False
    return True
