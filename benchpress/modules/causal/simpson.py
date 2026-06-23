"""B04 Simpson's paradox (numeric). A treatment wins within both subgroups but
loses in the pooled data. Counts are constructed to reverse, and the gate
recomputes every answer from the counts independently (no trust in the
generator's claim)."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _rate(success: int, n: int) -> float:
    return success / n if n else 0.0


def _draw(rng: random.Random) -> dict:
    for _ in range(5000):
        n_at, n_ac = rng.randint(10, 40), rng.randint(60, 120)   # small format: mostly control
        n_bt, n_bc = rng.randint(60, 120), rng.randint(10, 40)   # large format: mostly treatment
        p_ac = rng.uniform(0.60, 0.80)
        p_bc = rng.uniform(0.20, 0.40)
        s_at = round((p_ac + rng.uniform(0.05, 0.15)) * n_at)
        s_ac = round(p_ac * n_ac)
        s_bt = round((p_bc + rng.uniform(0.05, 0.15)) * n_bt)
        s_bc = round(p_bc * n_bc)
        within_t = _rate(s_at, n_at) > _rate(s_ac, n_ac) and _rate(s_bt, n_bt) > _rate(s_bc, n_bc)
        pooled_t = _rate(s_at + s_bt, n_at + n_bt)
        pooled_c = _rate(s_ac + s_bc, n_ac + n_bc)
        if within_t and pooled_c > pooled_t:
            return {
                "n_at": n_at, "s_at": s_at, "n_ac": n_ac, "s_ac": s_ac,
                "n_bt": n_bt, "s_bt": s_bt, "n_bc": n_bc, "s_bc": s_bc,
                "pooled_treatment_rate": round(pooled_t, 2),
            }
    raise RuntimeError("could not construct a Simpson reversal")


def _winners(p: dict) -> tuple[str, str]:
    within_t = (_rate(p["s_at"], p["n_at"]) > _rate(p["s_ac"], p["n_ac"])
                and _rate(p["s_bt"], p["n_bt"]) > _rate(p["s_bc"], p["n_bc"]))
    pooled_t = _rate(p["s_at"] + p["s_bt"], p["n_at"] + p["n_bt"])
    pooled_c = _rate(p["s_ac"] + p["s_bc"], p["n_ac"] + p["n_bc"])
    within = "treatment" if within_t else "control"
    aggregate = "treatment" if pooled_t > pooled_c else "control"
    return within, aggregate


def _render(p: dict) -> str:
    return f"""A retailer trialed a new checkout layout (treatment) against the old layout (control) across two store formats. "Completed" counts customers who finished their purchase.

Small-format stores:
- treatment: {p['s_at']} completed of {p['n_at']}
- control: {p['s_ac']} completed of {p['n_ac']}
Large-format stores:
- treatment: {p['s_bt']} completed of {p['n_bt']}
- control: {p['s_bc']} completed of {p['n_bc']}

Determine:
1. within each store format, which layout has the higher completion rate (it is the same winner in both formats): treatment or control;
2. in the pooled data (all stores combined), which layout has the higher completion rate;
3. the pooled completion rate for the treatment layout, as a proportion rounded to 2 decimals.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
WITHIN_WINNER: <treatment or control>
AGGREGATE_WINNER: <treatment or control>
POOLED_TREATMENT_RATE: <proportion rounded to 2 decimals>

WORKED EXAMPLE (unrelated numbers, format only):
WITHIN_WINNER: treatment
AGGREGATE_WINNER: treatment
POOLED_TREATMENT_RATE: 0.50"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    within, aggregate = _winners(p)
    parts = [
        Part("WITHIN_WINNER", "categorical", within, {"vocab": ["treatment", "control"]}, ["simpson"]),
        Part("AGGREGATE_WINNER", "categorical", aggregate, {"vocab": ["treatment", "control"]}, ["simpson"]),
        Part("POOLED_TREATMENT_RATE", "numeric_tolerance", p["pooled_treatment_rate"], {"tol": 0.02}, ["simpson", "estimation"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B04-{draw:04d}",
        module="causal", bundle_id="B04", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["simpson", "aggregation", "estimation"],
    )


def verify_simpson(item: Item) -> bool:
    p = item.gen_params
    within, aggregate = _winners(p)
    if within == aggregate:  # not a paradox
        return False
    parts = {pt.part_id: pt for pt in item.parts}
    if str(parts["WITHIN_WINNER"].expected).lower() != within:
        return False
    if str(parts["AGGREGATE_WINNER"].expected).lower() != aggregate:
        return False
    pooled_t = _rate(p["s_at"] + p["s_bt"], p["n_at"] + p["n_bt"])
    if abs(float(parts["POOLED_TREATMENT_RATE"].expected) - pooled_t) > 0.01:
        return False
    return True
