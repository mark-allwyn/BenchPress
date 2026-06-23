"""B06 base-rate / Bayesian inference (numeric). A reliable flag on a rare
condition is usually a false alarm - the base-rate-neglect trap. The posterior,
flag rate, and verdict are recomputed from the parameters in the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _bayes(base: float, sens: float, fpr: float) -> tuple[float, float]:
    flag = base * sens + (1 - base) * fpr
    posterior = base * sens / flag
    return posterior, flag


def _draw(rng: random.Random) -> dict:
    base = round(rng.uniform(0.05, 0.25), 2)
    sens = round(rng.uniform(0.80, 0.95), 2)
    fpr = round(rng.uniform(0.10, 0.30), 2)
    posterior, flag = _bayes(base, sens, fpr)
    return {
        "base": base, "sens": sens, "fpr": fpr,
        "posterior": round(posterior, 2), "flag_rate": round(flag, 2),
    }


def _render(p: dict) -> str:
    return f"""A factory scanner flags units as defective on a production line.

- {p['base'] * 100:.0f}% of units produced are actually defective (the base rate).
- The scanner flags an actually-defective unit {p['sens'] * 100:.0f}% of the time (sensitivity).
- The scanner wrongly flags a non-defective unit {p['fpr'] * 100:.0f}% of the time (false-positive rate).

Determine:
1. the probability that a unit the scanner flags is actually defective, as a proportion rounded to 2 decimals;
2. the overall probability that a randomly chosen unit gets flagged, as a proportion rounded to 2 decimals;
3. whether a flagged unit is more likely defective than not (yes or no).

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
POSTERIOR: <proportion rounded to 2 decimals>
FLAG_RATE: <proportion rounded to 2 decimals>
MORE_LIKELY_DEFECTIVE: <yes or no>

WORKED EXAMPLE (unrelated numbers, format only):
POSTERIOR: 0.30
FLAG_RATE: 0.20
MORE_LIKELY_DEFECTIVE: no"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    more = "yes" if p["posterior"] > 0.5 else "no"
    parts = [
        Part("POSTERIOR", "numeric_tolerance", p["posterior"], {"tol": 0.02}, ["bayes", "base_rate"]),
        Part("FLAG_RATE", "numeric_tolerance", p["flag_rate"], {"tol": 0.02}, ["bayes"]),
        Part("MORE_LIKELY_DEFECTIVE", "categorical", more, {"vocab": ["yes", "no"]}, ["bayes"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B06-{draw:04d}",
        module="causal", bundle_id="B06", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["bayes", "base_rate", "estimation"],
    )


def verify_base_rate(item: Item) -> bool:
    p = item.gen_params
    posterior, flag = _bayes(p["base"], p["sens"], p["fpr"])
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["POSTERIOR"].expected) - posterior) > 0.02:
        return False
    if abs(float(parts["FLAG_RATE"].expected) - flag) > 0.02:
        return False
    more = "yes" if posterior > 0.5 else "no"
    if str(parts["MORE_LIKELY_DEFECTIVE"].expected).lower() != more:
        return False
    return True
