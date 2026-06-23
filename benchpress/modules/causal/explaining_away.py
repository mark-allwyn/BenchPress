"""B15 explaining away / Berkson (numeric). Two independent causes of a common
effect become dependent once the effect is observed: learning one cause is
present lowers the probability of the other. Quantities recomputed in the gate."""

from __future__ import annotations

import random

from benchpress.core.types import Item, Part


def _draw(rng: random.Random) -> dict:
    return {"pA": round(rng.uniform(0.10, 0.40), 2), "pB": round(rng.uniform(0.10, 0.40), 2)}


def _vals(p: dict) -> tuple[float, float, str]:
    pA, pB = p["pA"], p["pB"]
    p_a_given_c = round(pA / (pA + pB - pA * pB), 2)   # alert fired
    p_a_given_c_and_b = round(pA, 2)                   # B explains the alert -> back to prior
    explaining = "yes" if p_a_given_c > p_a_given_c_and_b + 0.005 else "no"
    return p_a_given_c, p_a_given_c_and_b, explaining


def _render(p: dict) -> str:
    return f"""Two independent faults can each set off a warning light on a machine. The light turns on if either fault is present (or both).

- Fault A occurs with probability {p['pA']:.2f}.
- Fault B occurs with probability {p['pB']:.2f}.
- The faults occur independently.

Determine (as proportions to 2 decimals where numeric):
1. the probability that fault A is present given that the warning light is on;
2. the probability that fault A is present given that the light is on AND fault B is known to be present;
3. whether learning that B is present (with the light on) lowers the probability of A - yes or no.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
P_A_GIVEN_LIGHT: <proportion to 2 decimals>
P_A_GIVEN_LIGHT_AND_B: <proportion to 2 decimals>
EXPLAINING_AWAY: <yes or no>

WORKED EXAMPLE (unrelated numbers, format only):
P_A_GIVEN_LIGHT: 0.45
P_A_GIVEN_LIGHT_AND_B: 0.20
EXPLAINING_AWAY: yes"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _draw(rng)
    p_c, p_cb, explaining = _vals(p)
    parts = [
        Part("P_A_GIVEN_LIGHT", "numeric_tolerance", p_c, {"tol": 0.02}, ["explaining_away"]),
        Part("P_A_GIVEN_LIGHT_AND_B", "numeric_tolerance", p_cb, {"tol": 0.02}, ["explaining_away"]),
        Part("EXPLAINING_AWAY", "categorical", explaining, {"vocab": ["yes", "no"]}, ["explaining_away", "collider"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B15-{draw:04d}",
        module="causal", bundle_id="B15", variant="numeric", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["explaining_away", "collider", "bayes"],
    )


def verify_explaining_away(item: Item) -> bool:
    p_c, p_cb, explaining = _vals(item.gen_params)
    parts = {pt.part_id: pt for pt in item.parts}
    if abs(float(parts["P_A_GIVEN_LIGHT"].expected) - p_c) > 0.01:
        return False
    if abs(float(parts["P_A_GIVEN_LIGHT_AND_B"].expected) - p_cb) > 0.01:
        return False
    if str(parts["EXPLAINING_AWAY"].expected).lower() != explaining or explaining != "yes":
        return False
    return True
