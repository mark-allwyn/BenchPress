"""The causal-inference module (flagship). Importing it registers the generator."""

from __future__ import annotations

import random

from benchpress.core.registry import register_module
from benchpress.core.types import Item, ModuleMeta, Part
from benchpress.modules.causal import naming, render, scm
from benchpress.modules.causal.verify import verify_item

VERSION = "1"
N_ITEMS = 5


@register_module("causal")
def generate(seed: int, difficulty: str = "hard"):
    rng = random.Random(seed)
    items: list[Item] = []
    draw = 0
    while len(items) < N_ITEMS:
        scenario = naming.pick(rng)
        sc = scm.draw_scenario(rng)
        gold_est = round(scm.partial_regression_coef(sc["r_ty"], sc["r_tz"], sc["r_zy"]), 2)
        gen_params = {
            **sc,
            "confounder": scenario["Z"],
            "treatment": scenario["T"],
            "outcome": scenario["Y"],
            "sim_seed": seed * 1000 + draw,
        }
        parts = [
            Part("ADJUSTMENT_SET", "set_match", {scenario["Z"]}, {}, ["confounding", "backdoor"]),
            Part("ESTIMATE", "numeric_tolerance", gold_est, {"tol": 0.02}, ["confounding", "estimation"]),
            Part("IDENTIFIABLE", "categorical", "yes", {"vocab": ["yes", "no"]}, ["identification"]),
        ]
        item = Item(
            item_id=f"causal-v{VERSION}-B01-{draw:04d}",
            module="causal",
            bundle_id="B01",
            variant="numeric",
            difficulty="hard",
            gen_params=gen_params,
            prompt=render.render(scenario, sc),
            parts=parts,
            skill_tags=["confounding", "backdoor", "estimation", "identification"],
        )
        draw += 1
        if verify_item(item):
            items.append(item)
    meta = ModuleMeta(
        name="causal",
        version=VERSION,
        variants=["numeric"],
        bundles=["B01"],
        part_types=["set_match", "numeric_tolerance", "categorical"],
    )
    return items, meta
