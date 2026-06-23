"""The causal-inference module (flagship). Importing it registers the generator."""

from __future__ import annotations

import random

from benchpress.core.registry import register_module
from benchpress.core.types import Item, ModuleMeta, Part
from benchpress.modules.causal import (
    adjustment_drill, base_rate, dsep_drill, effect_mod, frontdoor, iv,
    mediation_effects, naming, rates, render, scm, selection, simpson, transfer,
)
from benchpress.modules.causal.verify import verify_item

VERSION = "1"
N_B01 = 3  # numeric confounding (emitted first)
N_B02 = 2  # transfer DAG: confounding
N_B03 = 2  # transfer DAG: M-bias trap
N_B04 = 2  # Simpson's paradox numeric
N_B05 = 2  # transfer DAG: mediator trap
N_B06 = 2  # base-rate / Bayes numeric
N_B07 = 2  # instrumental variables
N_B08 = 2  # rate difference / ratio numeric
N_B09 = 2  # front-door
N_B10 = 2  # selection / collider bias
N_B11 = 2  # effect modification / interaction
N_B12 = 2  # conditional-independence drill
N_B13 = 2  # total vs direct effect (mediation)
N_B14 = 2  # adjustment-sufficiency drill


@register_module("causal")
def generate(seed: int, difficulty: str = "hard"):
    rng = random.Random(seed)
    items: list[Item] = []
    items.extend(_generate_numeric(rng))
    items.extend(_generate_transfer(rng, "B02", N_B02))
    items.extend(_generate_transfer(rng, "B03", N_B03))
    items.extend(_generate_simpson(rng))
    items.extend(_generate_transfer(rng, "B05", N_B05))
    items.extend(_generate_simple(rng, base_rate, N_B06))
    items.extend(_generate_simple(rng, iv, N_B07))
    items.extend(_generate_simple(rng, rates, N_B08))
    items.extend(_generate_simple(rng, frontdoor, N_B09))
    items.extend(_generate_simple(rng, selection, N_B10))
    items.extend(_generate_simple(rng, effect_mod, N_B11))
    items.extend(_generate_simple(rng, dsep_drill, N_B12))
    items.extend(_generate_simple(rng, mediation_effects, N_B13))
    items.extend(_generate_simple(rng, adjustment_drill, N_B14))
    meta = ModuleMeta(
        name="causal", version=VERSION, variants=["numeric", "transfer"],
        bundles=[f"B{i:02d}" for i in range(1, 15)],
        part_types=["set_match", "numeric_tolerance", "categorical"],
    )
    return items, meta


def _generate_simple(rng: random.Random, module, n: int) -> list[Item]:
    items: list[Item] = []
    draw = 0
    while len(items) < n:
        item = module.make_item(rng, draw, VERSION)
        draw += 1
        if verify_item(item):
            items.append(item)
    return items


def _generate_transfer(rng: random.Random, bundle: str, n: int) -> list[Item]:
    items: list[Item] = []
    draw = 0
    while len(items) < n:
        item = transfer.make_item(rng, bundle, draw, VERSION)
        draw += 1
        if verify_item(item):
            items.append(item)
    return items


def _generate_simpson(rng: random.Random) -> list[Item]:
    items: list[Item] = []
    draw = 0
    while len(items) < N_B04:
        item = simpson.make_item(rng, draw, VERSION)
        draw += 1
        if verify_item(item):
            items.append(item)
    return items


def _generate_numeric(rng: random.Random) -> list[Item]:
    items: list[Item] = []
    draw = 0
    while len(items) < N_B01:
        scenario = naming.pick(rng)
        sc = scm.draw_scenario(rng)
        gold_est = round(scm.partial_regression_coef(sc["r_ty"], sc["r_tz"], sc["r_zy"]), 2)
        gen_params = {
            **sc,
            "confounder": scenario["Z"],
            "treatment": scenario["T"],
            "outcome": scenario["Y"],
            "sim_seed": rng.getrandbits(32),
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
    return items
