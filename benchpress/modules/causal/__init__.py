"""The causal-inference module (flagship). Importing it registers the generator."""

from __future__ import annotations

import random

from benchpress.core.registry import register_module
from benchpress.core.types import Item, ModuleMeta, Part
from benchpress.modules.causal import base_rate, naming, render, scm, simpson, transfer
from benchpress.modules.causal.verify import verify_item

VERSION = "1"
N_B01 = 3  # numeric confounding (emitted first)
N_B02 = 2  # transfer DAG: confounding
N_B03 = 2  # transfer DAG: M-bias trap
N_B04 = 2  # Simpson's paradox numeric
N_B05 = 2  # transfer DAG: mediator trap
N_B06 = 2  # base-rate / Bayes numeric


@register_module("causal")
def generate(seed: int, difficulty: str = "hard"):
    rng = random.Random(seed)
    items: list[Item] = []
    items.extend(_generate_numeric(rng))
    items.extend(_generate_transfer(rng, "B02", N_B02))
    items.extend(_generate_transfer(rng, "B03", N_B03))
    items.extend(_generate_simpson(rng))
    items.extend(_generate_transfer(rng, "B05", N_B05))
    items.extend(_generate_base_rate(rng))
    meta = ModuleMeta(
        name="causal", version=VERSION, variants=["numeric", "transfer"],
        bundles=["B01", "B02", "B03", "B04", "B05", "B06"],
        part_types=["set_match", "numeric_tolerance", "categorical"],
    )
    return items, meta


def _generate_base_rate(rng: random.Random) -> list[Item]:
    items: list[Item] = []
    draw = 0
    while len(items) < N_B06:
        item = base_rate.make_item(rng, draw, VERSION)
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
