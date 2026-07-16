"""The simulation tier: faithfully run a deterministic machine for N steps and
report the exact final state. This is where a thinking frontier model genuinely
fails (tools-off): there is no shortcut (must simulate), errors compound serially,
and the oracle is a trivially-correct simulator (sim.py) - no judge, no subtle gold.

Confirmed not token-bound: Conway's Life holds its low score when the budget is
raised from 40k to 64k, so the difficulty is real reasoning, not truncation. MUST be
run tools-off at a fixed generous budget (>=48k) with stop_reason verified end_turn.

Each 2D item's parts are the ROWS of the final grid: item_correct = exact grid match
(conjunctive), and the per-row marginal is a graded, non-saturating score with headroom.
"""

from __future__ import annotations

from benchpress.core.registry import register_module
from benchpress.core.types import Item, ModuleMeta, Part
from benchpress.modules.simulate import sim

VERSION = "1"
N_PER_BUNDLE = 8

# FROZEN as Benchpress-Simulate v1 on 2026-07-13 (see frozen_v1.json). These values
# are IMMUTABLE: changing any of them, or adding/removing a task, is a NEW version - it
# must NOT edit v1, so that scores stay comparable across models over time.
# Official run: seed=42, n=25, tools OFF, thinking adaptive/effort high, max_tokens=64000,
# read_timeout 900s. Only trust items with stop_reason == "end_turn".
# Opus 4.8 baseline: LIFE 36% / DAYNIGHT 68% / ECA110 56% / ECA30 88% (exact-match).
CONFIG = {
    "LIFE":     {"kind": "life", "S": 7, "gens": 7, "density": 0.40, "born": [3], "survive": [2, 3],
                 "name": "Conway's Game of Life"},
    "DAYNIGHT": {"kind": "life", "S": 7, "gens": 7, "density": 0.45,
                 "born": [3, 6, 7, 8], "survive": [3, 4, 6, 7, 8], "name": "Day & Night"},
    "ECA110":   {"kind": "eca", "W": 30, "N": 35, "rule": 110, "density": 0.5},
    "ECA30":    {"kind": "eca", "W": 30, "N": 35, "rule": 30, "density": 0.5},
}

_CAT = {}  # categorical exact-match, no synonyms


def _grid_answer_format(S):
    return ("\n\nReply with the final grid as exactly these labelled lines:\n"
            + "\n".join(f"ROW{i + 1}: <{S} digits>" for i in range(S)))


def _grid_item(bundle, seed, initial, gold_rows, header, gen_extra, tag):
    S = len(gold_rows)
    show = "\n".join("".join(str(c) for c in r) for r in initial)
    prompt = header.replace("{GRID}", show) + _grid_answer_format(S)
    parts = [Part(f"ROW{i + 1}", "categorical", gold_rows[i], _CAT, [tag]) for i in range(S)]
    gp = {"kind": CONFIG[bundle]["kind"], "initial": [list(r) for r in initial]}
    gp.update(gen_extra)
    return Item(
        item_id=f"simulate-{bundle}-{seed:06d}", module="simulate", bundle_id=bundle,
        variant="sim", difficulty="hard", gen_params=gp, prompt=prompt, parts=parts, skill_tags=[tag],
    )


def _life(bundle):
    cfg = CONFIG[bundle]
    born, survive = set(cfg["born"]), set(cfg["survive"])
    bs = "".join(map(str, sorted(born)))
    ss = "".join(map(str, sorted(survive)))

    def build(seed):
        S, N = cfg["S"], cfg["gens"]
        grid = sim.seeded_grid(seed, S, cfg["density"])
        gold_rows = sim.rows_str(sim.life_like(grid, N, born, survive))
        header = (
            f"{cfg['name']}, a 2D cellular automaton on a {S}x{S} TOROIDAL grid (edges wrap; each "
            f"cell has 8 neighbours). Rule: a DEAD cell becomes live iff its number of live "
            f"neighbours is in {{{bs}}}; a LIVE cell survives iff its number of live neighbours is in "
            f"{{{ss}}}; otherwise the cell is dead next generation. All cells update simultaneously.\n\n"
            f"Initial grid:\n{{GRID}}\n\nEvolve for EXACTLY {N} generations.")
        return _grid_item(bundle, seed, grid, gold_rows, header,
                          {"gens": N, "born": sorted(born), "survive": sorted(survive)}, bundle.lower())
    return build


def _eca(bundle):
    cfg = CONFIG[bundle]
    rule = cfg["rule"]

    def build(seed):
        W, N = cfg["W"], cfg["N"]
        row = sim.seeded_row(seed, W, cfg["density"])
        gold = "".join(map(str, sim.eca(row, rule, N)))
        table = "\n".join(f"  {l}{c}{r} -> {(rule >> (l*4+c*2+r)) & 1}"
                          for l in (1, 0) for c in (1, 0) for r in (1, 0))
        prompt = (
            f"Elementary cellular automaton, Rule {rule}, CYCLIC boundary (cell 1's left neighbour "
            f"is cell {W}; cell {W}'s right neighbour is cell 1). Each step every cell is replaced "
            f"simultaneously using (left,center,right):\n{table}\n\n"
            f"Initial row ({W} cells): {''.join(map(str,row))}\n\nEvolve for EXACTLY {N} steps.\n\n"
            f"Reply with exactly:\nROW1: <{W} digits>")
        parts = [Part("ROW1", "categorical", gold, _CAT, ["eca"])]
        return Item(
            item_id=f"simulate-{bundle}-{seed:06d}", module="simulate", bundle_id=bundle,
            variant="sim", difficulty="hard",
            gen_params={"kind": "eca", "row": row, "rule": rule, "steps": N},
            prompt=prompt, parts=parts, skill_tags=["eca"])
    return build


_BUILDERS = {
    "LIFE": _life("LIFE"), "DAYNIGHT": _life("DAYNIGHT"),
    "ECA110": _eca("ECA110"), "ECA30": _eca("ECA30"),
}


@register_module("simulate")
def generate(seed: int, difficulty: str = "hard", n: int = N_PER_BUNDLE):
    items: list[Item] = []
    for bi, (bundle, builder) in enumerate(_BUILDERS.items()):
        base = seed * 1_000_000 + bi * 10_000
        for k in range(n):
            items.append(builder(base + k))
    meta = ModuleMeta(
        name="simulate", version=VERSION, variants=["sim"],
        bundles=list(_BUILDERS.keys()), part_types=["categorical"],
    )
    return items, meta
