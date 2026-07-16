import benchpress.modules.simulate  # noqa: F401  (registers "simulate")
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.simulate import CONFIG
from benchpress.modules.simulate import sim

BUNDLES = {"LIFE", "DAYNIGHT", "ECA110", "ECA30"}


def _gen(seed=1):
    return registry.get_module("simulate")(seed)


def test_emits_all_bundles():
    items, meta = _gen()
    assert set(meta.bundles) == BUNDLES
    by = {b: [i for i in items if i.bundle_id == b] for b in meta.bundles}
    for b, lst in by.items():
        assert len(lst) == 8, f"{b} has {len(lst)}"


def test_grid_items_have_one_part_per_row():
    items, _ = _gen()
    for it in items:
        if CONFIG[it.bundle_id]["kind"] == "eca":
            assert len(it.parts) == 1
        else:
            assert len(it.parts) == CONFIG[it.bundle_id]["S"]
        for p in it.parts:
            assert p.part_type == "categorical"


def test_gold_recomputes_from_stored_state():
    # Independently re-run each simulator from stored inputs; must match the key.
    items, _ = _gen()
    for it in items:
        kind = it.gen_params["kind"]
        if kind == "eca":
            row = it.gen_params["row"]
            gold = "".join(map(str, sim.eca(row, it.gen_params["rule"], it.gen_params["steps"])))
            assert gold == it.parts[0].expected
        else:
            grid = [list(r) for r in it.gen_params["initial"]]
            if kind == "brain":
                out = sim.brians_brain(grid, it.gen_params["gens"])
            else:
                out = sim.life_like(grid, it.gen_params["gens"],
                                    set(it.gen_params["born"]), set(it.gen_params["survive"]))
            rows = sim.rows_str(out)
            assert rows == [p.expected for p in it.parts]


def test_prompt_states_exact_step_count():
    items, _ = _gen()
    for it in items:
        assert "EXACTLY" in it.prompt


def test_generation_deterministic():
    a, _ = _gen(3)
    b, _ = _gen(3)
    assert [(i.item_id, i.prompt) for i in a] == [(i.item_id, i.prompt) for i in b]


def test_perfect_response_scores_correct():
    from benchpress.runner.score import score_response
    items, meta = _gen()
    for b in meta.bundles:
        it = next(i for i in items if i.bundle_id == b)
        content = "\n".join(f"{p.part_id}: {p.expected}" for p in it.parts)
        result = score_response(it, content, "end_turn")
        assert result.item_correct is True, f"{b} perfect response did not score correct"


def test_eca_rule110_known_step():
    # 0010000 under Rule 110 (cyclic) -> 0110000
    assert sim.eca([0, 0, 1, 0, 0, 0, 0], 110, 1) == [0, 1, 1, 0, 0, 0, 0]


def test_config_matches_frozen_v1_manifest():
    # v1 is IMMUTABLE: the live CONFIG must match the frozen manifest. If this fails,
    # you changed a frozen knob - that must be a NEW version, not an edit to v1.
    import json
    import pathlib
    manifest = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "modules/simulate/frozen_v1.json").read_text())
    assert set(CONFIG) == set(manifest["tasks"])
    for b, spec in manifest["tasks"].items():
        for key, val in spec.items():
            if key == "name":
                continue
            assert CONFIG[b][key] == val, f"{b}.{key}: live {CONFIG[b].get(key)} != frozen {val}"
