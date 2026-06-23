import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=55):
    return registry.get_module("causal")(seed)


def test_includes_b09_and_b10_and_count():
    items, meta = _items()
    assert {"B09", "B10"} <= set(meta.bundles)
    assert len(meta.bundles) == 10


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_front_door_identifiable_but_not_by_adjustment():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B09"):
        g = {p.part_id: p.expected for p in item.parts}
        assert g["IDENTIFIABLE"] == "yes"
        assert g["IDENTIFIABLE_BY_ADJUSTMENT"] == "no"
        assert len(g["FRONT_DOOR_SET"]) == 1


def test_selection_bias_pattern():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B10"):
        g = {p.part_id: p.expected for p in item.parts}
        # independent until you select on the collider; no real causal effect
        assert g["INDEPENDENT_UNCONDITIONAL"] == "yes"
        assert g["INDEPENDENT_GIVEN_SELECTION"] == "no"
        assert g["CAUSAL_EFFECT_EXISTS"] == "no"


def test_front_door_gate_rejects_wrong_set():
    from benchpress.modules.causal.frontdoor import verify_frontdoor
    items, _ = _items()
    item = next(i for i in items if i.bundle_id == "B09")
    fd = next(p for p in item.parts if p.part_id == "FRONT_DOOR_SET")
    object.__setattr__(fd, "expected", {item.gen_params["x"]})
    assert verify_frontdoor(item) is False
