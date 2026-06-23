import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=77):
    return registry.get_module("causal")(seed)


def test_includes_b13_and_b14():
    _, meta = _items()
    assert {"B13", "B14"} <= set(meta.bundles)
    assert len(meta.bundles) >= 14


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_total_effect_exceeds_direct_effect():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B13"):
        g = {p.part_id: p.expected for p in item.parts}
        gp = item.gen_params
        assert g["TOTAL_EFFECT"] == round(gp["c"] + gp["a"] * gp["b"], 2)
        assert g["DIRECT_EFFECT"] == gp["c"]
        assert g["TOTAL_EFFECT"] > g["DIRECT_EFFECT"]


def test_adjustment_drill_has_mixed_verdicts():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B14"):
        verdicts = {p.expected for p in item.parts if p.part_id.endswith("SUFFICIENT")}
        assert verdicts == {"yes", "no"}  # at least one of each


def test_adjustment_drill_gate_rejects_flip():
    from benchpress.modules.causal.adjustment_drill import verify_adjustment_drill
    items, _ = _items()
    item = next(i for i in items if i.bundle_id == "B14")
    p = item.parts[0]
    object.__setattr__(p, "expected", "no" if p.expected == "yes" else "yes")
    assert verify_adjustment_drill(item) is False
