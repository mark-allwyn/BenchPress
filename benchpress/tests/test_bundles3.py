import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.iv import verify_iv
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=44):
    return registry.get_module("causal")(seed)


def test_includes_b07_and_b08():
    _, meta = _items()
    assert {"B07", "B08"} <= set(meta.bundles)


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_iv_marks_effect_not_identifiable_by_adjustment():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B07"):
        ident = next(p for p in item.parts if p.part_id == "IDENTIFIABLE_BY_ADJUSTMENT")
        assert ident.expected == "no"  # latent confounder
        inst = next(p for p in item.parts if p.part_id == "INSTRUMENT")
        assert len(inst.expected) == 1


def test_iv_gate_rejects_wrong_instrument():
    items, _ = _items()
    item = next(i for i in items if i.bundle_id == "B07")
    inst = next(p for p in item.parts if p.part_id == "INSTRUMENT")
    object.__setattr__(inst, "expected", {item.gen_params["x"]})  # name X, not the instrument
    assert verify_iv(item) is False


def test_rate_metrics_are_consistent():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B08"):
        gp = item.gen_params
        rd = round(gp["a"] / gp["n_e"] - gp["c"] / gp["n_u"], 2)
        got = next(p for p in item.parts if p.part_id == "RATE_DIFFERENCE").expected
        assert abs(got - rd) <= 0.01
