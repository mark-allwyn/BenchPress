import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=88):
    return registry.get_module("causal")(seed)


def test_includes_b15_and_b16():
    _, meta = _items()
    assert {"B15", "B16"} <= set(meta.bundles)
    assert len(meta.bundles) >= 16


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_explaining_away_lowers_probability():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B15"):
        g = {p.part_id: p.expected for p in item.parts}
        assert g["EXPLAINING_AWAY"] == "yes"
        assert g["P_A_GIVEN_LIGHT"] > g["P_A_GIVEN_LIGHT_AND_B"]


def test_proxy_does_not_identify():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B16"):
        g = {p.part_id: p.expected for p in item.parts}
        assert g["PROXY_SUFFICIENT"] == "no"
        assert g["IDENTIFIABLE_BY_ADJUSTMENT"] == "no"
