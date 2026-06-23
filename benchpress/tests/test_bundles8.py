import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=99):
    return registry.get_module("causal")(seed)


def test_twenty_bundles_complete():
    items, meta = _items()
    assert set(meta.bundles) == {f"B{i:02d}" for i in range(1, 21)}


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_count_sets_reports_two():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B17"):
        g = {p.part_id: p.expected for p in item.parts}
        assert g["NUM_MINIMAL_SETS"] == 2.0
        assert g["SMALLEST_SET_SIZE"] == 1.0


def test_chain_partial_is_zero_and_marginal_is_product():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B19"):
        g = {p.part_id: p.expected for p in item.parts}
        gp = item.gen_params
        assert g["PARTIAL_CORR_GIVEN_M"] == 0.0
        assert abs(g["MARGINAL_CORR"] - round(gp["r_xm"] * gp["r_my"], 2)) < 1e-9


def test_synthesis_gold_is_two_confounders():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B20"):
        adj = next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET")
        assert len(adj.expected) == 2


def test_determinism_full_set():
    a, _ = _items(99)
    b, _ = _items(99)
    assert [(i.item_id, i.prompt) for i in a] == [(i.item_id, i.prompt) for i in b]
