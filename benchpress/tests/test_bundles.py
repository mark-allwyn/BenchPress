import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _by_bundle(seed=21):
    items, meta = registry.get_module("causal")(seed)
    return items, meta


def test_generator_includes_b01_through_b04():
    _, meta = _by_bundle()
    assert set(meta.bundles) >= {"B01", "B02", "B03", "B04"}


def test_all_items_pass_their_gate_and_lint():
    items, _ = _by_bundle()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_m_bias_gold_adjustment_set_is_empty():
    items, _ = _by_bundle()
    b03 = [i for i in items if i.bundle_id == "B03"]
    assert b03
    for item in b03:
        adj = next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET")
        assert set(adj.expected) == set()  # adjust nothing (collider trap)


def test_simpson_reversal_holds():
    items, _ = _by_bundle()
    b04 = [i for i in items if i.bundle_id == "B04"]
    assert b04
    for item in b04:
        parts = {p.part_id: p.expected for p in item.parts}
        # The within-group winner and the aggregate winner disagree (the paradox).
        assert parts["WITHIN_WINNER"] != parts["AGGREGATE_WINNER"]


def test_simpson_gate_rejects_miskeyed_winner():
    from benchpress.modules.causal.simpson import verify_simpson
    items, _ = _by_bundle()
    item = next(i for i in items if i.bundle_id == "B04")
    p = next(pt for pt in item.parts if pt.part_id == "WITHIN_WINNER")
    object.__setattr__(p, "expected", "neither")
    assert verify_simpson(item) is False


def test_determinism_across_bundles():
    a, _ = _by_bundle(21)
    b, _ = _by_bundle(21)
    assert [(i.item_id, i.prompt) for i in a] == [(i.item_id, i.prompt) for i in b]
