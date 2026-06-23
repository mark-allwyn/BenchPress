import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.base_rate import verify_base_rate
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=33):
    items, meta = registry.get_module("causal")(seed)
    return items, meta


def test_generator_now_includes_b05_and_b06():
    _, meta = _items()
    assert {"B05", "B06"} <= set(meta.bundles)


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_mediator_gold_adjusts_confounder_not_mediator():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B05"):
        adj = next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET")
        assert len(adj.expected) == 1  # exactly the confounder, never the mediator


def test_base_rate_posterior_is_bayes_consistent():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B06"):
        gp = item.gen_params
        flag = gp["base"] * gp["sens"] + (1 - gp["base"]) * gp["fpr"]
        posterior = gp["base"] * gp["sens"] / flag
        got = next(p for p in item.parts if p.part_id == "POSTERIOR").expected
        assert abs(got - posterior) <= 0.01


def test_base_rate_gate_rejects_miskeyed_posterior():
    items, _ = _items()
    item = next(i for i in items if i.bundle_id == "B06")
    p = next(pt for pt in item.parts if pt.part_id == "POSTERIOR")
    object.__setattr__(p, "expected", p.expected + 0.3)
    assert verify_base_rate(item) is False
