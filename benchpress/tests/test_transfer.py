import benchpress.modules.causal  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.transfer import verify_transfer


BACKDOOR_BUNDLES = {"B02", "B03", "B05"}  # transfer items scored via verify_transfer


def _transfer_items(seed=11):
    items, _ = registry.get_module("causal")(seed)
    return [i for i in items if i.bundle_id in BACKDOOR_BUNDLES]


def test_transfer_items_exist_and_have_expected_parts():
    items = _transfer_items()
    assert items
    for item in items:
        ptypes = {p.part_id: p.part_type for p in item.parts}
        assert ptypes["ADJUSTMENT_SET"] == "set_match"
        assert ptypes["D_SEPARATED"] == "categorical"
        assert ptypes["IDENTIFIABLE"] == "categorical"


def test_transfer_items_pass_the_gate():
    for item in _transfer_items():
        assert verify_transfer(item) is True


def test_transfer_gate_rejects_miskeyed_adjustment_set():
    item = _transfer_items()[0]
    adj = next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET")
    object.__setattr__(adj, "expected", set(adj.expected) | {"V99"})
    assert verify_transfer(item) is False


def test_transfer_prompts_are_refusal_neutral():
    for item in _transfer_items():
        assert lint_text(item.prompt) == []


def test_transfer_generation_is_deterministic():
    a = _transfer_items(11)
    b = _transfer_items(11)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    assert [i.prompt for i in a] == [i.prompt for i in b]
