import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text
from benchpress.modules.causal.verify import verify_item


def _items(seed=66):
    return registry.get_module("causal")(seed)


def test_includes_b11_and_b12_for_twelve_bundles():
    _, meta = _items()
    assert {"B11", "B12"} <= set(meta.bundles)
    assert len(meta.bundles) >= 12


def test_all_items_pass_gate_and_lint():
    items, _ = _items()
    for item in items:
        assert verify_item(item) is True, item.item_id
        assert lint_text(item.prompt) == []


def test_interaction_present_and_effects_differ():
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B11"):
        g = {p.part_id: p.expected for p in item.parts}
        assert g["INTERACTION_PRESENT"] == "yes"
        assert abs(g["EFFECT_NEW"] - g["EFFECT_RETURNING"]) > 0.05


def test_dsep_drill_has_three_queries_matching_networkx():
    import networkx as nx
    items, _ = _items()
    for item in (i for i in items if i.bundle_id == "B12"):
        assert len([p for p in item.parts if p.part_id.startswith("Q")]) == 3
        G = nx.DiGraph([tuple(e) for e in item.gen_params["edges"]])
        G.add_nodes_from(item.gen_params["nodes"])
        for i, q in enumerate(item.gen_params["queries"], 1):
            expect = "yes" if nx.is_d_separator(G, {q["a"]}, {q["b"]}, set(q["cond"])) else "no"
            got = next(p for p in item.parts if p.part_id == f"Q{i}").expected
            assert got == expect


def test_dsep_drill_gate_rejects_flipped_answer():
    from benchpress.modules.causal.dsep_drill import verify_dsep_drill
    items, _ = _items()
    item = next(i for i in items if i.bundle_id == "B12")
    q1 = next(p for p in item.parts if p.part_id == "Q1")
    object.__setattr__(q1, "expected", "no" if q1.expected == "yes" else "yes")
    assert verify_dsep_drill(item) is False
