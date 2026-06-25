import benchpress.modules.frontier  # noqa: F401  (registers "frontier")
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.core.tagged_text import parse_tagged_fields
from benchpress.modules.frontier import graphs as g
import networkx as nx


def _gen(seed=1):
    return registry.get_module("frontier")(seed)


def test_emits_all_five_bundles():
    items, meta = _gen()
    assert set(meta.bundles) == {"DSEP", "LINEXT", "VSTRUCT", "OPENPATH", "MINSEP_COUNT"}
    by = {b: [i for i in items if i.bundle_id == b] for b in meta.bundles}
    for b, lst in by.items():
        assert len(lst) == 8, f"{b} has {len(lst)}"


def test_count_items_have_one_numeric_part():
    items, _ = _gen()
    for it in (i for i in items if i.variant == "count"):
        assert len(it.parts) == 1
        assert it.parts[0].part_id == "ANSWER"
        assert it.parts[0].part_type == "numeric_tolerance"


def test_dsep_battery_is_conjunctive():
    items, _ = _gen()
    dsep = [i for i in items if i.bundle_id == "DSEP"]
    from benchpress.modules.frontier import CONFIG
    for it in dsep:
        assert len(it.parts) == CONFIG["DSEP"]["queries"]
        assert all(p.part_type == "categorical" for p in it.parts)


def test_gold_recomputes_from_stored_edges():
    # Re-derive each count item's gold from its stored graph; must match the key.
    items, _ = _gen()
    for it in (i for i in items if i.variant == "count"):
        edges = [tuple(e) for e in it.gen_params["edges"]]
        nodes = it.gen_params["nodes"]
        G = nx.DiGraph(edges); G.add_nodes_from(nodes)
        gold = it.parts[0].expected
        if it.bundle_id == "MEC":
            assert g.mec_size(edges, nodes) == gold
        elif it.bundle_id == "LINEXT":
            assert g.linear_extension_count(G) == gold
        elif it.bundle_id == "VSTRUCT":
            assert g.vstructure_count(edges, nodes) == gold
        elif it.bundle_id == "COMPELLED":
            assert g.compelled_count(edges, nodes) == gold


def test_prompt_contains_answer_format():
    items, _ = _gen()
    for it in items:
        assert "ANSWER" in it.prompt


def test_generation_deterministic():
    a, _ = _gen(3)
    b, _ = _gen(3)
    assert [(i.item_id, i.prompt) for i in a] == [(i.item_id, i.prompt) for i in b]


def test_scoring_a_perfect_response(tmp_path):
    # Build the gold answer string for one count item and confirm it scores correct.
    from benchpress.runner.score import score_response
    items, _ = _gen()
    it = next(i for i in items if i.bundle_id == "MEC")
    content = f"ANSWER: {int(it.parts[0].expected)}"
    result = score_response(it, content, "end_turn")
    assert result.item_correct is True
