import networkx as nx

import benchpress.modules.frontier  # noqa: F401  (registers "frontier")
import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.modules.frontier import CONFIG
from benchpress.modules.frontier import graphs as g

BUNDLES = {"FRONTDOOR", "IV", "SEM"}


def _gen(seed=1):
    return registry.get_module("frontier")(seed)


def test_emits_all_bundles():
    items, meta = _gen()
    assert set(meta.bundles) == BUNDLES
    by = {b: [i for i in items if i.bundle_id == b] for b in meta.bundles}
    for b, lst in by.items():
        assert len(lst) == 8, f"{b} has {len(lst)}"


def test_every_item_is_a_battery_of_the_configured_length():
    items, _ = _gen()
    for it in items:
        assert it.variant == "battery"
        assert len(it.parts) == CONFIG[it.bundle_id]["queries"]


def test_yesno_batteries_have_a_mix():
    # FRONTDOOR/IV are yes/no; an all-yes or all-no battery would be guessable.
    items, _ = _gen()
    for it in items:
        if it.bundle_id not in ("FRONTDOOR", "IV"):
            continue
        yes = sum(1 for p in it.parts if p.expected == "yes")
        assert 0 < yes < len(it.parts), f"{it.item_id} degenerate ({yes}/{len(it.parts)})"


def test_gold_recomputes_from_stored_graph():
    # Independently re-derive every sub-question's gold from the stored graph.
    items, _ = _gen()
    for it in items:
        nodes = it.gen_params["nodes"]
        edges = [tuple(e) for e in it.gen_params["edges"]]
        G = nx.DiGraph(edges)
        G.add_nodes_from(nodes)
        q = it.gen_params["queries"]
        kind, params = q["kind"], q["q"]
        assert len(params) == len(it.parts)
        if kind == "sem":
            w = {(a, b): wt for a, b, wt in it.gen_params["weights"]}
            for (x, y), part in zip(params, it.parts):
                assert abs(round(g.total_effect(G, w, x, y), 2) - part.expected) < 1e-9
        else:
            for p, part in zip(params, it.parts):
                if kind == "frontdoor":
                    x, y, Z = p
                    gold = g.frontdoor_set_valid(G, x, y, Z)
                elif kind == "iv":
                    v, x, y = p
                    gold = g.instrument_valid(G, v, x, y)
                else:
                    raise AssertionError(f"unknown kind {kind}")
                assert ("yes" if gold else "no") == part.expected


def test_prompt_contains_answer_format():
    items, _ = _gen()
    for it in items:
        assert "ANSWER FORMAT" in it.prompt


def test_generation_deterministic():
    a, _ = _gen(3)
    b, _ = _gen(3)
    assert [(i.item_id, i.prompt) for i in a] == [(i.item_id, i.prompt) for i in b]


def test_scoring_a_perfect_response():
    from benchpress.runner.score import score_response
    items, _ = _gen()
    for bundle in ("FRONTDOOR", "SEM"):
        it = next(i for i in items if i.bundle_id == bundle)
        if bundle == "SEM":
            content = "\n".join(f"{p.part_id}: {p.expected:.2f}" for p in it.parts)
        else:
            content = "\n".join(f"{p.part_id}: {p.expected}" for p in it.parts)
        result = score_response(it, content, "end_turn")
        assert result.item_correct is True, f"{bundle} perfect response did not score correct"
