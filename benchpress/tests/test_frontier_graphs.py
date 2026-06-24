import networkx as nx

from benchpress.modules.frontier import graphs as g


def _dg(edges):
    G = nx.DiGraph(edges)
    G.add_nodes_from({n for e in edges for n in e})
    return G


def test_d_separation_collider_rule():
    G = _dg([("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")])
    assert g.d_separated(G, "A", "B", []) is True            # collider C closed
    assert g.d_separated(G, "A", "B", ["C"]) is False         # conditioning C opens it


def test_vstructure_count():
    assert g.vstructure_count([("A", "C"), ("B", "C"), ("C", "D"), ("E", "D")],
                              ["A", "B", "C", "D", "E"]) == 2


def test_mec_size_chain_is_three():
    assert g.mec_size([("A", "B"), ("B", "C")], ["A", "B", "C"]) == 3


def test_mec_size_vstructure_is_one():
    assert g.mec_size([("A", "C"), ("B", "C")], ["A", "B", "C"]) == 1


def test_compelled_count_chain_is_zero():
    # No v-structures -> nothing is forced.
    assert g.compelled_count([("A", "B"), ("B", "C")], ["A", "B", "C"]) == 0


def test_compelled_count_vstructure_is_two():
    assert g.compelled_count([("A", "C"), ("B", "C")], ["A", "B", "C"]) == 2


def test_linear_extension_count():
    assert g.linear_extension_count(_dg([("A", "B"), ("A", "C")])) == 2  # ABC, ACB


def test_min_separator_size():
    G = _dg([("Z1", "X"), ("Z1", "Y"), ("Z2", "X"), ("Z2", "Y")])
    assert g.min_separator_size(G, "X", "Y") == 2  # must block both confounders


def test_minimal_separators_single_confounder():
    G = _dg([("Z", "X"), ("Z", "Y")])
    seps = g.minimal_separators(G, "X", "Y")
    assert ("Z",) in seps and () not in seps


def test_open_path_count_matches_dsep():
    G = _dg([("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")])
    # zero open paths iff d-separated, across several conditioning sets
    for S in ([], ["C"], ["D"], ["C", "D"]):
        assert (g.open_path_count(G, "A", "B", S) == 0) == g.d_separated(G, "A", "B", S)


def test_seeded_dag_is_reproducible():
    a = g.seeded_dag(7, 10, 0.3)
    b = g.seeded_dag(7, 10, 0.3)
    assert a[1] == b[1]
