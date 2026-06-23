import networkx as nx

from benchpress.modules.causal import dag


def _confounded():
    # C1, C2 confound X->Y; M is a mediator (descendant of X).
    return nx.DiGraph([
        ("C1", "X"), ("C1", "Y"),
        ("C2", "X"), ("C2", "Y"),
        ("X", "Y"), ("X", "M"), ("M", "Y"),
    ])


def test_minimal_backdoor_is_the_confounders():
    assert dag.minimal_backdoor_set(_confounded(), "X", "Y") == frozenset({"C1", "C2"})


def test_satisfies_backdoor_true_for_confounders():
    assert dag.satisfies_backdoor(_confounded(), "X", "Y", {"C1", "C2"})


def test_empty_set_does_not_satisfy_backdoor_when_confounded():
    assert not dag.satisfies_backdoor(_confounded(), "X", "Y", set())


def test_mediator_is_not_a_valid_adjustment():
    # Adjusting a descendant of X violates the backdoor criterion.
    assert not dag.satisfies_backdoor(_confounded(), "X", "Y", {"M"})


def test_superset_satisfies_but_is_not_minimal():
    g = nx.DiGraph([("U", "C1"), ("C1", "X"), ("C1", "Y"), ("X", "Y")])
    assert dag.satisfies_backdoor(g, "X", "Y", {"C1"})
    assert dag.is_minimal_backdoor(g, "X", "Y", {"C1"})
    assert dag.satisfies_backdoor(g, "X", "Y", {"C1", "U"})
    assert not dag.is_minimal_backdoor(g, "X", "Y", {"C1", "U"})
