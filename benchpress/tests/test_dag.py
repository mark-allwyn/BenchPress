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


def _iv_graph():
    # Z is an instrument; U is a latent confounder of X and Y.
    return nx.DiGraph([("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")])


def test_valid_instrument_recognized():
    assert dag.is_instrument(_iv_graph(), "Z", "X", "Y") is True


def test_confounder_is_not_an_instrument():
    # U affects Y directly (violates exclusion).
    assert dag.is_instrument(_iv_graph(), "U", "X", "Y") is False


def test_outcome_is_not_an_instrument():
    assert dag.is_instrument(_iv_graph(), "Y", "X", "Y") is False


def test_not_identifiable_by_adjustment_when_confounder_latent():
    g = _iv_graph()
    assert dag.identifiable_by_adjustment(g, "X", "Y", observed=["Z", "X", "Y"]) is False


def test_identifiable_by_adjustment_when_confounder_observed():
    g = nx.DiGraph([("C", "X"), ("C", "Y"), ("X", "Y")])
    assert dag.identifiable_by_adjustment(g, "X", "Y", observed=["C", "X", "Y"]) is True


def _front_door_graph():
    # X -> M -> Y with latent U confounding X and Y; M is the front-door set.
    return nx.DiGraph([("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")])


def test_front_door_set_recognized():
    assert dag.is_front_door(_front_door_graph(), {"M"}, "X", "Y") is True


def test_front_door_fails_with_direct_x_to_y_edge():
    g = _front_door_graph()
    g.add_edge("X", "Y")  # M no longer intercepts all directed paths
    assert dag.is_front_door(g, {"M"}, "X", "Y") is False


def test_outcome_is_not_a_front_door_set():
    assert dag.is_front_door(_front_door_graph(), {"Y"}, "X", "Y") is False
