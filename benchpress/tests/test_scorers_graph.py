import benchpress.scorers  # noqa: F401
from benchpress.core import registry
from benchpress.core.types import Part


def _score(part_type, expected, raw, params=None):
    scorer = registry.get_part_scorer(part_type)
    gold = Part(part_id="P", part_type=part_type, expected=expected, params=params or {})
    return scorer(gold, raw)


# ---- edge_list ---------------------------------------------------------------

def test_edge_list_exact_arrow():
    r = _score("edge_list", {("A", "B"), ("B", "C")}, "A->B, B->C")
    assert r.correct


def test_edge_list_unicode_arrow_and_order_insensitive():
    assert _score("edge_list", {("A", "B"), ("B", "C")}, "B→C, A→B").correct


def test_edge_list_pair_form():
    assert _score("edge_list", {("A", "B")}, "(A, B)").correct


def test_edge_list_wrong_edge():
    assert not _score("edge_list", {("A", "B")}, "A->C").correct


def test_edge_list_direction_matters():
    assert not _score("edge_list", {("A", "B")}, "B->A").correct


def test_edge_list_unparseable():
    r = _score("edge_list", {("A", "B")}, None)
    assert not r.correct and r.parsed is None


# ---- sequence_match ----------------------------------------------------------

def test_sequence_exact_comma():
    assert _score("sequence_match", ["A", "B", "C"], "A, B, C").correct


def test_sequence_arrow():
    assert _score("sequence_match", ["A", "B", "C"], "A -> B -> C").correct


def test_sequence_wrong_order():
    assert not _score("sequence_match", ["A", "B", "C"], "B, A, C").correct


def test_sequence_unparseable():
    r = _score("sequence_match", ["A", "B"], None)
    assert not r.correct and r.parsed is None
