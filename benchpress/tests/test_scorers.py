import benchpress.scorers  # noqa: F401  (registers scorers)
from benchpress.core import registry
from benchpress.core.types import Part


def _score(part_type, expected, raw, params=None):
    scorer = registry.get_part_scorer(part_type)
    gold = Part(part_id="P", part_type=part_type, expected=expected, params=params or {})
    return scorer(gold, raw)


# ---- set_match ---------------------------------------------------------------

def test_set_match_exact():
    r = _score("set_match", {"X", "Z"}, "{X, Z}")
    assert r.correct
    assert sorted(r.parsed) == ["X", "Z"]


def test_set_match_order_and_brace_insensitive():
    assert _score("set_match", {"X", "Z"}, "Z, X").correct
    assert _score("set_match", {"X", "Z"}, "[X Z]").correct


def test_set_match_empty_set_synonyms():
    assert _score("set_match", set(), "{}").correct
    assert _score("set_match", set(), "none").correct
    assert _score("set_match", set(), "empty set").correct


def test_set_match_wrong_set():
    r = _score("set_match", {"Z"}, "{X, Z}")
    assert not r.correct


def test_set_match_unparseable_is_wrong_not_error():
    r = _score("set_match", {"Z"}, None)
    assert not r.correct
    assert r.parsed is None
    assert r.note == "unparseable"


# ---- numeric_tolerance -------------------------------------------------------

def test_numeric_within_tolerance():
    assert _score("numeric_tolerance", 0.40, "0.41", {"tol": 0.02}).correct


def test_numeric_outside_tolerance():
    assert not _score("numeric_tolerance", 0.40, "0.50", {"tol": 0.02}).correct


def test_numeric_strips_percent_and_symbols():
    assert _score("numeric_tolerance", 0.42, "≈ 0.42", {"tol": 0.001}).correct
    assert _score("numeric_tolerance", 12.0, "12%", {"tol": 0.001}).correct


def test_numeric_parses_fraction():
    assert _score("numeric_tolerance", 3.5, "7/2", {"tol": 0.001}).correct


def test_numeric_negative():
    assert _score("numeric_tolerance", -0.05, "-0.05", {"tol": 0.001}).correct


def test_numeric_unparseable_is_wrong():
    r = _score("numeric_tolerance", 0.4, "I am not sure", {"tol": 0.02})
    assert not r.correct
    assert r.parsed is None


# ---- categorical -------------------------------------------------------------

def test_categorical_exact():
    assert _score("categorical", "yes", "yes").correct


def test_categorical_case_and_punctuation_insensitive():
    assert _score("categorical", "yes", "Yes.").correct
    assert _score("categorical", "no", "  NO  ").correct


def test_categorical_wrong():
    assert not _score("categorical", "yes", "no").correct


def test_categorical_unparseable_is_wrong():
    r = _score("categorical", "yes", None)
    assert not r.correct
    assert r.parsed is None
