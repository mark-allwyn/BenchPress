from benchpress import stats
from benchpress.core.types import ItemResult, PartResult


def _result(item_id, status, parts_ok: dict):
    parts = [PartResult(pid, "t", ok, None, None, "") for pid, ok in parts_ok.items()]
    item_correct = status == "ok" and all(parts_ok.values())
    return ItemResult(item_id, status, item_correct, parts)


def _synthetic():
    rs = []
    for i in range(3):  # fully correct
        rs.append(_result(f"ok{i}", "ok", {"A": True, "B": True, "C": True}))
    for i in range(2):  # ok but C wrong -> item wrong
        rs.append(_result(f"partial{i}", "ok", {"A": True, "B": True, "C": False}))
    rs.append(_result("ref", "refusal", {"A": False, "B": False, "C": False}))
    rs.append(_result("inv", "invalid_answer", {"A": False, "B": False, "C": False}))
    return rs


def test_accuracy_counts():
    acc = stats.accuracy(_synthetic())
    assert acc["attempted"] == 7
    assert acc["correct"] == 3
    assert acc["refusals"] == 1
    assert acc["invalids"] == 1
    assert acc["accuracy"] == 3 / 7


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    rs = _synthetic()
    point = stats.accuracy(rs)["accuracy"]
    lo, hi = stats.bootstrap_ci(rs, seed=0)
    assert lo <= point <= hi
    assert (lo, hi) == stats.bootstrap_ci(rs, seed=0)  # deterministic


def test_bootstrap_ci_empty_is_zero():
    assert stats.bootstrap_ci([]) == (0.0, 0.0)


def test_part_marginals_each_at_least_conjunctive_accuracy():
    rs = _synthetic()
    conjunctive = stats.accuracy(rs)["accuracy"]
    marg = stats.part_marginals(rs)
    assert marg["A"]["accuracy"] == 5 / 7
    assert marg["C"]["accuracy"] == 3 / 7
    for pid, m in marg.items():
        assert m["accuracy"] >= conjunctive


def test_report_is_deterministic_and_has_keys():
    rs = _synthetic()
    r1 = stats.report(rs)
    r2 = stats.report(rs)
    assert r1 == r2
    assert r1["accuracy"] == stats.accuracy(rs)["accuracy"]  # single source
    assert "ci95" in r1 and "part_marginals" in r1


def test_refusal_invalid_reconcile_with_statuses():
    rs = _synthetic()
    acc = stats.accuracy(rs)
    assert acc["refusals"] == sum(1 for r in rs if r.status == "refusal")
    assert acc["invalids"] == sum(1 for r in rs if r.status == "invalid_answer")
