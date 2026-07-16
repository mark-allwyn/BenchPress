"""per_task_summary rolls scored results into per-task + overall metrics, using
the status taxonomy for validity columns."""

from benchpress.core.types import Item, ItemResult, Part, PartResult
from benchpress.runner.summary import format_console, per_task_summary


def _item(iid, bundle, nparts=2):
    parts = [Part(f"ROW{i+1}", "categorical", "0") for i in range(nparts)]
    return Item(iid, "simulate", bundle, "sim", "hard", {}, "prompt", parts, [])


def _result(iid, status, item_correct, part_flags):
    parts = [PartResult(f"ROW{i+1}", "categorical", f, None, "0") for i, f in enumerate(part_flags)]
    return ItemResult(iid, status, item_correct, parts)


def test_two_tasks_exact_and_per_row():
    items = [_item("a1", "LIFE"), _item("a2", "LIFE"), _item("b1", "ECA")]
    results = [
        _result("a1", "ok", True, [True, True]),     # LIFE item fully correct
        _result("a2", "ok", False, [True, False]),    # LIFE item 1/2 rows
        _result("b1", "ok", True, [True, True]),      # ECA item correct
    ]
    s = per_task_summary(items, results)
    assert s["tasks"]["LIFE"]["n"] == 2
    assert s["tasks"]["LIFE"]["exact_pct"] == 50.0       # 1 of 2 items exact
    assert s["tasks"]["LIFE"]["per_row_pct"] == 75.0     # 3 of 4 rows
    assert s["tasks"]["ECA"]["exact_pct"] == 100.0
    assert s["overall"]["n"] == 3
    assert s["overall"]["exact_pct"] == round(2 / 3 * 100, 1)


def test_truncation_and_error_columns():
    items = [_item("a1", "LIFE"), _item("a2", "LIFE"), _item("a3", "LIFE")]
    results = [
        _result("a1", "truncated", False, [True, False]),
        _result("a2", "api_error", False, [False, False]),
        _result("a3", "ok", True, [True, True]),
    ]
    s = per_task_summary(items, results)
    assert s["tasks"]["LIFE"]["truncated"] == 1
    assert s["tasks"]["LIFE"]["errors"] == 1
    assert s["overall"]["truncated"] == 1 and s["overall"]["errors"] == 1


def test_task_order_follows_items_not_results():
    items = [_item("z", "ZEBRA"), _item("a", "APPLE")]
    results = [_result("a", "ok", True, [True, True]), _result("z", "ok", True, [True, True])]
    s = per_task_summary(items, results)
    assert list(s["tasks"].keys()) == ["ZEBRA", "APPLE"]


def test_missing_results_do_not_crash():
    items = [_item("a1", "LIFE")]
    s = per_task_summary(items, [])
    assert s["tasks"]["LIFE"]["n"] == 0
    assert s["tasks"]["LIFE"]["exact_pct"] == 0.0
    assert s["overall"]["n"] == 0


def test_format_console_contains_tasks_and_overall():
    items = [_item("a1", "LIFE")]
    results = [_result("a1", "ok", True, [True, True])]
    text = format_console(per_task_summary(items, results), title="TEST")
    assert "TEST" in text and "LIFE" in text and "OVERALL" in text
