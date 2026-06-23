from benchpress import stats
from benchpress.core.types import ItemResult


def _r(item_id, correct):
    return ItemResult(item_id, "ok", correct, [])


# ---- Pareto ------------------------------------------------------------------

def test_pareto_keeps_nondominated_points():
    points = [
        {"model": "A", "accuracy": 0.60, "cost": 10},
        {"model": "B", "accuracy": 0.50, "cost": 5},
        {"model": "C", "accuracy": 0.55, "cost": 20},  # dominated by A
    ]
    frontier = {p["model"] for p in stats.pareto_frontier(points, minimize="cost")}
    assert frontier == {"A", "B"}


# ---- saturation --------------------------------------------------------------

def test_saturation_true_when_top_cis_overlap():
    entries = [("A", 0.80, (0.75, 0.85)), ("B", 0.78, (0.73, 0.83)), ("C", 0.60, (0.55, 0.65))]
    assert stats.saturation(entries, top_n=2) is True


def test_saturation_false_with_clear_leader():
    entries = [("A", 0.90, (0.87, 0.93)), ("B", 0.70, (0.65, 0.75))]
    assert stats.saturation(entries, top_n=2) is False


# ---- item stats --------------------------------------------------------------

def _model_results():
    return {
        "A": [_r("easy", True), _r("disc", True), _r("hard", True)],
        "B": [_r("easy", True), _r("disc", True), _r("hard", False)],
        "C": [_r("easy", True), _r("disc", False), _r("hard", False)],
    }


def test_item_p_values():
    s = stats.item_stats(_model_results())
    assert s["easy"]["p_value"] == 1.0
    assert s["disc"]["p_value"] == 2 / 3


def test_dead_item_has_zero_discrimination():
    s = stats.item_stats(_model_results())
    assert s["easy"]["discrimination"] == 0.0  # everyone correct -> no variance


def test_discriminating_item_has_positive_discrimination():
    s = stats.item_stats(_model_results())
    assert s["disc"]["discrimination"] > 0
    assert s["hard"]["discrimination"] > 0


# ---- efficiency --------------------------------------------------------------

def test_model_efficiency_from_results_data():
    data = {
        "model_name": "m",
        "runs": {
            "i1": [{"content": "x", "latency_s": 1.0, "thinking_tokens": 100,
                    "scored": {"status": "ok", "item_correct": True, "parts": []}}],
            "i2": [{"content": "x", "latency_s": 3.0, "thinking_tokens": 300,
                    "scored": {"status": "ok", "item_correct": False, "parts": []}}],
        },
    }
    e = stats.model_efficiency(data)
    assert e["model"] == "m"
    assert e["attempted"] == 2
    assert e["accuracy"] == 0.5
    assert e["median_latency"] == 2.0
    assert e["median_thinking"] == 200
