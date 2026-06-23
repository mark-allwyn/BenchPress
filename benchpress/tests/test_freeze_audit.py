import pytest

import benchpress.modules.causal  # noqa: F401
from benchpress import stats
from benchpress.core import registry
from benchpress.manifest import build_manifest, load_manifest, write_manifest


def test_manifest_regeneration_reproduces_item_ids(tmp_path):
    items, meta = registry.get_module("causal")(seed=42)
    manifest = build_manifest("causal", seed=42, version=meta.version,
                              item_ids=[i.item_id for i in items])
    path = tmp_path / "bp-causal-v1.json"
    write_manifest(path, manifest)

    loaded = load_manifest(path)
    regen, _ = registry.get_module("causal")(loaded["seed"])
    assert sorted(i.item_id for i in regen) == loaded["item_ids"]
    assert loaded["n_items"] == len(items)


def test_audit_gap_flags_large_public_vs_fresh_drop():
    canonical = {"A": 0.60, "B": 0.55}
    fresh = {"A": 0.58, "B": 0.30}  # B drops a lot -> contamination signal
    gaps = stats.audit_gap(canonical, fresh, threshold=0.1)
    assert gaps["A"]["flagged"] is False
    assert gaps["B"]["flagged"] is True
    assert gaps["B"]["gap"] == pytest.approx(0.25)


def test_review_queue_surfaces_all_wrong_and_dead():
    istats = {
        "i_dead": {"p_value": 1.0, "discrimination": 0.0, "n_models": 3},
        "i_allwrong": {"p_value": 0.0, "discrimination": 0.0, "n_models": 3},
        "i_ok": {"p_value": 0.5, "discrimination": 0.4, "n_models": 3},
    }
    q = stats.review_queue(istats)
    assert q["dead"] == ["i_dead"]
    assert q["all_wrong"] == ["i_allwrong"]
