from dataclasses import asdict

from benchpress.core.types import Item, ItemResult, ModuleMeta, Part, PartResult


def test_part_holds_gold_and_scoring_metadata():
    part = Part(
        part_id="ADJUSTMENT_SET",
        part_type="set_match",
        expected={"X", "Z"},
        params={"tol": 0.02},
        skill_tags=["confounding"],
    )
    assert part.part_id == "ADJUSTMENT_SET"
    assert part.part_type == "set_match"
    assert part.expected == {"X", "Z"}


def test_item_carries_prompt_and_parts():
    part = Part(part_id="ESTIMATE", part_type="numeric_tolerance", expected=0.0)
    item = Item(
        item_id="causal-v1-B01-0007",
        module="causal",
        bundle_id="B01",
        variant="numeric",
        difficulty="hard",
        gen_params={"seed": 7},
        prompt="...prompt text...",
        parts=[part],
    )
    assert item.parts[0] is part
    assert item.module == "causal"


def test_item_result_is_serializable():
    result = ItemResult(
        item_id="causal-v1-B01-0007",
        status="ok",
        item_correct=False,
        parts=[
            PartResult(
                part_id="ESTIMATE",
                part_type="numeric_tolerance",
                correct=False,
                parsed=0.42,
                expected=0.0,
                note="tolerance",
            )
        ],
    )
    d = asdict(result)
    assert d["status"] == "ok"
    assert d["parts"][0]["part_id"] == "ESTIMATE"


def test_module_meta_defaults():
    meta = ModuleMeta(name="causal", version="1")
    assert meta.name == "causal"
    assert meta.variants == []
    assert meta.bundles == []
