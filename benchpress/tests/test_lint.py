import benchpress.modules.causal  # noqa: F401
from benchpress.core import registry
from benchpress.modules.causal.lint import lint_text


def test_lint_flags_health_terms():
    flags = lint_text("Each patient received a vaccine to prevent the disease.")
    assert "patient" in flags
    assert "vaccine" in flags
    assert "disease" in flags


def test_lint_flags_security_terms():
    assert "malware" in lint_text("the malware sample")


def test_lint_clean_text_has_no_flags():
    assert lint_text("Fertilizer use raised crop yield on the farm.") == []


def test_all_generated_prompts_are_refusal_neutral():
    items, _ = registry.get_module("causal")(seed=3)
    for item in items:
        assert lint_text(item.prompt) == [], f"{item.item_id} has banned vocab: {lint_text(item.prompt)}"
