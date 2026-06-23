import math

import pytest

from benchpress.core import registry
import benchpress.modules.causal  # noqa: F401  (registers the "causal" module)
from benchpress.modules.causal import scm
from benchpress.modules.causal.verify import verify_item


# ---- SCM math ----------------------------------------------------------------

def test_partial_coef_no_confounding_equals_marginal():
    # With r_tz = 0 there is no backdoor path; partial == marginal.
    assert scm.partial_regression_coef(r_ty=0.5, r_tz=0.0, r_zy=0.0) == pytest.approx(0.5)


def test_partial_coef_removes_backdoor():
    # r_ty inflated by a backdoor through Z; partial recovers the direct effect.
    r_tz, r_zy, b = 0.6, 0.5, 0.3
    r_ty = b * (1 - r_tz**2) + r_tz * r_zy
    assert scm.partial_regression_coef(r_ty, r_tz, r_zy) == pytest.approx(b, abs=1e-9)


def test_simulation_agrees_with_closed_form():
    r_tz, r_zy, b = 0.5, 0.4, 0.35
    r_ty = b * (1 - r_tz**2) + r_tz * r_zy
    closed = scm.partial_regression_coef(r_ty, r_tz, r_zy)
    simulated = scm.simulate_partial_coef(r_ty, r_tz, r_zy, n=40000, seed=1)
    assert simulated == pytest.approx(closed, abs=0.03)


# ---- generator ---------------------------------------------------------------

def _generate(seed=42):
    return registry.get_module("causal")(seed, "hard")


def test_generation_is_deterministic():
    items_a, _ = _generate(42)
    items_b, _ = _generate(42)
    a = [(i.item_id, i.prompt, [(p.part_id, p.expected) for p in i.parts]) for i in items_a]
    b = [(i.item_id, i.prompt, [(p.part_id, p.expected) for p in i.parts]) for i in items_b]
    assert a == b
    assert len(items_a) >= 3


def test_each_item_has_three_conjunctive_parts():
    items, _ = _generate()
    for item in items:
        ptypes = {p.part_id: p.part_type for p in item.parts}
        assert ptypes["ADJUSTMENT_SET"] == "set_match"
        assert ptypes["ESTIMATE"] == "numeric_tolerance"
        assert ptypes["IDENTIFIABLE"] == "categorical"


def test_estimate_gold_matches_closed_form():
    items, _ = _generate()
    for item in items:
        gp = item.gen_params
        expected = scm.partial_regression_coef(gp["r_ty"], gp["r_tz"], gp["r_zy"])
        est = next(p for p in item.parts if p.part_id == "ESTIMATE")
        assert est.expected == pytest.approx(expected, abs=0.005)


def test_every_generated_item_passes_the_dual_verification_gate():
    items, _ = _generate()
    for item in items:
        assert verify_item(item) is True


def test_gate_rejects_a_miskeyed_estimate():
    items, _ = _generate()
    item = items[0]
    bad = next(p for p in item.parts if p.part_id == "ESTIMATE")
    object.__setattr__(bad, "expected", bad.expected + 0.5)  # corrupt the key
    assert verify_item(item) is False


def test_prompt_contains_format_spec_and_worked_example():
    items, _ = _generate()
    prompt = items[0].prompt
    assert "ADJUSTMENT_SET:" in prompt
    assert "ESTIMATE:" in prompt
    assert "IDENTIFIABLE:" in prompt
    assert "example" in prompt.lower()


def test_identifiable_is_yes_and_adjustment_set_is_the_confounder():
    items, _ = _generate()
    item = items[0]
    adj = next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET")
    ident = next(p for p in item.parts if p.part_id == "IDENTIFIABLE")
    assert ident.expected == "yes"
    assert len(adj.expected) == 1  # exactly the single confounder
