"""Dual-verification admission gate for causal items.

An item ships only if its stored answer key agrees with both an independent
closed-form solve and a seeded simulation of the same SCM. Catches miskeyed
items (the v2 failure mode) before they ever reach a model.
"""

from __future__ import annotations

from benchpress.core.types import Item
from benchpress.modules.causal import scm


def verify_item(item: Item) -> bool:
    if item.bundle_id in ("B02", "B03", "B05"):
        from benchpress.modules.causal.transfer import verify_transfer
        return verify_transfer(item)
    if item.bundle_id == "B04":
        from benchpress.modules.causal.simpson import verify_simpson
        return verify_simpson(item)
    if item.bundle_id == "B06":
        from benchpress.modules.causal.base_rate import verify_base_rate
        return verify_base_rate(item)
    if item.bundle_id == "B07":
        from benchpress.modules.causal.iv import verify_iv
        return verify_iv(item)
    if item.bundle_id == "B08":
        from benchpress.modules.causal.rates import verify_rates
        return verify_rates(item)
    if item.bundle_id == "B09":
        from benchpress.modules.causal.frontdoor import verify_frontdoor
        return verify_frontdoor(item)
    if item.bundle_id == "B10":
        from benchpress.modules.causal.selection import verify_selection
        return verify_selection(item)
    if item.bundle_id == "B11":
        from benchpress.modules.causal.effect_mod import verify_effect_mod
        return verify_effect_mod(item)
    if item.bundle_id == "B12":
        from benchpress.modules.causal.dsep_drill import verify_dsep_drill
        return verify_dsep_drill(item)
    return _verify_numeric(item)


def _verify_numeric(item: Item) -> bool:
    gp = item.gen_params
    closed = scm.partial_regression_coef(gp["r_ty"], gp["r_tz"], gp["r_zy"])
    simulated = scm.simulate_partial_coef(
        gp["r_ty"], gp["r_tz"], gp["r_zy"], n=20000, seed=gp.get("sim_seed", 1)
    )
    if abs(simulated - closed) > 0.03:
        return False

    parts = {p.part_id: p for p in item.parts}
    if abs(float(parts["ESTIMATE"].expected) - closed) > 0.02:
        return False
    if {str(x) for x in parts["ADJUSTMENT_SET"].expected} != {gp["confounder"]}:
        return False
    if str(parts["IDENTIFIABLE"].expected).lower() != "yes":
        return False
    return True
