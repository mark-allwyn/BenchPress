"""Dual-verification admission gate for causal items.

An item ships only if its stored answer key agrees with both an independent
closed-form solve and a seeded simulation of the same SCM. Catches miskeyed
items (the v2 failure mode) before they ever reach a model.
"""

from __future__ import annotations

from benchpress.core.types import Item
from benchpress.modules.causal import scm


def verify_item(item: Item) -> bool:
    if item.bundle_id in ("B02", "B03"):
        from benchpress.modules.causal.transfer import verify_transfer
        return verify_transfer(item)
    if item.bundle_id == "B04":
        from benchpress.modules.causal.simpson import verify_simpson
        return verify_simpson(item)
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
