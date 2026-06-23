"""Refusal-neutral cover stories. No health/bio/security topics."""

from __future__ import annotations

import random

SCENARIOS = [
    {"domain": "an agricultural extension program", "Z": "SoilQuality", "T": "FertilizerUse", "Y": "CropYield"},
    {"domain": "a manufacturing line", "Z": "MachineAge", "T": "MaintenanceHours", "Y": "DefectRate"},
    {"domain": "a regional logistics network", "Z": "RouteCongestion", "T": "DispatchFrequency", "Y": "DeliveryTime"},
    {"domain": "an online education platform", "Z": "PriorAptitude", "T": "StudyHours", "Y": "TestScore"},
    {"domain": "a retail marketing campaign", "Z": "StoreSize", "T": "AdSpend", "Y": "Sales"},
]


def pick(rng: random.Random) -> dict:
    return rng.choice(SCENARIOS)
