"""Banned-vocabulary lint: keep generated content refusal-neutral.

Rejects health/bio and security topics that trip provider safety filters (the
cause of Fable's causal refusals). "treatment"/"effect" are deliberately NOT
banned - they are core causal vocabulary.
"""

from __future__ import annotations

import re

_BANNED = {
    # health / bio
    "patient", "disease", "cancer", "tumor", "tumour", "vaccine", "infection",
    "virus", "pathogen", "mortality", "clinical", "symptom", "diagnosis",
    "gene", "dna", "biological", "epidemic", "outbreak", "drug", "antibiotic",
    # security
    "weapon", "malware", "exploit", "bomb", "terror", "terrorist",
    "surveillance", "hacking", "ransomware",
}


def lint_text(text: str) -> list[str]:
    low = text.lower()
    return sorted(w for w in _BANNED if re.search(rf"\b{re.escape(w)}\b", low))
