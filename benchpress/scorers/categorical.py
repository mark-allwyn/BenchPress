"""Categorical / verdict scorer (e.g. identifiable: yes/no)."""

from __future__ import annotations

import re

from benchpress.core.registry import register_part_scorer
from benchpress.core.types import Part, PartResult

# Synonym classes -> canonical token. Extend per item via params["synonyms"].
_DEFAULT_SYNONYMS = {
    "yes": "yes", "true": "yes", "identifiable": "yes",
    "no": "no", "false": "no", "not identifiable": "no", "unidentifiable": "no",
}


def _normalize(text: str, synonyms: dict[str, str]) -> str:
    token = re.sub(r"[^a-z0-9 ]+", "", text.strip().lower()).strip()
    return synonyms.get(token, token)


@register_part_scorer("categorical")
def categorical(gold: Part, raw: str | None) -> PartResult:
    synonyms = {**_DEFAULT_SYNONYMS, **gold.params.get("synonyms", {})}
    expected = _normalize(str(gold.expected), synonyms)
    if raw is None:
        return PartResult(gold.part_id, "categorical", False, None, expected, "unparseable")
    parsed = _normalize(raw, synonyms)
    correct = parsed == expected
    note = "ok" if correct else "mismatch"
    return PartResult(gold.part_id, "categorical", correct, parsed, expected, note)
