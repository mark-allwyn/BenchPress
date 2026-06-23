"""Response status taxonomy.

Every model response is classified into exactly one status from its provider
``stop_reason`` plus whether an answer could be extracted. Refusal and
truncation are recorded explicitly so they are never silently scored as wrong
answers (the legacy harness conflated them, which is why a run of provider
refusals looked like incorrect answers).
"""

from __future__ import annotations

# Normalized (lower-case) stop reasons across providers that mean the model
# declined to answer for safety/policy reasons.
_REFUSAL = {
    "refusal",
    "content_filter",
    "content_filtered",
    "safety",
    "recitation",
    "blocklist",
    "prohibited_content",
}

# Stop reasons that mean the response was cut off before completing.
_TRUNCATED = {
    "max_tokens",
    "length",
    "model_length",
    "max_output_tokens",
}

Status = str  # one of: ok | refusal | invalid_answer | truncated | api_error


def classify_status(
    stop_reason: str | None,
    *,
    extraction_ok: bool,
    error: str | None = None,
) -> Status:
    """Classify a response. Precedence: api_error > refusal > truncated >
    invalid_answer > ok."""
    if error:
        return "api_error"
    reason = (stop_reason or "").strip().lower()
    if reason in _REFUSAL:
        return "refusal"
    if reason in _TRUNCATED:
        return "truncated"
    if not extraction_ok:
        return "invalid_answer"
    return "ok"
