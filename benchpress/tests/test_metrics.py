"""Reporting-layer metrics: token accounting rule + efficiency/resilience math."""

from benchpress.metrics import efficiency_per_1k, generated_tokens, resilience_pct


def test_google_adds_thinking_tokens_to_output():
    # Gemini's output_tokens is the visible answer only; thoughts are separate.
    assert generated_tokens("google", 2600, 63000) == 65600


def test_non_google_providers_do_not_add_thinking():
    # Bedrock/OpenAI/Anthropic already fold reasoning into output_tokens.
    assert generated_tokens("bedrock", 40000, None) == 40000
    assert generated_tokens("openai", 40000, 30000) == 40000  # thinking is a subset, not added
    assert generated_tokens("anthropic", 40000, 30000) == 40000


def test_google_with_no_thinking_tokens_falls_back_to_output():
    assert generated_tokens("google", 2600, None) == 2600


def test_missing_output_tokens_excludes_item():
    assert generated_tokens("google", None, 63000) is None
    assert generated_tokens("bedrock", None, None) is None


def test_efficiency_pooled_scaled_per_1k():
    # 315 correct rows over 4.2M generated tokens -> 0.075 rows / 1k.
    assert efficiency_per_1k(315, 4_200_000) == 0.075


def test_efficiency_none_when_no_tokens():
    assert efficiency_per_1k(10, 0) is None
    assert efficiency_per_1k(10, None) is None


def test_resilience_pct():
    assert resilience_pct(100, 10) == 90.0
    assert resilience_pct(100, 0) == 100.0
    assert resilience_pct(0, 0) is None
