from benchpress.core.status import classify_status


def test_ok_when_normal_stop_and_extraction_succeeds():
    assert classify_status("end_turn", extraction_ok=True) == "ok"


def test_refusal_from_anthropic_reason():
    assert classify_status("refusal", extraction_ok=False) == "refusal"


def test_refusal_from_openai_content_filter():
    assert classify_status("content_filter", extraction_ok=False) == "refusal"


def test_refusal_from_google_safety():
    assert classify_status("SAFETY", extraction_ok=False) == "refusal"


def test_truncated_from_max_tokens():
    assert classify_status("max_tokens", extraction_ok=False) == "truncated"


def test_truncated_from_openai_length():
    assert classify_status("length", extraction_ok=False) == "truncated"


def test_invalid_answer_when_stop_normal_but_no_answer_extracted():
    assert classify_status("end_turn", extraction_ok=False) == "invalid_answer"


def test_api_error_takes_precedence():
    assert classify_status("end_turn", extraction_ok=True, error="429 rate limit") == "api_error"


def test_truncated_takes_precedence_over_invalid_answer():
    # A truncated response also fails extraction, but the cause is truncation.
    assert classify_status("max_tokens", extraction_ok=False) == "truncated"


def test_refusal_takes_precedence_over_invalid_answer():
    assert classify_status("refusal", extraction_ok=False) == "refusal"


def test_none_stop_reason_with_error_is_api_error():
    assert classify_status(None, extraction_ok=False, error="connection reset") == "api_error"
