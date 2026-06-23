from benchpress.core.tagged_text import parse_tagged_fields


def test_basic_label_value():
    assert parse_tagged_fields("ADJUSTMENT_SET: {X, Z}") == {"ADJUSTMENT_SET": "{X, Z}"}


def test_label_lookup_is_case_insensitive():
    # Lower-case label in the response is normalized to the canonical upper key.
    assert parse_tagged_fields("estimate: 0.42") == {"ESTIMATE": "0.42"}


def test_multiple_fields():
    text = "ADJUSTMENT_SET: {X, Z}\nESTIMATE: 0.42\nIDENTIFIABLE: yes"
    assert parse_tagged_fields(text) == {
        "ADJUSTMENT_SET": "{X, Z}",
        "ESTIMATE": "0.42",
        "IDENTIFIABLE": "yes",
    }


def test_markdown_bold_label_is_tolerated():
    assert parse_tagged_fields("**ESTIMATE:** 0.42") == {"ESTIMATE": "0.42"}


def test_leading_list_marker_is_tolerated():
    assert parse_tagged_fields("- ESTIMATE: 0.42") == {"ESTIMATE": "0.42"}


def test_repeated_label_last_wins():
    text = "ESTIMATE: 0.10\nsome reconsideration\nESTIMATE: 0.42"
    assert parse_tagged_fields(text) == {"ESTIMATE": "0.42"}


def test_prose_lines_are_ignored():
    text = "Let me think about this carefully.\nESTIMATE: 0.42\nThat is my answer."
    assert parse_tagged_fields(text) == {"ESTIMATE": "0.42"}


def test_missing_label_is_absent():
    result = parse_tagged_fields("ESTIMATE: 0.42")
    assert "ADJUSTMENT_SET" not in result


def test_value_keeps_internal_colon():
    # Only the first colon separates label from value.
    assert parse_tagged_fields("NOTE: ratio is 3:1") == {"NOTE": "ratio is 3:1"}


def test_value_strips_wrapping_backticks():
    assert parse_tagged_fields("ESTIMATE: `0.42`") == {"ESTIMATE": "0.42"}


def test_empty_text_returns_empty_dict():
    assert parse_tagged_fields("") == {}
