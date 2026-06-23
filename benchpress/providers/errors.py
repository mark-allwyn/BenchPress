"""Strip API keys and tokens from error strings before they are stored."""

from __future__ import annotations

import re


def sanitize_error(error_msg: str) -> str:
    s = error_msg
    s = re.sub(r'(key=)[^&\s\'"]+', r"\1[REDACTED]", s)
    s = re.sub(r'(Bearer\s+)[^\s\'"]+', r"\1[REDACTED]", s)
    s = re.sub(r'(x-api-key[:\s]+)[^\s\'"]+', r"\1[REDACTED]", s, flags=re.IGNORECASE)
    s = re.sub(r"sk-ant-api\S+", "[REDACTED]", s)
    s = re.sub(r"sk-proj-\S+", "[REDACTED]", s)
    s = re.sub(r"sk-ant-\S+", "[REDACTED]", s)
    s = re.sub(r"AIzaSy\S+", "[REDACTED]", s)
    s = re.sub(r"xai-\S+", "[REDACTED]", s)
    s = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", s)
    s = re.sub(r"AKIA[A-Z0-9]+", "[REDACTED]", s)
    return s
