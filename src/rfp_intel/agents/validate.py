"""Check that a field value is cited and supported by its chunk."""

from __future__ import annotations

import re

from rfp_intel.agents.fields import DATE_FIELDS

_DATE = re.compile(
    r"(\d{4}-\d{2}-\d{2})|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|"
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2})",
    re.IGNORECASE,
)
_SENTINELS = {"none", "not required", "not applicable", "n/a", "no", "not stated"}


def judge(field_name: str, value: str | None, chunk_text: str | None) -> str:
    """Return ``pass``, ``fail``, or ``not_found``."""
    if value is None or not str(value).strip():
        return "not_found"
    if not chunk_text or not chunk_text.strip():
        return "fail"
    normalized_value = _normalize(value)
    if normalized_value in _SENTINELS:
        return "pass"
    if field_name in DATE_FIELDS and field_name == "Due Date" and not _DATE.search(value):
        return "fail"
    if len(normalized_value) <= 80:
        if normalized_value not in _normalize(chunk_text):
            return "fail"
        return "pass"
    if _shared_tokens(value, chunk_text) < 3:
        return "fail"
    return "pass"


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _shared_tokens(value: str, chunk_text: str) -> int:
    chunk_tokens = set(_normalize(chunk_text).split())
    significant = [token for token in _normalize(value).split() if len(token) >= 4]
    return sum(1 for token in significant if token in chunk_tokens)
