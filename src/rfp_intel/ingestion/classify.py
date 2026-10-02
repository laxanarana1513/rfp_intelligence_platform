"""Infer document type from the file name. Not tied to a specific bid."""

from __future__ import annotations

import re
from pathlib import Path

_ADDENDUM = re.compile(r"(?:addendum|amendment)\s*#?\s*(\d+)", re.IGNORECASE)
_HTML = {".html", ".htm"}


def classify_document(path: Path) -> tuple[str, int | None]:
    """Return ``(doc_type, addendum_number)``.

    ``doc_type`` is one of ``bid_page``, ``addendum``, ``affidavit``, ``specs``, ``rfp``.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in _HTML:
        return "bid_page", None
    match = _ADDENDUM.search(name)
    if match or "addendum" in name or "amendment" in name:
        number = int(match.group(1)) if match else None
        return "addendum", number
    if "affidavit" in name:
        return "affidavit", None
    if "spec" in name:
        return "specs", None
    return "rfp", None
