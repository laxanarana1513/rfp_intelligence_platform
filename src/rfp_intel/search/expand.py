"""Deterministic procurement query expansion.

The synonym map is always applied so retrieval evaluation does not depend on an LLM rewrite.
"""

from __future__ import annotations

SYNONYMS: dict[str, list[str]] = {
    "deadline": ["due date", "closing date", "submission deadline", "bids due"],
    "due date": ["closing date", "submission deadline", "deadline"],
    "closing": ["closing date", "due date", "submission deadline"],
    "bond": ["bid bond", "bid security", "surety", "bid guarantee"],
    "warranty": ["extended warranty", "warranty period", "coverage"],
    "affidavit": ["affidavits", "required forms", "certification", "sworn statement"],
    "delivery": ["delivery date", "delivery window", "ship date", "lead time"],
    "installation": ["install", "deployment", "imaging", "professional services"],
    "payment": ["payment terms", "net 30", "invoice", "invoicing"],
    "pre-bid": ["prebid", "pre bid meeting", "pre-proposal conference"],
    "prebid": ["pre-bid meeting", "pre bid conference"],
    "contact": ["procurement contact", "buyer", "email", "phone"],
    "model": ["model number", "model no", "sku"],
    "part": ["part number", "part no", "sku"],
    "specification": ["specifications", "technical requirements", "minimum requirements"],
    "insurance": ["certificate of insurance", "liability"],
    "submission": ["bid submission", "how to submit", "electronic submission", "sealed bid"],
}


def expand_query(query: str) -> str:
    lowered = query.lower()
    extra: list[str] = []
    for key, phrases in SYNONYMS.items():
        if key in lowered:
            extra.extend(phrases)
    if not extra:
        return query
    # Keep the original query first so exact terms stay dominant for full-text search.
    unique = []
    seen = set()
    for phrase in extra:
        if phrase not in seen and phrase not in lowered:
            seen.add(phrase)
            unique.append(phrase)
    if not unique:
        return query
    return f"{query} {' '.join(unique)}"
