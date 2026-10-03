"""Recall@k and MRR for relevance defined by a passage substring."""

from __future__ import annotations


def is_relevant(hit: dict, question: dict) -> bool:
    text = (hit.get("text") or "").lower()
    passage = (question.get("passage") or "").lower()
    if not passage or passage not in text:
        return False
    file_needle = (question.get("file_contains") or "").lower()
    if file_needle and file_needle not in (hit.get("file_name") or "").lower():
        return False
    bid_id = question.get("bid_id")
    if bid_id and hit.get("bid_id") and hit.get("bid_id") != bid_id:
        return False
    return True


def recall_at_k(hits: list[dict], question: dict, k: int = 5) -> float:
    return 1.0 if any(is_relevant(hit, question) for hit in hits[:k]) else 0.0


def reciprocal_rank(hits: list[dict], question: dict) -> float:
    for index, hit in enumerate(hits, start=1):
        if is_relevant(hit, question):
            return 1.0 / index
    return 0.0


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)