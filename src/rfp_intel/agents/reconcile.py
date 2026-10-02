"""Pick the latest addendum value for each field and record what changed."""

from __future__ import annotations

from collections import defaultdict


def reconcile_candidates(candidates: list[dict], field_names: list[str]) -> tuple[dict[str, dict | None], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        value = candidate.get("value")
        if value is None or not str(value).strip():
            continue
        grouped[candidate["field_name"]].append(candidate)

    chosen: dict[str, dict | None] = {}
    changes: list[dict] = []
    for field in field_names:
        items = list(grouped.get(field, []))
        if not items:
            chosen[field] = None
            continue
        items.sort(key=lambda item: ((item.get("addendum_number") or 0), float(item.get("confidence") or 0)))
        winner = dict(items[-1])
        winner_addendum = winner.get("addendum_number") or 0
        earlier = [item for item in items if (item.get("addendum_number") or 0) < winner_addendum]
        if earlier and winner_addendum > 0 and earlier[-1].get("value") != winner.get("value"):
            previous = earlier[-1].get("value")
            changes.append(
                {
                    "field": field,
                    "previous_value": previous,
                    "new_value": winner.get("value"),
                    "addendum_number": winner.get("addendum_number"),
                    "file": winner.get("file_name"),
                    "page": winner.get("page"),
                }
            )
            note = f"Updated by addendum {winner.get('addendum_number')} (original: {previous})"
            existing = (winner.get("notes") or "").strip()
            winner["notes"] = f"{existing} {note}".strip()
        chosen[field] = winner
    return chosen, changes
