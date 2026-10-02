"""Compare on-disk file hashes with the hashes already stored for a bid."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileDiff:
    added: list[str]
    changed: list[str]
    removed: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.removed)


def diff_hashes(disk: dict[str, str], stored: dict[str, str]) -> FileDiff:
    disk_keys = set(disk)
    stored_keys = set(stored)
    added = sorted(disk_keys - stored_keys)
    removed = sorted(stored_keys - disk_keys)
    changed = sorted(key for key in disk_keys & stored_keys if disk[key] != stored[key])
    return FileDiff(added=added, changed=changed, removed=removed)
