"""Locate bid folders and hash the files inside them."""

from __future__ import annotations

import hashlib
from pathlib import Path

SUPPORTED = {".pdf", ".html", ".htm"}


def list_bid_folders(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    folders = []
    for child in sorted(data_dir.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and not child.name.startswith(".") and _has_documents(child):
            folders.append(child)
    return folders


def _has_documents(folder: Path) -> bool:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            return True
    return False


def hash_folder(folder: Path) -> dict[str, str]:
    """Map a path relative to ``folder`` to the sha256 of that file."""
    hashes: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(folder).as_posix()
        hashes[relative] = sha256_file(path)
    return hashes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
