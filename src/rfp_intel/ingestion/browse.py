"""Read the rows each manual pipeline step writes, for the UI."""

from __future__ import annotations

from datetime import datetime

from rfp_intel.config import get_settings
from rfp_intel.db.session import session_scope
from rfp_intel.ingestion.discover import list_bid_folders

_STEPS = ("files", "sections", "tables", "chunks")


def pipeline_catalog() -> list[dict]:
    """Bid folders on disk, with row counts from the tables each step fills."""
    folders = list_bid_folders(get_settings().data_dir)
    counts: dict[str, dict] = {}
    with session_scope() as conn:
        rows = conn.execute(
            """
            SELECT b.folder_name,
                   b.status,
                   count(DISTINCT f.id) AS files,
                   count(DISTINCT f.id) FILTER (WHERE f.parse_status LIKE 'parsed%') AS parsed_files,
                   count(DISTINCT f.id) FILTER (WHERE f.parse_status = 'failed') AS failed_files,
                   count(DISTINCT s.id) AS sections,
                   count(DISTINCT t.id) AS tables,
                   count(DISTINCT c.id) AS chunks,
                   count(DISTINCT c.id) FILTER (WHERE c.embedded_at IS NOT NULL) AS embedded
            FROM bids b
            LEFT JOIN source_files f ON f.bid_id = b.id
            LEFT JOIN sections s ON s.document_id = f.id
            LEFT JOIN tables t ON t.section_id = s.id
            LEFT JOIN chunks c ON c.bid_id = b.id
            GROUP BY b.id
            """
        ).fetchall()
        counts = {row["folder_name"]: dict(row) for row in rows}
    catalog = []
    for folder in folders:
        stored = counts.get(folder.name, {})
        parsed_files = int(stored.get("parsed_files") or 0)
        chunks = int(stored.get("chunks") or 0)
        embedded = int(stored.get("embedded") or 0)
        catalog.append(
            {
                "folder_name": folder.name,
                "path": str(folder),
                "status": _display_status(
                    stored.get("status"),
                    parsed_files,
                    chunks,
                    embedded,
                    failed_files=int(stored.get("failed_files") or 0),
                ),
                "files": int(stored.get("files") or 0),
                "parsed_files": parsed_files,
                "failed_files": int(stored.get("failed_files") or 0),
                "sections": int(stored.get("sections") or 0),
                "tables": int(stored.get("tables") or 0),
                "chunks": chunks,
                "embedded": embedded,
                "can_chunk": parsed_files > 0,
                "can_embed": chunks > 0 and embedded < chunks,
            }
        )
    return catalog


def step_rows(bid_folder: str, step: str, limit: int = 200, file: str | None = None) -> list[dict]:
    if step not in _STEPS:
        raise ValueError(f"unknown step: {step}")
    query = {
        "files": """
            SELECT f.relative_path, f.doc_type, f.addendum_number, f.parse_status,
                   f.error, f.parsed_at, left(f.sha256, 12) AS sha256
            FROM source_files f
            JOIN bids b ON b.id = f.bid_id
            WHERE b.folder_name = %(bid)s
              AND (%(file)s = '' OR f.relative_path = %(file)s)
            ORDER BY f.relative_path
            LIMIT %(limit)s
        """,
        "sections": """
            SELECT s.id::text AS id, f.relative_path AS file, s.ordinal, array_to_string(s.heading_path, ' > ') AS heading,
                   s.level, s.page_start, s.page_end,
                   left(s.body_text, 500) AS body_text,
                   length(s.body_text) AS body_chars
            FROM sections s
            JOIN source_files f ON f.id = s.document_id
            JOIN bids b ON b.id = f.bid_id
            WHERE b.folder_name = %(bid)s
              AND (%(file)s = '' OR f.relative_path = %(file)s)
            ORDER BY f.relative_path, s.ordinal
            LIMIT %(limit)s
        """,
        "tables": """
            SELECT t.id::text AS id, f.relative_path AS file, s.ordinal AS section_ordinal,
                   array_to_string(s.heading_path, ' > ') AS heading,
                   t.page_number, t.caption, left(t.markdown, 500) AS markdown
            FROM tables t
            JOIN sections s ON s.id = t.section_id
            JOIN source_files f ON f.id = s.document_id
            JOIN bids b ON b.id = f.bid_id
            WHERE b.folder_name = %(bid)s
              AND (%(file)s = '' OR f.relative_path = %(file)s)
            ORDER BY f.relative_path, s.ordinal, t.page_number
            LIMIT %(limit)s
        """,
        "chunks": """
            SELECT c.id::text AS id, f.relative_path AS file, c.ordinal, c.page_number, c.token_count, c.doc_type,
                   c.addendum_number, array_to_string(c.heading_path, ' > ') AS heading,
                   left(c.text, 500) AS text,
                   (c.embedded_at IS NOT NULL) AS embedded,
                   c.qdrant_point_id
            FROM chunks c
            JOIN source_files f ON f.id = c.source_file_id
            WHERE c.bid_folder = %(bid)s
              AND (%(file)s = '' OR f.relative_path = %(file)s)
            ORDER BY f.relative_path, c.ordinal
            LIMIT %(limit)s
        """,
    }[step]
    with session_scope() as conn:
        rows = conn.execute(
            query,
            {"bid": bid_folder, "limit": limit, "file": file or ""},
        ).fetchall()
    return [_jsonable(dict(row)) for row in rows]


def step_row(bid_folder: str, step: str, row_id: str) -> dict | None:
    """One section, table, or chunk with its text left intact."""
    query = {
        "sections": """
            SELECT s.id::text AS id, f.relative_path AS file, s.ordinal,
                   array_to_string(s.heading_path, ' > ') AS heading,
                   s.level, s.page_start, s.page_end, s.body_text
            FROM sections s
            JOIN source_files f ON f.id = s.document_id
            JOIN bids b ON b.id = f.bid_id
            WHERE b.folder_name = %(bid)s AND s.id = %(row_id)s::uuid
        """,
        "tables": """
            SELECT t.id::text AS id, f.relative_path AS file, s.ordinal AS section_ordinal,
                   array_to_string(s.heading_path, ' > ') AS heading,
                   t.page_number, t.caption, t.markdown, t.rows_json
            FROM tables t
            JOIN sections s ON s.id = t.section_id
            JOIN source_files f ON f.id = s.document_id
            JOIN bids b ON b.id = f.bid_id
            WHERE b.folder_name = %(bid)s AND t.id = %(row_id)s::uuid
        """,
        "chunks": """
            SELECT c.id::text AS id, f.relative_path AS file, c.ordinal, c.page_number,
                   c.token_count, c.doc_type, c.addendum_number,
                   array_to_string(c.heading_path, ' > ') AS heading,
                   c.text, (c.embedded_at IS NOT NULL) AS embedded, c.qdrant_point_id
            FROM chunks c
            JOIN source_files f ON f.id = c.source_file_id
            WHERE c.bid_folder = %(bid)s AND c.id = %(row_id)s::uuid
        """,
    }.get(step)
    if query is None:
        raise ValueError(f"unknown step: {step}")
    with session_scope() as conn:
        row = conn.execute(query, {"bid": bid_folder, "row_id": row_id}).fetchone()
    return _jsonable(dict(row)) if row else None


def _display_status(
    stored: str | None,
    parsed_files: int,
    chunks: int,
    embedded: int,
    failed_files: int = 0,
) -> str:
    if stored is None:
        return "not_started"
    if stored in {"parsing", "chunking", "embedding"}:
        return stored
    if chunks and embedded == chunks:
        return "ready"
    if chunks:
        return "chunked"
    if parsed_files:
        return "parsed"
    if failed_files:
        return "failed"
    return stored or "not_started"


def _jsonable(row: dict) -> dict:
    item = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
        else:
            item[key] = value
    return item
