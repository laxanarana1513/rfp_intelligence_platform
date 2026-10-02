"""Parse, chunk, and embed one bid folder. Each step runs only when called."""

from __future__ import annotations

import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.http import models as qmodels

from rfp_intel.config import get_settings
from rfp_intel.db.models import Bid, Chunk, Section, SourceFile, Table
from rfp_intel.db.session import connect, session_scope
from rfp_intel.db.store import (
    chunk_count,
    delete_bid,
    delete_chunks_for_file,
    delete_sections_for_file,
    delete_source_file,
    get_bid_by_folder,
    get_source_file,
    has_pending_embed,
    insert_bid,
    insert_chunk,
    insert_section,
    insert_source_file,
    insert_table,
    mark_chunk_embedded,
    pending_chunks,
    save_bid,
    save_source_file,
    sections_for_document,
    source_files_for_bid,
)
from rfp_intel.ingestion.classify import classify_document
from rfp_intel.ingestion.diff import diff_hashes
from rfp_intel.ingestion.discover import hash_folder
from rfp_intel.ingestion.parser import parse_file
from rfp_intel.logging_setup import get_logger
from rfp_intel.search.chunking import ChunkDraft, chunks_from_docling
from rfp_intel.search.embeddings import embed_documents
from rfp_intel.search.qdrant_store import delete_by_bid, delete_by_document, upsert_points

logger = get_logger("pipeline")


def process_bid_folder(folder: Path) -> None:
    """Run parse, chunk, and embed in order. Used by the explicit CLI index command."""
    parse_bid_folder(folder)
    chunk_bid_folder(folder)
    embed_bid_folder(folder)


def parse_bid_folder(folder: Path) -> None:
    """Parse new or changed files into source_files, sections, and tables."""
    folder = folder.resolve()

    def work(conn, bid_folder: str) -> None:
        bid = _ensure_bid(conn, folder, status="parsing")
        conn.commit()
        disk = hash_folder(folder)
        stored_rows = source_files_for_bid(conn, bid.id)
        stored = {row.relative_path: row for row in stored_rows}
        diff = diff_hashes(disk, {path: row.sha256 for path, row in stored.items()})
        for relative in diff.removed:
            _drop_file(conn, stored[relative])
            conn.commit()
        for relative in diff.added + diff.changed:
            existing = stored.get(relative)
            if existing is not None:
                current = get_source_file(conn, existing.id)
                if current is not None:
                    _drop_file(conn, current)
                    conn.commit()
            bid = get_bid_by_folder(conn, bid_folder)
            _ingest_file(conn, bid, folder / relative, relative, disk[relative])
            conn.commit()
        bid = get_bid_by_folder(conn, bid_folder)
        if bid is not None:
            bid.status = "parsed"
            bid.path = str(folder)
            save_bid(conn, bid)
            conn.commit()

    _with_bid_lock(folder.name, work)
    logger.info("bid parsed", extra={"extra_data": {"bid_id": folder.name}})


def chunk_bid_folder(folder: Path) -> None:
    """Build chunks for every file that has already been parsed."""
    folder = folder.resolve()

    def work(conn, bid_folder: str) -> None:
        bid = get_bid_by_folder(conn, bid_folder)
        if bid is None:
            raise RuntimeError(f"parse the bid before chunking: {bid_folder}")
        files = [
            row
            for row in source_files_for_bid(conn, bid.id)
            if row.parse_status.startswith("parsed")
        ]
        if not files:
            raise RuntimeError(f"no parsed files to chunk: {bid_folder}")
        bid.status = "chunking"
        save_bid(conn, bid)
        conn.commit()
        for source in files:
            _replace_chunks(conn, bid, source, folder / source.relative_path)
            conn.commit()
        bid = get_bid_by_folder(conn, bid_folder)
        if bid is not None:
            bid.status = "chunked"
            save_bid(conn, bid)
            conn.commit()

    _with_bid_lock(folder.name, work)
    logger.info("bid chunked", extra={"extra_data": {"bid_id": folder.name}})


def embed_bid_folder(folder: Path) -> None:
    """Embed chunks that do not yet have a Qdrant point."""
    folder = folder.resolve()

    def work(conn, bid_folder: str) -> None:
        bid = get_bid_by_folder(conn, bid_folder)
        if bid is None:
            raise RuntimeError(f"parse and chunk the bid before embedding: {bid_folder}")
        if not has_pending_embed(conn, bid_folder):
            if chunk_count(conn, bid_folder) == 0:
                raise RuntimeError(f"no chunks to embed: {bid_folder}")
            bid.status = "ready"
            save_bid(conn, bid)
            conn.commit()
            return
        bid.status = "embedding"
        save_bid(conn, bid)
        conn.commit()
        _embed_pending(conn, bid_folder)
        conn.commit()
        bid = get_bid_by_folder(conn, bid_folder)
        if bid is not None:
            bid.status = "ready" if not has_pending_embed(conn, bid_folder) else "chunked"
            save_bid(conn, bid)
            conn.commit()

    _with_bid_lock(folder.name, work)
    logger.info("bid embedded", extra={"extra_data": {"bid_id": folder.name}})


def _with_bid_lock(bid_folder: str, work) -> None:
    """Hold a session advisory lock for one bid while ``work`` commits its own steps."""
    lock_key = zlib.crc32(bid_folder.encode("utf-8")) & 0x7FFFFFFF
    conn = connect()
    try:
        conn.execute("SELECT pg_advisory_lock(%(key)s::bigint)", {"key": lock_key})
        conn.commit()
        try:
            work(conn, bid_folder)
        finally:
            conn.rollback()
            conn.execute("SELECT pg_advisory_unlock(%(key)s::bigint)", {"key": lock_key})
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_bid_folder(folder_name: str) -> None:
    with session_scope() as conn:
        bid = get_bid_by_folder(conn, folder_name)
        if bid is None:
            return
        delete_by_bid(folder_name)
        delete_bid(conn, bid.id)


def _ensure_bid(conn, folder: Path, status: str = "parsing") -> Bid:
    bid = get_bid_by_folder(conn, folder.name)
    if bid is None:
        bid = insert_bid(conn, Bid(folder_name=folder.name, path=str(folder.resolve()), status=status))
    else:
        bid.status = status
        bid.path = str(folder.resolve())
        save_bid(conn, bid)
    return bid


def _drop_file(conn, source: SourceFile) -> None:
    delete_by_document(str(source.id))
    delete_chunks_for_file(conn, source.id)
    delete_sections_for_file(conn, source.id)
    delete_source_file(conn, source.id)


def _ingest_file(conn, bid: Bid, path: Path, relative: str, digest: str) -> None:
    doc_type, addendum_number = classify_document(path)
    source = insert_source_file(
        conn,
        SourceFile(
            bid_id=bid.id,
            relative_path=relative,
            sha256=digest,
            doc_type=doc_type,
            addendum_number=addendum_number,
            parse_status="running",
        ),
    )
    try:
        parsed = parse_file(path)
        _store_sections(conn, source, parsed.sections)
        source.parse_status = "parsed" if not parsed.warnings else "parsed_with_warnings"
        source.error = "\n".join(parsed.warnings) or None
        source.parsed_at = datetime.now(timezone.utc)
    except Exception as exc:
        logger.exception("file parse failed", extra={"extra_data": {"file": relative, "bid_id": bid.folder_name}})
        source.parse_status = "failed"
        source.error = str(exc)
        source.parsed_at = datetime.now(timezone.utc)
    save_source_file(conn, source)


def _replace_chunks(conn, bid: Bid, source: SourceFile, path: Path) -> None:
    parsed = parse_file(path)
    drafts = _chunk_parsed(parsed, get_settings().chunk_max_tokens, get_settings().embedding_model)
    delete_by_document(str(source.id))
    delete_chunks_for_file(conn, source.id)
    section_ids = _section_index(conn, source.id)
    _store_chunks(conn, bid, source, drafts, section_ids, parsed.sections)


def _section_index(conn, document_id) -> dict[tuple, uuid.UUID]:
    index: dict[tuple, uuid.UUID] = {}
    for section in sections_for_document(conn, document_id):
        heading = tuple(section.heading_path)
        index[(heading, section.ordinal)] = section.id
        index[heading] = section.id
    return index


def _chunk_parsed(parsed, max_tokens: int, embedding_model: str) -> list[ChunkDraft]:
    if parsed.document is not None:
        try:
            drafts = chunks_from_docling(parsed.document, max_tokens=max_tokens, embedding_model=embedding_model)
            if drafts:
                return drafts
        except Exception:
            logger.exception("HybridChunker failed; using section chunker")
    from rfp_intel.search.chunking import chunk_sections

    return chunk_sections(parsed.sections, max_tokens=max_tokens)


def _store_sections(conn, source: SourceFile, sections) -> dict[tuple, uuid.UUID]:
    """Map a heading path to the last section id with that path."""
    index: dict[tuple, uuid.UUID] = {}
    for section in sections:
        row = insert_section(
            conn,
            Section(
                document_id=source.id,
                heading_path=list(section.heading_path),
                level=section.level,
                ordinal=section.ordinal,
                page_start=section.page_start,
                page_end=section.page_end,
                body_text=section.body_text or "",
            ),
        )
        index[(tuple(section.heading_path), section.ordinal)] = row.id
        index[tuple(section.heading_path)] = row.id
        for table in section.tables:
            insert_table(
                conn,
                Table(
                    section_id=row.id,
                    page_number=table.page_number,
                    markdown=table.markdown,
                    rows_json=table.rows,
                    caption=table.caption,
                ),
            )
    return index


def _store_chunks(conn, bid, source, drafts: list[ChunkDraft], section_ids, sections) -> None:
    ordinal_to_heading = {section.ordinal: tuple(section.heading_path) for section in sections}
    for draft in drafts:
        section_id = None
        if draft.section_ordinal is not None and draft.section_ordinal in ordinal_to_heading:
            key = (ordinal_to_heading[draft.section_ordinal], draft.section_ordinal)
            section_id = section_ids.get(key)
        if section_id is None:
            section_id = section_ids.get(tuple(draft.heading_path))
        insert_chunk(
            conn,
            Chunk(
                section_id=section_id,
                source_file_id=source.id,
                bid_id=bid.id,
                bid_folder=bid.folder_name,
                ordinal=draft.ordinal,
                text=draft.text,
                page_number=draft.page_number,
                token_count=draft.token_count,
                heading_path=list(draft.heading_path),
                doc_type=source.doc_type,
                addendum_number=source.addendum_number,
                file_name=Path(source.relative_path).name,
            ),
        )


def _embed_pending(conn, bid_folder: str) -> None:
    pending = pending_chunks(conn, bid_folder)
    if not pending:
        return
    batch_size = 32
    now = datetime.now(timezone.utc)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embed_documents([row.text for row in batch])
        points = []
        for row, vector in zip(batch, vectors):
            point_id = str(row.id)
            mark_chunk_embedded(conn, row.id, point_id, now)
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "bid_id": row.bid_folder,
                        "document_id": str(row.source_file_id),
                        "section_id": str(row.section_id) if row.section_id else "",
                        "doc_type": row.doc_type,
                        "addendum_number": row.addendum_number if row.addendum_number is not None else -1,
                        "page_number": row.page_number if row.page_number is not None else -1,
                        "file_name": row.file_name,
                    },
                )
            )
        upsert_points(points)
