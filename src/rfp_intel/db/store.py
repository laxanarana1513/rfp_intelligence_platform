"""Insert, update, and load rows. Callers own the transaction."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from rfp_intel.db.models import (
    AddendumChangeRow,
    AgentRun,
    AgentStep,
    Bid,
    Chunk,
    ExtractedField,
    IngestionJob,
    Section,
    SourceFile,
    Table,
)
from rfp_intel.db.session import Conn

_BID = "id, folder_name, path, status, first_seen_at, updated_at"
_FILE = "id, bid_id, relative_path, sha256, doc_type, addendum_number, parse_status, error, parsed_at"
_SECTION = "id, document_id, heading_path, level, ordinal, page_start, page_end, body_text"
_CHUNK = (
    "id, section_id, source_file_id, bid_id, bid_folder, ordinal, text, page_number, "
    "token_count, heading_path, doc_type, addendum_number, file_name, embedded_at, qdrant_point_id"
)
_JOB = "id, bid_id, status, attempt, next_run_at, started_at, finished_at, error"
_RUN = "id, bid_folder, mode, status, question, plan, started_at, finished_at, error"
_FIELD = "id, run_id, bid_folder, field_name, value, sources, confidence, notes"
_CHANGE = "id, run_id, bid_folder, field_name, previous_value, new_value, addendum_number, file_name, page_number"


def jsonb(value):
    if value is None:
        return None
    return Jsonb(value)


def get_bid_by_folder(conn: Conn, folder_name: str) -> Bid | None:
    row = conn.execute(
        f"SELECT {_BID} FROM bids WHERE folder_name = %(folder_name)s",
        {"folder_name": folder_name},
    ).fetchone()
    return Bid.model_validate(row) if row else None


def get_bid(conn: Conn, bid_id: uuid.UUID) -> Bid | None:
    row = conn.execute(f"SELECT {_BID} FROM bids WHERE id = %(id)s", {"id": bid_id}).fetchone()
    return Bid.model_validate(row) if row else None


def list_bids(conn: Conn) -> list[Bid]:
    rows = conn.execute(f"SELECT {_BID} FROM bids ORDER BY folder_name").fetchall()
    return [Bid.model_validate(row) for row in rows]


def insert_bid(conn: Conn, bid: Bid) -> Bid:
    row = conn.execute(
        f"""
        INSERT INTO bids (id, folder_name, path, status)
        VALUES (%(id)s, %(folder_name)s, %(path)s, %(status)s)
        RETURNING {_BID}
        """,
        {"id": bid.id, "folder_name": bid.folder_name, "path": bid.path, "status": bid.status},
    ).fetchone()
    return Bid.model_validate(row)


def save_bid(conn: Conn, bid: Bid) -> None:
    conn.execute(
        """
        UPDATE bids
        SET path = %(path)s, status = %(status)s, updated_at = now()
        WHERE id = %(id)s
        """,
        {"id": bid.id, "path": bid.path, "status": bid.status},
    )


def delete_bid(conn: Conn, bid_id: uuid.UUID) -> None:
    conn.execute("DELETE FROM bids WHERE id = %(id)s", {"id": bid_id})


def source_files_for_bid(conn: Conn, bid_id: uuid.UUID) -> list[SourceFile]:
    rows = conn.execute(
        f"SELECT {_FILE} FROM source_files WHERE bid_id = %(bid_id)s",
        {"bid_id": bid_id},
    ).fetchall()
    return [SourceFile.model_validate(row) for row in rows]


def get_source_file(conn: Conn, file_id: uuid.UUID) -> SourceFile | None:
    row = conn.execute(
        f"SELECT {_FILE} FROM source_files WHERE id = %(id)s",
        {"id": file_id},
    ).fetchone()
    return SourceFile.model_validate(row) if row else None


def insert_source_file(conn: Conn, source: SourceFile) -> SourceFile:
    row = conn.execute(
        f"""
        INSERT INTO source_files (
            id, bid_id, relative_path, sha256, doc_type, addendum_number, parse_status, error, parsed_at
        )
        VALUES (
            %(id)s, %(bid_id)s, %(relative_path)s, %(sha256)s, %(doc_type)s,
            %(addendum_number)s, %(parse_status)s, %(error)s, %(parsed_at)s
        )
        RETURNING {_FILE}
        """,
        source.model_dump(),
    ).fetchone()
    return SourceFile.model_validate(row)


def save_source_file(conn: Conn, source: SourceFile) -> None:
    conn.execute(
        """
        UPDATE source_files
        SET sha256 = %(sha256)s,
            doc_type = %(doc_type)s,
            addendum_number = %(addendum_number)s,
            parse_status = %(parse_status)s,
            error = %(error)s,
            parsed_at = %(parsed_at)s
        WHERE id = %(id)s
        """,
        source.model_dump(),
    )


def delete_chunks_for_file(conn: Conn, source_file_id: uuid.UUID) -> None:
    conn.execute("DELETE FROM chunks WHERE source_file_id = %(id)s", {"id": source_file_id})


def delete_sections_for_file(conn: Conn, document_id: uuid.UUID) -> None:
    conn.execute("DELETE FROM sections WHERE document_id = %(id)s", {"id": document_id})


def delete_source_file(conn: Conn, file_id: uuid.UUID) -> None:
    conn.execute("DELETE FROM source_files WHERE id = %(id)s", {"id": file_id})


def insert_section(conn: Conn, section: Section) -> Section:
    row = conn.execute(
        f"""
        INSERT INTO sections (id, document_id, heading_path, level, ordinal, page_start, page_end, body_text)
        VALUES (
            %(id)s, %(document_id)s, %(heading_path)s, %(level)s, %(ordinal)s,
            %(page_start)s, %(page_end)s, %(body_text)s
        )
        RETURNING {_SECTION}
        """,
        section.model_dump(),
    ).fetchone()
    return Section.model_validate(row)


def sections_for_document(conn: Conn, document_id: uuid.UUID) -> list[Section]:
    rows = conn.execute(
        f"SELECT {_SECTION} FROM sections WHERE document_id = %(document_id)s ORDER BY ordinal",
        {"document_id": document_id},
    ).fetchall()
    return [Section.model_validate(row) for row in rows]


def insert_table(conn: Conn, table: Table) -> None:
    payload = table.model_dump()
    payload["rows_json"] = jsonb(table.rows_json)
    conn.execute(
        """
        INSERT INTO tables (id, section_id, page_number, markdown, rows_json, caption)
        VALUES (%(id)s, %(section_id)s, %(page_number)s, %(markdown)s, %(rows_json)s, %(caption)s)
        """,
        payload,
    )


def insert_chunk(conn: Conn, chunk: Chunk) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            id, section_id, source_file_id, bid_id, bid_folder, ordinal, text, page_number,
            token_count, heading_path, doc_type, addendum_number, file_name
        )
        VALUES (
            %(id)s, %(section_id)s, %(source_file_id)s, %(bid_id)s, %(bid_folder)s, %(ordinal)s,
            %(text)s, %(page_number)s, %(token_count)s, %(heading_path)s, %(doc_type)s,
            %(addendum_number)s, %(file_name)s
        )
        """,
        chunk.model_dump(),
    )


def pending_chunks(conn: Conn, bid_folder: str) -> list[Chunk]:
    rows = conn.execute(
        f"""
        SELECT {_CHUNK}
        FROM chunks
        WHERE bid_folder = %(bid_folder)s AND embedded_at IS NULL
        ORDER BY ordinal
        """,
        {"bid_folder": bid_folder},
    ).fetchall()
    return [Chunk.model_validate(row) for row in rows]


def chunks_by_ids(conn: Conn, ids: list[uuid.UUID]) -> list[Chunk]:
    if not ids:
        return []
    rows = conn.execute(
        f"SELECT {_CHUNK} FROM chunks WHERE id = ANY(%(ids)s)",
        {"ids": ids},
    ).fetchall()
    return [Chunk.model_validate(row) for row in rows]


def mark_chunk_embedded(conn: Conn, chunk_id: uuid.UUID, point_id: str, embedded_at) -> None:
    conn.execute(
        """
        UPDATE chunks
        SET qdrant_point_id = %(point_id)s, embedded_at = %(embedded_at)s
        WHERE id = %(id)s
        """,
        {"id": chunk_id, "point_id": point_id, "embedded_at": embedded_at},
    )


def chunk_count(conn: Conn, bid_folder: str) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM chunks WHERE bid_folder = %(name)s",
        {"name": bid_folder},
    ).fetchone()
    return int(row["n"]) if row else 0


def has_pending_embed(conn: Conn, bid_folder: str) -> bool:
    row = conn.execute(
        "SELECT 1 AS found FROM chunks WHERE bid_folder = %(name)s AND embedded_at IS NULL LIMIT 1",
        {"name": bid_folder},
    ).fetchone()
    return row is not None


def insert_job(conn: Conn, job: IngestionJob) -> IngestionJob:
    row = conn.execute(
        f"""
        INSERT INTO ingestion_jobs (id, bid_id, status, attempt, next_run_at)
        VALUES (%(id)s, %(bid_id)s, %(status)s, %(attempt)s, %(next_run_at)s)
        RETURNING {_JOB}
        """,
        job.model_dump(),
    ).fetchone()
    return IngestionJob.model_validate(row)


def get_job(conn: Conn, job_id: uuid.UUID) -> IngestionJob | None:
    row = conn.execute(f"SELECT {_JOB} FROM ingestion_jobs WHERE id = %(id)s", {"id": job_id}).fetchone()
    return IngestionJob.model_validate(row) if row else None


def save_job(conn: Conn, job: IngestionJob) -> None:
    conn.execute(
        """
        UPDATE ingestion_jobs
        SET status = %(status)s,
            attempt = %(attempt)s,
            next_run_at = %(next_run_at)s,
            started_at = %(started_at)s,
            finished_at = %(finished_at)s,
            error = %(error)s
        WHERE id = %(id)s
        """,
        job.model_dump(),
    )


def open_job_for_bid(conn: Conn, bid_id: uuid.UUID) -> IngestionJob | None:
    row = conn.execute(
        f"""
        SELECT {_JOB}
        FROM ingestion_jobs
        WHERE bid_id = %(bid_id)s AND status IN ('pending', 'running')
        LIMIT 1
        """,
        {"bid_id": bid_id},
    ).fetchone()
    return IngestionJob.model_validate(row) if row else None


def running_jobs(conn: Conn) -> list[IngestionJob]:
    rows = conn.execute(f"SELECT {_JOB} FROM ingestion_jobs WHERE status = 'running'").fetchall()
    return [IngestionJob.model_validate(row) for row in rows]


def claim_pending_job(conn: Conn, now) -> IngestionJob | None:
    row = conn.execute(
        f"""
        SELECT {_JOB}
        FROM ingestion_jobs
        WHERE status = 'pending' AND next_run_at <= %(now)s
        ORDER BY next_run_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """,
        {"now": now},
    ).fetchone()
    if row is None:
        return None
    job = IngestionJob.model_validate(row)
    job.status = "running"
    job.started_at = now
    job.error = None
    save_job(conn, job)
    return job


def insert_agent_run(conn: Conn, run: AgentRun) -> AgentRun:
    payload = run.model_dump()
    payload["plan"] = jsonb(run.plan)
    row = conn.execute(
        f"""
        INSERT INTO agent_runs (id, bid_folder, mode, status, question, plan)
        VALUES (%(id)s, %(bid_folder)s, %(mode)s, %(status)s, %(question)s, %(plan)s)
        RETURNING {_RUN}
        """,
        payload,
    ).fetchone()
    return AgentRun.model_validate(row)


def get_agent_run(conn: Conn, run_id: uuid.UUID) -> AgentRun | None:
    row = conn.execute(f"SELECT {_RUN} FROM agent_runs WHERE id = %(id)s", {"id": run_id}).fetchone()
    return AgentRun.model_validate(row) if row else None


def save_agent_run(conn: Conn, run: AgentRun) -> None:
    payload = run.model_dump()
    payload["plan"] = jsonb(run.plan)
    conn.execute(
        """
        UPDATE agent_runs
        SET bid_folder = %(bid_folder)s,
            mode = %(mode)s,
            status = %(status)s,
            question = %(question)s,
            plan = %(plan)s,
            finished_at = %(finished_at)s,
            error = %(error)s
        WHERE id = %(id)s
        """,
        payload,
    )


def latest_extract_run(conn: Conn, bid_folder: str) -> AgentRun | None:
    row = conn.execute(
        f"""
        SELECT {_RUN}
        FROM agent_runs
        WHERE bid_folder = %(bid_folder)s AND mode = 'extract' AND status = 'succeeded'
        ORDER BY finished_at DESC NULLS LAST
        LIMIT 1
        """,
        {"bid_folder": bid_folder},
    ).fetchone()
    return AgentRun.model_validate(row) if row else None


def insert_agent_step(conn: Conn, step: AgentStep) -> None:
    payload = step.model_dump()
    payload["input_json"] = jsonb(step.input_json)
    payload["output_json"] = jsonb(step.output_json)
    conn.execute(
        """
        INSERT INTO agent_steps (id, run_id, agent, input_json, output_json, tokens, latency_ms)
        VALUES (%(id)s, %(run_id)s, %(agent)s, %(input_json)s, %(output_json)s, %(tokens)s, %(latency_ms)s)
        """,
        payload,
    )


def agent_steps_for_run(conn: Conn, run_id: uuid.UUID) -> list[dict]:
    rows = conn.execute(
        """
        SELECT agent, input_json, output_json, tokens, latency_ms, created_at
        FROM agent_steps
        WHERE run_id = %(run_id)s
        ORDER BY created_at
        """,
        {"run_id": run_id},
    ).fetchall()
    return [dict(row) for row in rows]


def insert_extracted_field(conn: Conn, field: ExtractedField) -> None:
    payload = field.model_dump()
    payload["sources"] = jsonb(field.sources)
    conn.execute(
        f"""
        INSERT INTO extracted_fields ({_FIELD})
        VALUES (
            %(id)s, %(run_id)s, %(bid_folder)s, %(field_name)s, %(value)s,
            %(sources)s, %(confidence)s, %(notes)s
        )
        """,
        payload,
    )


def fields_for_run(conn: Conn, run_id: uuid.UUID) -> list[ExtractedField]:
    rows = conn.execute(
        f"SELECT {_FIELD} FROM extracted_fields WHERE run_id = %(run_id)s",
        {"run_id": run_id},
    ).fetchall()
    return [ExtractedField.model_validate(row) for row in rows]


def insert_addendum_change(conn: Conn, change: AddendumChangeRow) -> None:
    conn.execute(
        f"""
        INSERT INTO addendum_changes ({_CHANGE})
        VALUES (
            %(id)s, %(run_id)s, %(bid_folder)s, %(field_name)s, %(previous_value)s,
            %(new_value)s, %(addendum_number)s, %(file_name)s, %(page_number)s
        )
        """,
        change.model_dump(),
    )


def changes_for_run(conn: Conn, run_id: uuid.UUID) -> list[AddendumChangeRow]:
    rows = conn.execute(
        f"SELECT {_CHANGE} FROM addendum_changes WHERE run_id = %(run_id)s",
        {"run_id": run_id},
    ).fetchall()
    return [AddendumChangeRow.model_validate(row) for row in rows]
