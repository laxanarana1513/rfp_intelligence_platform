"""Queue and claim ingestion jobs. Only new or changed folders are enqueued."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rfp_intel.config import get_settings
from rfp_intel.db.models import Bid, IngestionJob
from rfp_intel.db.session import Conn, session_scope
from rfp_intel.db.store import (
    claim_pending_job,
    get_bid,
    get_job,
    has_pending_embed,
    insert_bid,
    insert_job,
    list_bids as list_bid_rows,
    open_job_for_bid,
    running_jobs,
    save_job,
    source_files_for_bid,
)
from rfp_intel.ingestion.diff import diff_hashes
from rfp_intel.ingestion.discover import hash_folder, list_bid_folders
from rfp_intel.ingestion.pipeline import delete_bid_folder, process_bid_folder
from rfp_intel.logging_setup import get_logger
from rfp_intel.resilience import backoff_delay

logger = get_logger("jobs")


def reconcile_data_dir(data_dir: Path | None = None) -> list[str]:
    """Enqueue a job for each folder that is new or whose files changed. Returns folder names."""
    settings = get_settings()
    root = (data_dir or settings.data_dir).resolve()
    enqueued: list[str] = []
    disk_folders = {folder.name: folder for folder in list_bid_folders(root)}
    with session_scope() as conn:
        _reset_stuck_jobs(conn)
        known = {bid.folder_name: bid for bid in list_bid_rows(conn)}
        removed = sorted(set(known) - set(disk_folders))
        for name in removed:
            logger.info("bid folder removed", extra={"extra_data": {"bid_id": name}})
    for name in removed:
        delete_bid_folder(name)

    with session_scope() as conn:
        known = {bid.folder_name: bid for bid in list_bid_rows(conn)}
        for name, folder in disk_folders.items():
            bid = known.get(name)
            if bid is None:
                bid = insert_bid(conn, Bid(folder_name=name, path=str(folder), status="pending"))
                _enqueue(conn, bid.id)
                enqueued.append(name)
                continue
            stored = {
                row.relative_path: row.sha256
                for row in source_files_for_bid(conn, bid.id)
            }
            # A failed parse is stored with its hash. Re-queue only when bytes change,
            # or when chunks were never embedded (recovered by process_bid_folder).
            diff = diff_hashes(hash_folder(folder), stored)
            if diff.has_changes or has_pending_embed(conn, name):
                if _enqueue(conn, bid.id):
                    enqueued.append(name)
    if enqueued:
        logger.info("jobs enqueued", extra={"extra_data": {"bids": enqueued}})
    return enqueued


def _reset_stuck_jobs(conn: Conn) -> None:
    for job in running_jobs(conn):
        job.status = "pending"
        job.error = "reset after process restart"
        save_job(conn, job)


def _enqueue(conn: Conn, bid_id) -> bool:
    if open_job_for_bid(conn, bid_id) is not None:
        return False
    insert_job(conn, IngestionJob(bid_id=bid_id, status="pending", attempt=0, next_run_at=datetime.now(timezone.utc)))
    return True


def claim_next_job() -> str | None:
    now = datetime.now(timezone.utc)
    with session_scope() as conn:
        job = claim_pending_job(conn, now)
        if job is None:
            return None
        return str(job.id)


def run_claimed_job(job_id: str) -> None:
    settings = get_settings()
    with session_scope() as conn:
        job = get_job(conn, _uuid(job_id))
        if job is None:
            return
        bid = get_bid(conn, job.bid_id)
        folder = Path(bid.path) if bid else None
        attempt = job.attempt
    if folder is None or not folder.exists():
        _finish_missing(job_id)
        return
    try:
        process_bid_folder(folder)
    except Exception as exc:
        logger.exception("ingestion job failed", extra={"extra_data": {"job_id": job_id}})
        _fail_or_retry(job_id, attempt, exc, settings)
        return
    with session_scope() as conn:
        job = get_job(conn, _uuid(job_id))
        if job is None:
            return
        job.status = "succeeded"
        job.finished_at = datetime.now(timezone.utc)
        job.error = None
        save_job(conn, job)


def _fail_or_retry(job_id: str, attempt: int, exc: Exception, settings) -> None:
    with session_scope() as conn:
        job = get_job(conn, _uuid(job_id))
        if job is None:
            return
        next_attempt = attempt + 1
        job.attempt = next_attempt
        job.error = str(exc)
        job.finished_at = datetime.now(timezone.utc)
        if next_attempt >= settings.ingestion_max_attempts:
            job.status = "failed"
        else:
            delay = backoff_delay(attempt, settings.backoff_base_seconds, settings.backoff_cap_seconds)
            job.status = "pending"
            job.next_run_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        save_job(conn, job)


def _finish_missing(job_id: str) -> None:
    with session_scope() as conn:
        job = get_job(conn, _uuid(job_id))
        if job is None:
            return
        job.status = "failed"
        job.error = "bid folder is no longer on disk"
        job.finished_at = datetime.now(timezone.utc)
        save_job(conn, job)


def _uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value)


def list_jobs(limit: int = 50) -> list[dict]:
    with session_scope() as conn:
        rows = conn.execute(
            """
            SELECT j.id::text AS id, b.folder_name, j.status, j.attempt, j.error,
                   j.started_at, j.finished_at, j.next_run_at
            FROM ingestion_jobs j
            JOIN bids b ON b.id = j.bid_id
            ORDER BY j.next_run_at DESC
            LIMIT %(limit)s
            """,
            {"limit": limit},
        ).fetchall()
        return [dict(row) for row in rows]


def list_bids() -> list[dict]:
    with session_scope() as conn:
        rows = conn.execute(
            """
            SELECT b.folder_name, b.status, b.path,
                   count(f.id) AS files,
                   count(f.id) FILTER (WHERE f.parse_status = 'failed') AS failed_files
            FROM bids b
            LEFT JOIN source_files f ON f.bid_id = b.id
            GROUP BY b.id
            ORDER BY b.folder_name
            """
        ).fetchall()
        return [dict(row) for row in rows]
