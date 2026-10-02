"""Optional folder watcher. The API does not start it; pipeline steps are manual."""

from __future__ import annotations

import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rfp_intel.config import get_settings
from rfp_intel.db.migrate import upgrade
from rfp_intel.ingestion.jobs import claim_next_job, reconcile_data_dir, run_claimed_job
from rfp_intel.logging_setup import get_logger
from rfp_intel.resilience import backoff_delay

logger = get_logger("watcher")


def watch_loop(stop: threading.Event) -> None:
    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while not stop.is_set():
        try:
            upgrade()
            break
        except Exception:
            logger.exception("database is not ready")
            delay = backoff_delay(attempt, settings.backoff_base_seconds, settings.backoff_cap_seconds)
            attempt += 1
            stop.wait(delay)

    dirty = threading.Event()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):  # noqa: ANN001
            dirty.set()

    observer = Observer()
    observer.schedule(Handler(), str(data_dir), recursive=True)
    observer.start()
    last_dirty: float | None = None
    try:
        try:
            reconcile_data_dir(data_dir)
        except Exception:
            logger.exception("startup reconcile failed")
        while not stop.is_set():
            if dirty.is_set():
                dirty.clear()
                last_dirty = time.time()
            if last_dirty is not None and time.time() - last_dirty >= settings.watcher_debounce_seconds:
                try:
                    reconcile_data_dir(data_dir)
                except Exception:
                    logger.exception("reconcile failed")
                last_dirty = None
            try:
                job_id = claim_next_job()
            except Exception:
                logger.exception("could not claim a job")
                stop.wait(1)
                continue
            if job_id:
                run_claimed_job(job_id)
                continue
            stop.wait(1)
    finally:
        observer.stop()
        observer.join(timeout=5)
