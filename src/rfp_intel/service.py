"""Operations shared by the CLI, the API, and Streamlit."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rfp_intel.agents.graph import build_ask_graph, build_extract_graph
from rfp_intel.config import get_settings
from rfp_intel.db.models import AddendumChangeRow, AgentRun, ExtractedField
from rfp_intel.db.session import session_scope
from rfp_intel.db.store import (
    agent_steps_for_run,
    changes_for_run,
    chunk_count,
    fields_for_run,
    get_agent_run,
    get_bid_by_folder,
    insert_addendum_change,
    insert_agent_run,
    insert_extracted_field,
    latest_extract_run,
    save_agent_run,
)
from rfp_intel.ingestion.browse import pipeline_catalog, step_rows
from rfp_intel.ingestion.discover import list_bid_folders
from rfp_intel.ingestion.jobs import list_bids
from rfp_intel.ingestion.pipeline import chunk_bid_folder, embed_bid_folder, parse_bid_folder
from rfp_intel.search.qdrant_store import point_count
from rfp_intel.observability.trace import bind_trace, reset_trace
from rfp_intel.schemas import AskResponse, BidRecord, SearchHit, SourceCitation
from rfp_intel.search.hybrid import search


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def index_folder(path: str, *, synchronous: bool = True) -> dict:
    """Run parse, chunk, and embed for one folder. Nothing starts this on its own."""
    del synchronous
    folder = Path(path).resolve()
    if not folder.is_dir():
        folder = (get_settings().data_dir / path).resolve()
    if not folder.is_dir():
        raise ServiceError(f"bid folder does not exist: {path}", 404)
    parse_bid_folder(folder)
    chunk_bid_folder(folder)
    embed_bid_folder(folder)
    return {"bid_id": folder.name, "status": "ready"}


def pipeline_status() -> dict:
    try:
        points = point_count()
    except Exception as exc:
        return {
            "retrieval_ready": False,
            "qdrant_points": 0,
            "qdrant_error": str(exc),
            "bids": _safe_catalog(),
        }
    return {"retrieval_ready": points > 0, "qdrant_points": points, "qdrant_error": None, "bids": _safe_catalog()}


def pipeline_rows(bid_id: str, step: str, file: str | None = None) -> dict:
    _known_folder(bid_id)
    try:
        rows = step_rows(bid_id, step, file=file)
    except ValueError as exc:
        raise ServiceError(str(exc)) from exc
    return {"bid_id": bid_id, "step": step, "file": file or None, "rows": rows}


def run_parse(bid_id: str) -> dict:
    folder = _known_folder(bid_id)
    try:
        parse_bid_folder(folder)
    except Exception as exc:
        raise ServiceError(str(exc), 500) from exc
    return {"bid_id": bid_id, "step": "parse", "status": "parsed"}


def run_chunk(bid_id: str) -> dict:
    folder = _known_folder(bid_id)
    bid = _catalog_row(bid_id)
    if not bid["can_chunk"]:
        raise ServiceError("Parse this bid before chunking. No parsed files are stored yet.", 409)
    try:
        chunk_bid_folder(folder)
    except Exception as exc:
        raise ServiceError(str(exc), 500) from exc
    return {"bid_id": bid_id, "step": "chunk", "status": "chunked"}


def run_embed(bid_id: str) -> dict:
    folder = _known_folder(bid_id)
    bid = _catalog_row(bid_id)
    if bid["chunks"] == 0:
        raise ServiceError("Chunk this bid before embedding. The chunks table is empty.", 409)
    if not bid["can_embed"]:
        return {"bid_id": bid_id, "step": "embed", "status": "ready", "detail": "Every chunk already has a Qdrant point."}
    try:
        embed_bid_folder(folder)
    except Exception as exc:
        raise ServiceError(str(exc), 500) from exc
    return {"bid_id": bid_id, "step": "embed", "status": "ready"}


def search_bids(
    query: str,
    *,
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
) -> list[SearchHit]:
    if not query.strip():
        raise ServiceError("query is required")
    _require_qdrant_data()
    with session_scope() as conn:
        return search(
            conn,
            query,
            bid_id=bid_id,
            doc_type=doc_type,
            addendum_number=addendum_number,
            top_k=top_k,
            mode=mode,
        )


def extract_bid(bid_folder: str) -> BidRecord:
    _require_qdrant_data()
    _require_indexed(bid_folder)
    run_id = uuid.uuid4()
    with session_scope() as conn:
        insert_agent_run(conn, AgentRun(id=run_id, bid_folder=bid_folder, mode="extract", status="running"))
    tokens = bind_trace(str(run_id))
    try:
        final = build_extract_graph().invoke({"bid_folder": bid_folder})
        record = BidRecord.model_validate(final["record"])
        _persist_record(run_id, record)
        _write_output(record)
        _export_trace(run_id)
        _finish_run(run_id, "succeeded")
        return record
    except Exception as exc:
        _finish_run(run_id, "failed", str(exc))
        raise
    finally:
        reset_trace(tokens)


def ask_question(question: str, bid_id: str | None = None) -> AskResponse:
    if not question.strip():
        raise ServiceError("question is required")
    _require_qdrant_data()
    bid_ids = _bids_for_question(question, bid_id)
    run_id = uuid.uuid4()
    with session_scope() as conn:
        insert_agent_run(
            conn,
            AgentRun(
                id=run_id,
                bid_folder=bid_id,
                mode="ask",
                status="running",
                question=question,
            ),
        )
    tokens = bind_trace(str(run_id))
    try:
        final = build_ask_graph().invoke({"question": question, "bid_ids": bid_ids})
        response = AskResponse(
            answer=final.get("answer") or "Not found in documents",
            citations=[SourceCitation.model_validate(item) for item in final.get("citations") or []],
            bid_id=bid_id,
        )
        _finish_run(run_id, "succeeded")
        _export_trace(run_id)
        return response
    except Exception as exc:
        _finish_run(run_id, "failed", str(exc))
        raise
    finally:
        reset_trace(tokens)


def latest_extraction(bid_folder: str) -> BidRecord | None:
    with session_scope() as conn:
        run = latest_extract_run(conn, bid_folder)
        if run is None:
            return None
        return _record_from_run(run, fields_for_run(conn, run.id), changes_for_run(conn, run.id))


def _record_from_run(run: AgentRun, fields, changes) -> BidRecord:
    record = BidRecord.model_validate(
        {
            "bid_id": run.bid_folder,
            "fields": {
                row.field_name: {
                    "value": row.value,
                    "sources": row.sources or [],
                    "confidence": row.confidence,
                    "notes": row.notes,
                }
                for row in fields
            },
            "addendum_changes": [
                {
                    "field": row.field_name,
                    "previous_value": row.previous_value,
                    "new_value": row.new_value,
                    "addendum_number": row.addendum_number,
                    "file": row.file_name,
                    "page": row.page_number,
                }
                for row in changes
            ],
            "validation": {"passed": 0, "failed": 0, "not_found": 0},
        }
    )
    passed = sum(1 for field in record.fields.values() if field.value)
    not_found = sum(1 for field in record.fields.values() if field.value is None and field.notes == "Not found in documents")
    failed = len(record.fields) - passed - not_found
    record.validation.passed = passed
    record.validation.failed = max(failed, 0)
    record.validation.not_found = not_found
    return record


def _persist_record(run_id: uuid.UUID, record: BidRecord) -> None:
    with session_scope() as conn:
        for name, field in record.fields.items():
            insert_extracted_field(
                conn,
                ExtractedField(
                    run_id=run_id,
                    bid_folder=record.bid_id,
                    field_name=name,
                    value=field.value,
                    sources=[source.model_dump() for source in field.sources],
                    confidence=field.confidence,
                    notes=field.notes,
                ),
            )
        for change in record.addendum_changes:
            insert_addendum_change(
                conn,
                AddendumChangeRow(
                    run_id=run_id,
                    bid_folder=record.bid_id,
                    field_name=change.field,
                    previous_value=change.previous_value,
                    new_value=change.new_value,
                    addendum_number=change.addendum_number,
                    file_name=change.file,
                    page_number=change.page,
                ),
            )


def _write_output(record: BidRecord) -> None:
    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    path = settings.output_dir / f"{record.bid_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def _export_trace(run_id: uuid.UUID) -> None:
    settings = get_settings()
    trace_dir = settings.output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with session_scope() as conn:
        rows = agent_steps_for_run(conn, run_id)
        steps = []
        for row in rows:
            item = dict(row)
            created = item.get("created_at")
            if isinstance(created, datetime):
                item["created_at"] = created.isoformat()
            steps.append(item)
    (trace_dir / f"{run_id}.json").write_text(json.dumps(steps, indent=2, default=str), encoding="utf-8")


def _finish_run(run_id: uuid.UUID, status: str, error: str | None = None) -> None:
    with session_scope() as conn:
        run = get_agent_run(conn, run_id)
        if run is None:
            return
        run.status = status
        run.error = error
        run.finished_at = datetime.now(timezone.utc)
        save_agent_run(conn, run)


def _require_qdrant_data() -> None:
    try:
        count = point_count()
    except Exception as exc:
        raise ServiceError(f"Qdrant is unavailable: {exc}", 503) from exc
    if count <= 0:
        raise ServiceError(
            "Search, ask, and extract stay off until at least one bid has been embedded in Qdrant.",
            409,
        )


def _known_folder(bid_id: str) -> Path:
    folder = (get_settings().data_dir / bid_id).resolve()
    root = get_settings().data_dir.resolve()
    if folder != root and root not in folder.parents:
        raise ServiceError("bid folder must sit inside the data directory", 400)
    if not folder.is_dir():
        raise ServiceError(f"bid folder does not exist: {bid_id}", 404)
    if not any(path.name == bid_id for path in list_bid_folders(root)):
        raise ServiceError(f"no documents found in {bid_id}", 404)
    return folder


def _catalog_row(bid_id: str) -> dict:
    for row in pipeline_catalog():
        if row["folder_name"] == bid_id:
            return row
    raise ServiceError(f"bid folder does not exist: {bid_id}", 404)


def _safe_catalog() -> list[dict]:
    try:
        return pipeline_catalog()
    except Exception:
        return [
            {
                "folder_name": folder.name,
                "path": str(folder),
                "status": "not_started",
                "files": 0,
                "parsed_files": 0,
                "failed_files": 0,
                "sections": 0,
                "tables": 0,
                "chunks": 0,
                "embedded": 0,
                "can_chunk": False,
                "can_embed": False,
            }
            for folder in list_bid_folders(get_settings().data_dir)
        ]


def _require_indexed(bid_folder: str) -> None:
    with session_scope() as conn:
        bid = get_bid_by_folder(conn, bid_folder)
        if bid is None:
            raise ServiceError(f"bid is not indexed: {bid_folder}", 404)
        if not chunk_count(conn, bid_folder):
            raise ServiceError(f"bid has no indexed chunks yet: {bid_folder}", 409)


def _bids_for_question(question: str, bid_id: str | None) -> list[str | None]:
    if bid_id:
        return [bid_id]
    lowered = question.lower()
    names = _known_bid_names()
    if any(word in lowered for word in ("compare", "both", "each bid", "all bids")):
        return names or [None]
    for name in names:
        if name.lower() in lowered:
            return [name]
    return [None]


def _known_bid_names() -> list[str]:
    try:
        return [row["folder_name"] for row in list_bids()]
    except Exception:
        return [folder.name for folder in list_bid_folders(get_settings().data_dir)]
