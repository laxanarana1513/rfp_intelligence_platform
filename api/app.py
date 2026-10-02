"""FastAPI surface. Ingestion is started by the user, one bid and one step at a time."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rfp_intel.config import get_settings
from rfp_intel.db.migrate import upgrade
from rfp_intel.ingestion.jobs import list_bids, list_jobs
from rfp_intel.logging_setup import get_logger, setup_logging
from rfp_intel.service import (
    ServiceError,
    ask_question,
    extract_bid,
    index_folder,
    latest_extraction,
    pipeline_rows,
    pipeline_status,
    run_chunk,
    run_embed,
    run_parse,
    search_bids,
)

logger = get_logger("api")


class IndexRequest(BaseModel):
    bid_path: str


class AskRequest(BaseModel):
    question: str
    bid_id: str | None = None


class ExtractRequest(BaseModel):
    bid_id: str = Field(description="Folder name under data/, for example 'Bid 1'")


class BidStepRequest(BaseModel):
    bid_id: str = Field(description="Folder name under data/, for example 'Bid 1'")


def _jsonable(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                item[key] = value.isoformat()
            else:
                item[key] = value
        converted.append(item)
    return converted


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(get_settings().log_level)
        try:
            upgrade()
        except Exception:
            logger.exception("database migration failed")
        yield

    app = FastAPI(title="RFP Intelligence Platform", lifespan=lifespan)

    @app.get("/health")
    def health():
        db_ok = False
        qdrant_ok = False
        try:
            from rfp_intel.db.session import session_scope

            with session_scope() as conn:
                conn.execute("SELECT 1")
            db_ok = True
        except Exception as exc:
            db_error = str(exc)
        else:
            db_error = None
        try:
            from rfp_intel.search.qdrant_store import get_client

            get_client().get_collections()
            qdrant_ok = True
        except Exception as exc:
            qdrant_error = str(exc)
        else:
            qdrant_error = None
        status = "ok" if db_ok and qdrant_ok else "degraded"
        return {"status": status, "postgres": db_ok, "qdrant": qdrant_ok, "postgres_error": db_error, "qdrant_error": qdrant_error}

    @app.post("/index")
    def index(body: IndexRequest):
        try:
            return index_folder(body.bid_path, synchronous=False)
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.get("/search")
    def search(
        q: str = Query(...),
        bid_id: str | None = None,
        doc_type: str | None = None,
        addendum_number: int | None = None,
        top_k: int = 5,
        mode: str = "hybrid_rerank",
    ):
        try:
            hits = search_bids(
                q,
                bid_id=bid_id,
                doc_type=doc_type,
                addendum_number=addendum_number,
                top_k=top_k,
                mode=mode,
            )
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        return {"query": q, "hits": [hit.model_dump() for hit in hits]}

    @app.post("/ask")
    def ask(body: AskRequest):
        try:
            return ask_question(body.question, body.bid_id).model_dump()
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/extract")
    def extract(body: ExtractRequest):
        try:
            return extract_bid(body.bid_id).model_dump()
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.get("/pipeline/status")
    def status():
        try:
            return pipeline_status()
        except Exception as exc:
            raise HTTPException(503, f"pipeline status is unavailable: {exc}") from exc

    @app.get("/pipeline/rows")
    def rows(bid_id: str = Query(...), step: str = Query(...), file: str | None = None):
        try:
            return pipeline_rows(bid_id, step, file=file)
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/pipeline/parse")
    def parse(body: BidStepRequest):
        try:
            return run_parse(body.bid_id)
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/pipeline/chunk")
    def chunk(body: BidStepRequest):
        try:
            return run_chunk(body.bid_id)
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/pipeline/embed")
    def embed(body: BidStepRequest):
        try:
            return run_embed(body.bid_id)
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.get("/jobs")
    def jobs():
        try:
            return {"jobs": _jsonable(list_jobs()), "bids": _jsonable(list_bids())}
        except Exception as exc:
            raise HTTPException(503, f"job status is unavailable: {exc}") from exc

    @app.get("/extractions")
    def extraction(bid_id: str = Query(...)):
        try:
            record = latest_extraction(bid_id)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
        if record is None:
            raise HTTPException(404, "no extraction for this bid")
        return record.model_dump()

    return app


app = create_app()
