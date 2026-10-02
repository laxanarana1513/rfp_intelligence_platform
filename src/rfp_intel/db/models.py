"""Pydantic row models for documents, chunks, jobs, and agent traces."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Bid(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    folder_name: str
    path: str
    status: str = "pending"
    first_seen_at: datetime | None = None
    updated_at: datetime | None = None


class SourceFile(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    bid_id: uuid.UUID
    relative_path: str
    sha256: str = ""
    doc_type: str = "rfp"
    addendum_number: int | None = None
    parse_status: str = "pending"
    error: str | None = None
    parsed_at: datetime | None = None


class Section(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document_id: uuid.UUID
    heading_path: list[str] = Field(default_factory=list)
    level: int = 0
    ordinal: int = 0
    page_start: int | None = None
    page_end: int | None = None
    body_text: str = ""


class Table(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    section_id: uuid.UUID
    page_number: int | None = None
    markdown: str = ""
    rows_json: list = Field(default_factory=list)
    caption: str | None = None


class Chunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    section_id: uuid.UUID | None = None
    source_file_id: uuid.UUID
    bid_id: uuid.UUID
    bid_folder: str
    ordinal: int
    text: str
    page_number: int | None = None
    token_count: int = 0
    heading_path: list[str] = Field(default_factory=list)
    doc_type: str = "rfp"
    addendum_number: int | None = None
    file_name: str
    embedded_at: datetime | None = None
    qdrant_point_id: str | None = None


class IngestionJob(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    bid_id: uuid.UUID
    status: str = "pending"
    attempt: int = 0
    next_run_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class AgentRun(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    bid_folder: str | None = None
    mode: str
    status: str = "running"
    question: str | None = None
    plan: dict | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class AgentStep(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: uuid.UUID
    agent: str
    input_json: dict | None = None
    output_json: dict | None = None
    tokens: int = 0
    latency_ms: int = 0
    created_at: datetime | None = None


class ExtractedField(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: uuid.UUID
    bid_folder: str
    field_name: str
    value: str | None = None
    sources: list = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


class AddendumChangeRow(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: uuid.UUID
    bid_folder: str
    field_name: str
    previous_value: str | None = None
    new_value: str | None = None
    addendum_number: int | None = None
    file_name: str | None = None
    page_number: int | None = None
