"""Structured outputs exchanged between agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedItem(BaseModel):
    field_name: str
    value: str | None = None
    confidence: float = 0.0
    notes: str = ""
    chunk_id: str | None = None


class SectionExtraction(BaseModel):
    fields: list[ExtractedItem] = Field(default_factory=list)


class SummaryResult(BaseModel):
    summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class CitedSource(BaseModel):
    file: str
    page: int | None = None
    section: str | None = None


class AnswerResult(BaseModel):
    answer: str = ""
    citations: list[CitedSource] = Field(default_factory=list)
