"""Pydantic contracts shared by the API, agents, and search engine."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SourceCitation(BaseModel):
    file: str
    page: int | None = None
    section: str | None = None


class FieldValue(BaseModel):
    value: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""

    @model_validator(mode="after")
    def require_citation_for_value(self) -> FieldValue:
        has_value = self.value is not None and self.value.strip() != ""
        if has_value and not self.sources:
            raise ValueError("A field value requires at least one source citation")
        if not has_value:
            self.value = None
            if not self.notes:
                self.notes = "Not found in documents"
        return self


class AddendumChange(BaseModel):
    field: str
    previous_value: str | None = None
    new_value: str | None = None
    addendum_number: int | None = None
    file: str | None = None
    page: int | None = None


class ValidationSummary(BaseModel):
    passed: int = 0
    failed: int = 0
    not_found: int = 0


class BidRecord(BaseModel):
    bid_id: str
    fields: dict[str, FieldValue]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationSummary = Field(default_factory=ValidationSummary)


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    file_name: str
    page_number: int | None = None
    section_heading: str = ""
    section_id: str | None = None
    bid_id: str
    doc_type: str
    addendum_number: int | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    bid_id: str | None = None
