"""Turn a flat reading-order stream into sections and tables.

Docling supplies the stream. This module does not import Docling, so the
section rules can be tested without the layout models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RawItem:
    kind: Literal["heading", "text", "table"]
    text: str
    level: int = 1
    page: int | None = None
    table_markdown: str | None = None
    table_rows: list | None = None
    caption: str | None = None


@dataclass
class TableDraft:
    page_number: int | None
    markdown: str
    rows: list
    caption: str | None = None


@dataclass
class SectionDraft:
    heading_path: list[str]
    level: int
    ordinal: int
    page_start: int | None
    page_end: int | None
    body_text: str = ""
    tables: list[TableDraft] = field(default_factory=list)


def _plain_table(rows: list) -> str:
    """One line per row, cells separated by spaces. Empty when there is nothing to read."""
    if not rows or not isinstance(rows[0], dict):
        return ""
    headers = [str(header).strip() for header in rows[0].keys()]
    lines: list[str] = []
    if any(headers):
        lines.append(" ".join(header for header in headers if header))
    for row in rows:
        cells = [str(row.get(header, "")).strip() for header in rows[0].keys()]
        line = " ".join(cell for cell in cells if cell)
        if line:
            lines.append(line)
    return "\n".join(lines)


def build_sections(items: list[RawItem]) -> list[SectionDraft]:
    """Group items under the heading that is in scope. Tables stay on that section."""
    sections: list[SectionDraft] = []
    stack: list[tuple[int, str]] = []

    def path() -> list[str]:
        return [title for _, title in stack] or ["Document"]

    def touch(page: int | None) -> SectionDraft:
        current = path()
        if sections and sections[-1].heading_path == current:
            section = sections[-1]
        else:
            section = SectionDraft(
                heading_path=list(current),
                level=stack[-1][0] if stack else 0,
                ordinal=len(sections),
                page_start=page,
                page_end=page,
            )
            sections.append(section)
        if page is not None:
            if section.page_start is None:
                section.page_start = page
            section.page_end = page
        return section

    for item in items:
        if item.kind == "heading":
            title = (item.text or "").strip() or "Section"
            level = item.level if item.level > 0 else 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            section = SectionDraft(
                heading_path=path(),
                level=level,
                ordinal=len(sections),
                page_start=item.page,
                page_end=item.page,
            )
            sections.append(section)
            continue
        if item.kind == "text":
            text = (item.text or "").strip()
            if not text:
                continue
            section = touch(item.page)
            section.body_text = f"{section.body_text}\n\n{text}".strip()
            continue
        section = touch(item.page)
        plain = _plain_table(item.table_rows or [])
        if plain:
            section.body_text = f"{section.body_text}\n\n{plain}".strip()
            continue
        markdown = (item.table_markdown or item.text or "").strip()
        if not markdown:
            continue
        section.tables.append(
            TableDraft(
                page_number=item.page,
                markdown=markdown,
                rows=list(item.table_rows or []),
                caption=item.caption,
            )
        )

    if not sections:
        sections.append(SectionDraft(heading_path=["Document"], level=0, ordinal=0, page_start=None, page_end=None))
    return sections
