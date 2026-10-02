"""Docling conversion for PDF and HTML bid files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rfp_intel.ingestion.sections import RawItem, SectionDraft, build_sections
from rfp_intel.logging_setup import get_logger

logger = get_logger("parser")

_MIN_TEXT_CHARS = 8
_converters: dict[bool, object] = {}


@dataclass
class ParsedFile:
    sections: list[SectionDraft]
    warnings: list[str] = field(default_factory=list)
    document: object | None = None


def page_has_extractable_text(text: str, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """A page with this much non-space text already has a text layer."""
    return sum(1 for char in text if not char.isspace()) >= min_chars


def text_page_ranges(flags: list[bool]) -> list[tuple[int, int, bool]]:
    """Group consecutive pages. Each tuple is a 1-based inclusive range and whether it has text."""
    if not flags:
        return []
    ranges: list[tuple[int, int, bool]] = []
    start = 1
    current = flags[0]
    for index, flag in enumerate(flags, start=1):
        if flag != current:
            ranges.append((start, index - 1, current))
            start = index
            current = flag
    ranges.append((start, len(flags), current))
    return ranges


def get_converter(*, ocr: bool = False):
    """Lazy converters. The OCR models load only when a page has no text layer."""
    cached = _converters.get(ocr)
    if cached is not None:
        return cached
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TableStructureOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.FAST,
    )
    if hasattr(pipeline_options, "force_backend_text"):
        pipeline_options.force_backend_text = not ocr
    _enable_heading_hierarchy(pipeline_options)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    _converters[ocr] = converter
    return converter


def _enable_heading_hierarchy(pipeline_options) -> None:
    """Turn on heading levels when this Docling build exposes the option."""
    try:
        from docling.datamodel.pipeline_options import HeadingHierarchyOptions

        if hasattr(pipeline_options, "heading_hierarchy_options"):
            pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
            return
    except Exception:
        logger.info("heading hierarchy options are not available in this Docling build")
    for attr in ("do_heading_hierarchy", "generate_heading_levels"):
        if hasattr(pipeline_options, attr):
            setattr(pipeline_options, attr, True)


def parse_file(path: Path) -> ParsedFile:
    if path.suffix.lower() != ".pdf":
        return _finish(path, [get_converter(ocr=False).convert(str(path))])
    flags = _text_flags(path)
    if flags is None:
        logger.info("pdf text layer unreadable; using OCR", extra={"extra_data": {"file": path.name}})
        return _finish(path, [get_converter(ocr=True).convert(str(path))])
    ranges = text_page_ranges(flags)
    if not ranges:
        return _finish(path, [get_converter(ocr=False).convert(str(path))])
    text_pages = sum(flags)
    logger.info(
        "pdf text layer",
        extra={"extra_data": {"file": path.name, "pages": len(flags), "text_pages": text_pages, "ocr_pages": len(flags) - text_pages}},
    )
    results = []
    for start, end, has_text in ranges:
        results.append(get_converter(ocr=not has_text).convert(str(path), page_range=(start, end)))
    return _finish(path, results)


def _text_flags(path: Path) -> list[bool] | None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return None
        return [page_has_extractable_text(page.extract_text() or "") for page in reader.pages]
    except Exception:
        logger.exception("could not read the pdf text layer", extra={"extra_data": {"file": path.name}})
        return None


def _finish(path: Path, results: list) -> ParsedFile:
    warnings: list[str] = []
    items: list[RawItem] = []
    for result in results:
        status = str(getattr(result, "status", ""))
        if status and "FAILURE" in status and "PARTIAL" not in status:
            raise RuntimeError(f"Docling failed to convert {path.name}: {status}")
        if status and "PARTIAL" in status:
            warnings.append(f"partial conversion: {status}")
        page_items, page_warnings = _raw_items(result.document)
        items.extend(page_items)
        warnings.extend(page_warnings)
    if len(results) > 1:
        warnings.append("text pages and scanned pages were parsed separately")
    sections = build_sections(items)
    if not any(section.body_text.strip() or section.tables for section in sections):
        warnings.append(f"no extractable text in {path.name}")
        logger.warning("empty document", extra={"extra_data": {"file": path.name, "warnings": warnings}})
    for warning in warnings:
        logger.warning(warning, extra={"extra_data": {"file": path.name}})
    document = results[0].document if len(results) == 1 else None
    return ParsedFile(sections=sections, warnings=warnings, document=document)


def _raw_items(document) -> tuple[list[RawItem], list[str]]:
    from docling_core.types.doc import SectionHeaderItem, TableItem, TextItem
    from docling_core.types.doc.labels import DocItemLabel

    items: list[RawItem] = []
    seen_pages: set[int] = set()
    for item, level in document.iterate_items():
        page = _page_of(item)
        if page is not None:
            seen_pages.add(page)
        if isinstance(item, TableItem):
            markdown, rows = _table_payload(item, document)
            items.append(
                RawItem(
                    kind="table",
                    text=markdown,
                    level=level,
                    page=page,
                    table_markdown=markdown,
                    table_rows=rows,
                    caption=_caption(item),
                )
            )
            continue
        text = (getattr(item, "text", None) or "").strip()
        label = getattr(item, "label", None)
        is_heading = isinstance(item, SectionHeaderItem) or label in {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
        if is_heading:
            items.append(RawItem(kind="heading", text=text, level=level or 1, page=page))
            continue
        if isinstance(item, TextItem) or text:
            items.append(RawItem(kind="text", text=text, level=level, page=page))

    warnings: list[str] = []
    pages = getattr(document, "pages", None) or {}
    for page_no in pages:
        number = int(page_no)
        if number not in seen_pages:
            warnings.append(f"page {number} had no body text after layout")
    return items, warnings


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def _caption(item) -> str | None:
    captions = getattr(item, "captions", None) or []
    texts = []
    for caption in captions:
        text = getattr(caption, "text", None)
        if text:
            texts.append(str(text).strip())
    return " ".join(texts) or None


def _table_payload(item, document) -> tuple[str, list]:
    try:
        frame = item.export_to_dataframe(doc=document)
        markdown = frame.to_markdown(index=False)
        rows = frame.fillna("").astype(str).to_dict(orient="records")
        return markdown, rows
    except Exception as exc:
        logger.warning("table export failed", extra={"extra_data": {"error": str(exc)}})
        text = getattr(item, "text", None) or ""
        return str(text), []
