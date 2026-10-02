"""Section-bounded chunks.

The indexing pipeline prefers Docling's HybridChunker. ``chunk_sections`` is the
same contract (heading path, page, no cross-section splits, header repeated on
table row splits) and is what the unit tests exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfp_intel.ingestion.sections import SectionDraft


@dataclass
class ChunkDraft:
    text: str
    heading_path: list[str]
    page_number: int | None
    token_count: int
    ordinal: int = 0
    section_ordinal: int | None = None


def _tokens(text: str) -> list[str]:
    return text.split()


def chunk_sections(sections: list[SectionDraft], max_tokens: int = 512) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for section in sections:
        heading = " > ".join(section.heading_path)
        prefix = f"{heading}\n\n" if heading else ""
        prose_parts = []
        if section.body_text.strip() and not section.tables:
            prose_parts.append(section.body_text.strip())
        elif section.body_text.strip() and section.tables:
            # Tables are emitted as their own chunks below. Keep non-table prose
            # by dropping the markdown tables that were appended to body_text.
            prose = section.body_text
            for table in section.tables:
                if table.markdown:
                    prose = prose.replace(table.markdown, " ")
            prose = " ".join(prose.split())
            if prose:
                prose_parts.append(prose)
        for prose in prose_parts:
            words = _tokens(prose)
            if not words:
                continue
            for start in range(0, len(words), max_tokens):
                piece = " ".join(words[start : start + max_tokens])
                text = f"{prefix}{piece}".strip()
                drafts.append(
                    ChunkDraft(
                        text=text,
                        heading_path=list(section.heading_path),
                        page_number=section.page_start,
                        token_count=len(_tokens(text)),
                        section_ordinal=section.ordinal,
                    )
                )
        for table in section.tables:
            drafts.extend(_chunk_table(section, table.markdown, prefix, max_tokens))
    for index, draft in enumerate(drafts):
        draft.ordinal = index
    return drafts


def _chunk_table(section: SectionDraft, markdown: str, prefix: str, max_tokens: int) -> list[ChunkDraft]:
    lines = [line for line in (markdown or "").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0]
    separator = lines[1] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= {"-", ":", " "} else ""
    body = lines[2:] if separator else lines[1:]
    rows = body or ([header] if not separator else [])
    if not rows:
        text = f"{prefix}{markdown}".strip()
        return [
            ChunkDraft(
                text=text,
                heading_path=list(section.heading_path),
                page_number=section.page_start,
                token_count=len(_tokens(text)),
                section_ordinal=section.ordinal,
            )
        ]
    chunks: list[ChunkDraft] = []
    bucket: list[str] = []
    bucket_tokens = len(_tokens(prefix)) + len(_tokens(header))

    def flush() -> None:
        if not bucket:
            return
        table_text = "\n".join([header, separator, *bucket] if separator else [header, *bucket])
        text = f"{prefix}{table_text}".strip()
        chunks.append(
            ChunkDraft(
                text=text,
                heading_path=list(section.heading_path),
                page_number=section.page_start,
                token_count=len(_tokens(text)),
                section_ordinal=section.ordinal,
            )
        )

    for row in rows:
        row_tokens = len(_tokens(row))
        if bucket and bucket_tokens + row_tokens > max_tokens:
            flush()
            bucket = []
            bucket_tokens = len(_tokens(prefix)) + len(_tokens(header))
        bucket.append(row)
        bucket_tokens += row_tokens
    flush()
    return chunks


def chunks_from_docling(document, *, max_tokens: int, embedding_model: str) -> list[ChunkDraft]:
    """Use Docling HybridChunker. Headings stay attached and tables repeat their header."""
    from docling.chunking import HybridChunker

    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model),
            max_tokens=max_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, repeat_table_header=True)
    except Exception:
        chunker = HybridChunker(merge_peers=True, repeat_table_header=True)

    drafts: list[ChunkDraft] = []
    for ordinal, chunk in enumerate(chunker.chunk(document)):
        meta = getattr(chunk, "meta", None)
        headings = list(getattr(meta, "headings", None) or []) or ["Document"]
        page = _page_from_meta(meta)
        text = (getattr(chunk, "text", None) or "").strip()
        if not text:
            continue
        heading = " > ".join(headings)
        if heading and not text.startswith(heading):
            text = f"{heading}\n\n{text}"
        drafts.append(
            ChunkDraft(
                text=text,
                heading_path=headings,
                page_number=page,
                token_count=len(text.split()),
                ordinal=ordinal,
            )
        )
    return drafts


def _page_from_meta(meta) -> int | None:
    if meta is None:
        return None
    for item in getattr(meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                return int(page_no)
    return None
