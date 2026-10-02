"""Hybrid search: Qdrant vectors, PostgreSQL full text, fusion, optional rerank."""

from __future__ import annotations

import uuid

from rfp_intel.config import get_settings
from rfp_intel.db.models import Chunk
from rfp_intel.db.session import Conn
from rfp_intel.db.store import chunks_by_ids
from rfp_intel.resilience import retry_with_backoff
from rfp_intel.schemas import SearchHit
from rfp_intel.search.embeddings import embed_query
from rfp_intel.search.expand import expand_query
from rfp_intel.search.fusion import reciprocal_rank_fusion
from rfp_intel.search.qdrant_store import search_vectors
from rfp_intel.search.rerank import rerank


def search(
    conn: Conn,
    query: str,
    *,
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
) -> list[SearchHit]:
    """``mode`` is ``vector``, ``hybrid``, or ``hybrid_rerank``."""
    settings = get_settings()
    expanded = expand_query(query)
    candidate_k = settings.search_candidate_k
    vector_ids: list[str] = []
    if mode in {"vector", "hybrid", "hybrid_rerank"}:
        vector = retry_with_backoff(
            lambda: embed_query(expanded),
            max_attempts=settings.io_max_attempts,
            base=settings.backoff_base_seconds,
            cap=settings.backoff_cap_seconds,
        )
        vector_hits = retry_with_backoff(
            lambda: search_vectors(
                vector,
                limit=candidate_k,
                bid_id=bid_id,
                doc_type=doc_type,
                addendum_number=addendum_number,
            ),
            max_attempts=settings.io_max_attempts,
            base=settings.backoff_base_seconds,
            cap=settings.backoff_cap_seconds,
        )
        vector_ids = [chunk_id for chunk_id, _score in vector_hits]

    keyword_ids: list[str] = []
    if mode in {"hybrid", "hybrid_rerank"}:
        keyword_ids = _keyword_ids(
            conn,
            expanded,
            limit=candidate_k,
            bid_id=bid_id,
            doc_type=doc_type,
            addendum_number=addendum_number,
        )

    if mode == "vector":
        ranked = [(chunk_id, float(len(vector_ids) - index)) for index, chunk_id in enumerate(vector_ids)]
    else:
        lists = [ids for ids in (vector_ids, keyword_ids) if ids]
        ranked = reciprocal_rank_fusion(lists, k=settings.rrf_k) if lists else []

    if not ranked:
        return []

    by_id = _load_chunks(conn, [chunk_id for chunk_id, _score in ranked])
    ordered = [(chunk_id, score) for chunk_id, score in ranked if chunk_id in by_id]
    if mode == "hybrid_rerank" and ordered:
        texts = [by_id[chunk_id].text for chunk_id, _score in ordered]
        scores = rerank(query, texts)
        ordered = sorted(zip((chunk_id for chunk_id, _s in ordered), scores), key=lambda item: item[1], reverse=True)

    hits: list[SearchHit] = []
    for chunk_id, score in ordered[:top_k]:
        row = by_id[chunk_id]
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                text=row.text,
                score=float(score),
                file_name=row.file_name,
                page_number=row.page_number,
                section_heading=" > ".join(row.heading_path or []),
                section_id=str(row.section_id) if row.section_id else None,
                bid_id=row.bid_folder,
                doc_type=row.doc_type,
                addendum_number=row.addendum_number,
            )
        )
    return hits


def _keyword_ids(
    conn: Conn,
    query: str,
    *,
    limit: int,
    bid_id: str | None,
    doc_type: str | None,
    addendum_number: int | None,
) -> list[str]:
    rows = conn.execute(
        """
        SELECT id::text AS id
        FROM chunks
        WHERE tsv @@ plainto_tsquery('english', %(query)s)
          AND (%(bid_id)s::text IS NULL OR bid_folder = %(bid_id)s::text)
          AND (%(doc_type)s::text IS NULL OR doc_type = %(doc_type)s::text)
          AND (%(addendum_number)s::integer IS NULL OR addendum_number = %(addendum_number)s::integer)
        ORDER BY ts_rank(tsv, plainto_tsquery('english', %(query)s)) DESC
        LIMIT %(limit)s
        """,
        {
            "query": query,
            "bid_id": bid_id,
            "doc_type": doc_type,
            "addendum_number": addendum_number,
            "limit": limit,
        },
    ).fetchall()
    return [row["id"] for row in rows]


def _load_chunks(conn: Conn, ids: list[str]) -> dict[str, Chunk]:
    uuids = []
    for chunk_id in ids:
        try:
            uuids.append(uuid.UUID(chunk_id))
        except ValueError:
            continue
    return {str(row.id): row for row in chunks_by_ids(conn, uuids)}
