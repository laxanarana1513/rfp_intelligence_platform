"""Qdrant collection for chunk vectors. Text lives in PostgreSQL."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rfp_intel.config import get_settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=get_settings().qdrant_url)
    return _client


def reset_client() -> None:
    global _client
    _client = None


def ensure_collection() -> None:
    settings = get_settings()
    client = get_client()
    existing = {collection.name for collection in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(size=settings.embedding_dim, distance=qmodels.Distance.COSINE),
        )
        for field_name in ("bid_id", "document_id", "doc_type", "addendum_number"):
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD if field_name != "addendum_number" else qmodels.PayloadSchemaType.INTEGER,
            )


def point_count() -> int:
    """How many vectors are stored. Zero means retrieval is not ready."""
    settings = get_settings()
    client = get_client()
    names = {collection.name for collection in client.get_collections().collections}
    if settings.qdrant_collection not in names:
        return 0
    counted = client.count(collection_name=settings.qdrant_collection, exact=True)
    return int(counted.count)


def upsert_points(points: list[qmodels.PointStruct]) -> None:
    if not points:
        return
    ensure_collection()
    settings = get_settings()
    client = get_client()
    client.upsert(collection_name=settings.qdrant_collection, points=points)


def delete_by_document(document_id: str) -> None:
    _delete(key="document_id", value=document_id)


def delete_by_bid(bid_id: str) -> None:
    _delete(key="bid_id", value=bid_id)


def _delete(key: str, value: str) -> None:
    settings = get_settings()
    client = get_client()
    existing = {collection.name for collection in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        return
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(must=[qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))])
        ),
    )


def search_vectors(
    vector: list[float],
    *,
    limit: int,
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum_number: int | None = None,
) -> list[tuple[str, float]]:
    ensure_collection()
    settings = get_settings()
    must: list[qmodels.FieldCondition] = []
    if bid_id:
        must.append(qmodels.FieldCondition(key="bid_id", match=qmodels.MatchValue(value=bid_id)))
    if doc_type:
        must.append(qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type)))
    if addendum_number is not None:
        must.append(qmodels.FieldCondition(key="addendum_number", match=qmodels.MatchValue(value=addendum_number)))
    query_filter = qmodels.Filter(must=must) if must else None
    response = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=query_filter,
        limit=limit,
    )
    hits = []
    for point in response.points:
        hits.append((str(point.id), float(point.score or 0.0)))
    return hits
