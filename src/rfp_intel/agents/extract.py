"""Retrieval and section-wise extraction used by the LangGraph nodes."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context

from rfp_intel.agents.fields import FIELD_GROUPS
from rfp_intel.agents.llm import complete_structured
from rfp_intel.agents.schemas import AnswerResult, SectionExtraction, SummaryResult
from rfp_intel.config import get_settings
from rfp_intel.observability.trace import trace_step
from rfp_intel.schemas import SearchHit

_search_override: ContextVar = ContextVar("search_override", default=None)

EXTRACT_SYSTEM = """You extract structured bid fields from the evidence chunks of a single section.
Rules:
- Use only the chunks provided. Do not use outside knowledge.
- Fill only the requested fields.
- If a field is not stated in these chunks, set value to null, chunk_id to null, and notes to "Not found in documents".
- Every non-null value must set chunk_id to one of the chunk ids provided.
- Copy the fact from the chunk. Do not invent dates, amounts, model numbers, or contacts.
- confidence is between 0 and 1.
"""

SUMMARY_SYSTEM = """Write a 3 to 6 sentence summary of this bid using only the evidence.
Every sentence must be supported by the chunks. Return the chunk ids you used.
If the evidence is empty, set summary to "Not found in documents" and chunk_ids to an empty list.
"""

ANSWER_SYSTEM = """Answer the question using only the evidence chunks.
Cite the file and page for each fact you use.
If the evidence does not contain the answer, reply exactly: Not found in documents.
Do not guess.
"""


def set_search_override(fn):
    return _search_override.set(fn)


def reset_search_override(token) -> None:
    _search_override.reset(token)


def search_hits(query: str, *, bid_id: str | None, top_k: int = 8, doc_type: str | None = None) -> list[SearchHit]:
    override = _search_override.get()
    if override is not None:
        return override(query, bid_id=bid_id, top_k=top_k, doc_type=doc_type)
    from rfp_intel.db.session import session_scope
    from rfp_intel.search.hybrid import search

    with session_scope() as conn:
        return search(conn, query, bid_id=bid_id, doc_type=doc_type, top_k=top_k, mode="hybrid_rerank")


def extract_all_groups(bid_folder: str) -> list[dict]:
    started = time.perf_counter()
    from rfp_intel.agents import llm as llm_mod
    from rfp_intel.observability import trace as trace_mod

    search_fn = _search_override.get()
    llm_fn = llm_mod._llm_override.get()
    run_id = trace_mod._run_id.get()
    sink = trace_mod._sink.get()

    def run_group(name: str, fields: list[str]) -> list[dict]:
        """Re-bind request context inside the worker. LangGraph has already entered the parent context."""
        held = []
        if search_fn is not None:
            held.append(("search", set_search_override(search_fn)))
        if llm_fn is not None:
            held.append(("llm", llm_mod.set_llm_override(llm_fn)))
        if run_id is not None:
            held.append(("trace", trace_mod.bind_trace(run_id, sink)))
        try:
            return extract_group(name, fields, bid_folder)
        finally:
            for kind, token in reversed(held):
                if kind == "search":
                    reset_search_override(token)
                elif kind == "llm":
                    llm_mod.reset_llm_override(token)
                else:
                    trace_mod.reset_trace(token)

    candidates: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(FIELD_GROUPS)) as pool:
        futures = [pool.submit(run_group, name, fields) for name, fields in FIELD_GROUPS.items()]
        for future in as_completed(futures):
            candidates.extend(future.result())
    trace_step(
        "extraction",
        {"bid_id": bid_folder, "groups": list(FIELD_GROUPS)},
        {"candidate_count": len(candidates)},
        latency_ms=_ms(started),
    )
    return candidates


def extract_group(group_name: str, fields: list[str], bid_folder: str) -> list[dict]:
    started = time.perf_counter()
    hits = search_hits(" ".join(fields), bid_id=bid_folder, top_k=8)
    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.section_id or hit.section_heading or "document"].append(hit)
    sections = sorted(grouped.values(), key=lambda group: max(hit.score for hit in group), reverse=True)[:4]
    found: list[dict] = []
    if sections:
        workers = max(1, min(get_settings().extract_section_concurrency, len(sections)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # One context per task. A single Context cannot be entered by two threads.
            futures = [
                pool.submit(copy_context().run, extract_section, fields, section_hits)
                for section_hits in sections
            ]
            for future in as_completed(futures):
                found.extend(future.result())
    trace_step(
        group_name,
        {"fields": fields, "hit_count": len(hits), "sections": len(sections)},
        {"candidate_count": len(found)},
        latency_ms=_ms(started),
    )
    return found


def extract_fields_from_query(fields: list[str], query: str, bid_folder: str) -> list[dict]:
    hits = search_hits(query, bid_id=bid_folder, top_k=6)
    if not hits:
        return []
    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.section_id or hit.section_heading or "document"].append(hit)
    found: list[dict] = []
    for section_hits in list(grouped.values())[:3]:
        found.extend(extract_section(fields, section_hits))
    return found


def extract_section(fields: list[str], hits: list[SearchHit]) -> list[dict]:
    payload = [
        {
            "chunk_id": hit.chunk_id,
            "file": hit.file_name,
            "page": hit.page_number,
            "section": hit.section_heading,
            "text": hit.text,
        }
        for hit in hits
    ]
    user = f"Fields: {fields}\n\nChunks:\n{payload}"
    try:
        parsed, tokens = complete_structured(SectionExtraction, EXTRACT_SYSTEM, user)
    except Exception as exc:
        trace_step("section_extract_error", {"fields": fields}, {"error": str(exc)})
        return []
    by_id = {hit.chunk_id: hit for hit in hits}
    candidates = []
    for item in parsed.fields:
        if item.field_name not in fields or not item.value or not str(item.value).strip():
            continue
        hit = by_id.get(item.chunk_id or "")
        if hit is None:
            continue
        candidates.append(
            {
                "field_name": item.field_name,
                "value": str(item.value).strip(),
                "confidence": float(item.confidence or 0),
                "notes": item.notes or "",
                "chunk_id": hit.chunk_id,
                "file_name": hit.file_name,
                "page": hit.page_number,
                "section": hit.section_heading,
                "addendum_number": hit.addendum_number or 0,
                "doc_type": hit.doc_type,
                "chunk_text": hit.text,
            }
        )
    trace_step(
        "section_extract",
        {"fields": fields, "chunk_ids": list(by_id)},
        {"fields": [candidate["field_name"] for candidate in candidates]},
        tokens=tokens,
    )
    return candidates


def summarize(bid_folder: str, chosen: dict[str, dict | None]) -> tuple[str | None, list[dict]]:
    evidence = []
    seen = set()
    for field, winner in chosen.items():
        if not winner or not winner.get("chunk_text"):
            continue
        if winner["chunk_id"] in seen:
            continue
        seen.add(winner["chunk_id"])
        evidence.append(
            {
                "chunk_id": winner["chunk_id"],
                "field": field,
                "file": winner.get("file_name"),
                "page": winner.get("page"),
                "section": winner.get("section"),
                "text": winner.get("chunk_text"),
            }
        )
    if not evidence:
        return None, []
    parsed, tokens = complete_structured(
        SummaryResult,
        SUMMARY_SYSTEM,
        f"Bid: {bid_folder}\n\nEvidence:\n{evidence}",
    )
    allowed = {item["chunk_id"]: item for item in evidence}
    used = [allowed[chunk_id] for chunk_id in parsed.chunk_ids if chunk_id in allowed]
    summary = (parsed.summary or "").strip()
    trace_step("report", {"bid_id": bid_folder}, {"summary": summary, "chunks": [item["chunk_id"] for item in used]}, tokens=tokens)
    if not summary or summary.lower() == "not found in documents" or not used:
        return None, []
    return summary, used


def answer_from_hits(question: str, hits: list[SearchHit]) -> tuple[str, list[dict], int]:
    if not hits:
        return "Not found in documents", [], 0
    payload = [
        {
            "file": hit.file_name,
            "page": hit.page_number,
            "section": hit.section_heading,
            "bid_id": hit.bid_id,
            "text": hit.text,
        }
        for hit in hits
    ]
    parsed, tokens = complete_structured(AnswerResult, ANSWER_SYSTEM, f"Question: {question}\n\nChunks:\n{payload}")
    answer = (parsed.answer or "").strip() or "Not found in documents"
    allowed = {(hit.file_name, hit.page_number) for hit in hits}
    citations = []
    for source in parsed.citations:
        if (source.file, source.page) in allowed or any(source.file == hit.file_name for hit in hits):
            citations.append({"file": source.file, "page": source.page, "section": source.section})
    if answer != "Not found in documents" and not citations:
        answer = "Not found in documents"
    return answer, citations, tokens


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
