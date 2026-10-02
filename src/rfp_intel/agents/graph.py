"""LangGraph orchestration for extraction and question answering."""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from rfp_intel.agents.extract import answer_from_hits, extract_all_groups, extract_fields_from_query, search_hits, summarize
from rfp_intel.agents.fields import ALL_FIELDS, FIELD_GROUPS, SPECIALIST_FIELDS, SUMMARY_FIELD
from rfp_intel.agents.reconcile import reconcile_candidates
from rfp_intel.agents.validate import judge
from rfp_intel.config import get_settings
from rfp_intel.observability.trace import trace_step
from rfp_intel.resilience import backoff_delay, sleep
from rfp_intel.schemas import AddendumChange, BidRecord, FieldValue, SearchHit, SourceCitation, ValidationSummary


class ExtractState(TypedDict, total=False):
    bid_folder: str
    candidates: list
    chosen: dict
    changes: list
    field_status: dict
    rejected: list
    retry_counts: dict
    record: dict


class AskState(TypedDict, total=False):
    question: str
    bid_ids: list
    hits: list
    answer: str
    citations: list


def build_extract_graph():
    graph = StateGraph(ExtractState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_extract", retrieve_extract_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve_extract")
    graph.add_edge("retrieve_extract", "reconcile")
    graph.add_edge("reconcile", "validate")
    graph.add_conditional_edges("validate", route_after_validation, {"retry": "retry", "report": "report"})
    graph.add_edge("retry", "validate")
    graph.add_edge("report", END)
    return graph.compile()


def build_ask_graph():
    graph = StateGraph(AskState)
    graph.add_node("retrieve", ask_retrieve_node)
    graph.add_node("answer", ask_answer_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def plan_node(state: ExtractState) -> dict:
    started = time.perf_counter()
    plan = [{"group": name, "fields": fields} for name, fields in FIELD_GROUPS.items()]
    trace_step(
        "orchestrator",
        {"bid_id": state["bid_folder"], "mode": "extract"},
        {"plan": plan},
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return {"retry_counts": {}, "candidates": []}


def retrieve_extract_node(state: ExtractState) -> dict:
    return {"candidates": extract_all_groups(state["bid_folder"])}


def reconcile_node(state: ExtractState) -> dict:
    started = time.perf_counter()
    chosen, changes = reconcile_candidates(state.get("candidates") or [], SPECIALIST_FIELDS)
    trace_step(
        "addendum",
        {"candidate_count": len(state.get("candidates") or [])},
        {"changes": changes},
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return {"chosen": chosen, "changes": changes}


def validate_node(state: ExtractState) -> dict:
    started = time.perf_counter()
    settings = get_settings()
    counts = state.get("retry_counts") or {}
    status: dict[str, str] = {}
    rejected: list[str] = []
    for field in SPECIALIST_FIELDS:
        winner = (state.get("chosen") or {}).get(field)
        if not winner:
            status[field] = "not_found"
            continue
        outcome = judge(field, winner.get("value"), winner.get("chunk_text"))
        status[field] = outcome
        if outcome == "fail" and counts.get(field, 0) < settings.validator_max_retries:
            rejected.append(field)
    trace_step(
        "validator",
        {"retry_counts": counts},
        {"status": status, "rejected": rejected},
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return {"field_status": status, "rejected": rejected}


def route_after_validation(state: ExtractState) -> str:
    if state.get("rejected"):
        return "retry"
    return "report"


def retry_node(state: ExtractState) -> dict:
    settings = get_settings()
    counts = dict(state.get("retry_counts") or {})
    rejected = list(state.get("rejected") or [])
    attempt = max((counts.get(field, 0) for field in rejected), default=0)
    delay = backoff_delay(attempt, settings.backoff_base_seconds, settings.backoff_cap_seconds)
    trace_step("validator_backoff", {"attempt": attempt, "fields": rejected}, {"delay_seconds": round(delay, 3)})
    sleep(delay)
    kept = [candidate for candidate in (state.get("candidates") or []) if candidate.get("field_name") not in rejected]
    for field in rejected:
        counts[field] = counts.get(field, 0) + 1
        reason = (state.get("field_status") or {}).get(field, "unsupported")
        kept.extend(
            extract_fields_from_query(
                [field],
                f"{field}. Previous attempt was rejected: {reason}. Quote the value from the document.",
                state["bid_folder"],
            )
        )
    chosen, changes = reconcile_candidates(kept, SPECIALIST_FIELDS)
    return {"candidates": kept, "chosen": chosen, "changes": changes, "retry_counts": counts}


def report_node(state: ExtractState) -> dict:
    chosen = state.get("chosen") or {}
    status = state.get("field_status") or {}
    summary, used = summarize(state["bid_folder"], chosen)
    fields: dict[str, FieldValue] = {}
    passed = failed = not_found = 0
    for field in SPECIALIST_FIELDS:
        winner = chosen.get(field)
        outcome = status.get(field, "not_found")
        if winner and outcome == "pass":
            fields[field] = FieldValue(
                value=winner.get("value"),
                sources=[
                    SourceCitation(
                        file=winner.get("file_name") or "",
                        page=winner.get("page"),
                        section=winner.get("section"),
                    )
                ],
                confidence=float(winner.get("confidence") or 0),
                notes=winner.get("notes") or "",
            )
            passed += 1
            continue
        if winner and outcome == "fail":
            fields[field] = FieldValue(
                value=None,
                sources=[],
                confidence=0.0,
                notes="Not found in documents",
            )
            failed += 1
            continue
        fields[field] = FieldValue(value=None, notes="Not found in documents")
        not_found += 1

    if summary and used:
        fields[SUMMARY_FIELD] = FieldValue(
            value=summary,
            sources=[
                SourceCitation(file=item.get("file") or "", page=item.get("page"), section=item.get("section"))
                for item in used
            ],
            confidence=0.7,
            notes="Generated from cited chunks",
        )
        passed += 1
    else:
        fields[SUMMARY_FIELD] = FieldValue(value=None, notes="Not found in documents")
        not_found += 1

    record = BidRecord(
        bid_id=state["bid_folder"],
        fields={name: fields[name] for name in ALL_FIELDS},
        addendum_changes=[AddendumChange.model_validate(change) for change in (state.get("changes") or [])],
        validation=ValidationSummary(passed=passed, failed=failed, not_found=not_found),
    )
    return {"record": record.model_dump()}


def ask_retrieve_node(state: AskState) -> dict:
    started = time.perf_counter()
    question = state["question"]
    bid_ids = state.get("bid_ids") or [None]
    trace_step("orchestrator", {"mode": "ask", "question": question, "bid_ids": bid_ids}, {"steps": ["retrieve", "answer"]})
    hits: list[SearchHit] = []
    for bid_id in bid_ids:
        hits.extend(search_hits(question, bid_id=bid_id, top_k=6))
    trace_step(
        "retrieval",
        {"question": question, "bid_ids": bid_ids},
        {"hit_count": len(hits), "files": [hit.file_name for hit in hits]},
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return {"hits": [hit.model_dump() for hit in hits]}


def ask_answer_node(state: AskState) -> dict:
    started = time.perf_counter()
    hits = [SearchHit.model_validate(hit) for hit in (state.get("hits") or [])]
    answer, citations, tokens = answer_from_hits(state["question"], hits)
    trace_step(
        "qa",
        {"question": state["question"]},
        {"answer": answer, "citations": citations},
        tokens=tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return {"answer": answer, "citations": citations}
