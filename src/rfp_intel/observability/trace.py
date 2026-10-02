"""Write one agent step to PostgreSQL and to the JSON log."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from rfp_intel.db.models import AgentStep
from rfp_intel.db.session import session_scope
from rfp_intel.db.store import insert_agent_step
from rfp_intel.logging_setup import get_logger

logger = get_logger("trace")

_sink: ContextVar[list | None] = ContextVar("trace_sink", default=None)
_run_id: ContextVar[str | None] = ContextVar("trace_run_id", default=None)


def bind_trace(run_id: str, sink: list | None = None):
    token_run = _run_id.set(run_id)
    token_sink = _sink.set(sink)
    return token_run, token_sink


def reset_trace(tokens) -> None:
    token_run, token_sink = tokens
    _run_id.reset(token_run)
    _sink.reset(token_sink)


def trace_step(agent: str, step_input: dict | None, step_output: dict | None, *, tokens: int = 0, latency_ms: int = 0) -> None:
    payload = {
        "agent": agent,
        "input": step_input,
        "output": step_output,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }
    logger.info("agent step", extra={"extra_data": payload})
    sink = _sink.get()
    if sink is not None:
        sink.append(payload)
    run_id = _run_id.get()
    if not run_id or sink is not None and run_id == "memory":
        return
    try:
        with session_scope() as conn:
            insert_agent_step(
                conn,
                AgentStep(
                    run_id=uuid.UUID(run_id),
                    agent=agent,
                    input_json=step_input,
                    output_json=step_output,
                    tokens=tokens,
                    latency_ms=latency_ms,
                ),
            )
    except Exception:
        logger.exception("could not persist agent step")
