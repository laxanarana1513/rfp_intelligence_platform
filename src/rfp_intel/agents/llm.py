"""Gemini chat model with structured output and exponential backoff."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import TypeVar

from pydantic import BaseModel

from rfp_intel.config import get_settings
from rfp_intel.resilience import NonRetryableError, retry_with_backoff

T = TypeVar("T", bound=BaseModel)

_llm_override: ContextVar = ContextVar("llm_override", default=None)
_model = None


def set_llm_override(fn):
    return _llm_override.set(fn)


def reset_llm_override(token) -> None:
    _llm_override.reset(token)


def complete_structured(schema: type[T], system: str, user: str) -> tuple[T, int]:
    override = _llm_override.get()
    if override is not None:
        parsed = override(schema, system, user)
        if not isinstance(parsed, schema):
            parsed = schema.model_validate(parsed)
        return parsed, 0

    settings = get_settings()
    if not settings.google_api_key:
        raise NonRetryableError("GOOGLE_API_KEY is not set")
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    def call():
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = _chat_model()
        structured = llm.with_structured_output(schema, include_raw=True)
        return structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    result = retry_with_backoff(
        call,
        max_attempts=settings.io_max_attempts,
        base=settings.backoff_base_seconds,
        cap=settings.backoff_cap_seconds,
    )
    parsed, tokens = _unpack(result)
    if parsed is None:
        parsed, repair_tokens = _repair(schema, system, user, result)
        tokens += repair_tokens
    if not isinstance(parsed, schema):
        parsed = schema.model_validate(parsed)
    return parsed, tokens


def _chat_model():
    global _model
    if _model is not None:
        return _model
    from langchain.chat_models import init_chat_model

    settings = get_settings()
    kwargs = {"api_key": settings.google_api_key}
    if settings.llm_thinking_level:
        kwargs["thinking_level"] = settings.llm_thinking_level
    try:
        _model = init_chat_model(settings.llm_model, model_provider="google_genai", **kwargs)
    except TypeError:
        kwargs.pop("thinking_level", None)
        _model = init_chat_model(settings.llm_model, model_provider="google_genai", **kwargs)
    return _model


def _unpack(result) -> tuple[object | None, int]:
    if isinstance(result, dict):
        raw = result.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        total = usage.get("total_tokens") if isinstance(usage, dict) else getattr(usage, "total_tokens", 0)
        return result.get("parsed"), int(total or 0)
    return result, 0


def _repair(schema, system: str, user: str, previous) -> tuple[object, int]:
    from langchain_core.messages import HumanMessage, SystemMessage

    settings = get_settings()
    raw_text = ""
    if isinstance(previous, dict) and previous.get("raw") is not None:
        raw_text = str(getattr(previous["raw"], "content", previous["raw"]))[:4000]

    def call():
        llm = _chat_model()
        structured = llm.with_structured_output(schema)
        return structured.invoke(
            [
                SystemMessage(content=system + " Return only the structured object. Do not add fields."),
                HumanMessage(content=f"{user}\n\nThe previous response was invalid:\n{raw_text}"),
            ]
        )

    try:
        parsed = retry_with_backoff(
            call,
            max_attempts=1,
            base=settings.backoff_base_seconds,
            cap=settings.backoff_cap_seconds,
        )
    except Exception as exc:
        raise NonRetryableError(f"model returned invalid structured output: {exc}") from exc
    if parsed is None:
        raise NonRetryableError("model returned invalid structured output")
    return parsed, 0
