"""OpenRouter chat model with structured output and exponential backoff."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from typing import TypeVar

import requests
from pydantic import BaseModel

from rfp_intel.config import get_settings
from rfp_intel.resilience import NonRetryableError, retry_with_backoff

T = TypeVar("T", bound=BaseModel)

_llm_override: ContextVar = ContextVar("llm_override", default=None)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
    if not settings.openrouter_api_key:
        raise NonRetryableError("OPENROUTER_API_KEY is not set")

    def call():
        return _openrouter(
            schema,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    parsed, tokens, raw_text = retry_with_backoff(
        call,
        max_attempts=min(settings.io_max_attempts, 3),
        base=settings.backoff_base_seconds,
        cap=settings.backoff_cap_seconds,
    )
    if parsed is None:
        parsed, repair_tokens = _repair(schema, system, user, raw_text)
        tokens += repair_tokens
    if not isinstance(parsed, schema):
        parsed = schema.model_validate(parsed)
    return parsed, tokens


def _openrouter(schema: type[BaseModel], messages: list[dict]) -> tuple[BaseModel | None, int, str]:
    settings = get_settings()
    payload: dict = {
        "model": settings.llm_model,
        "messages": messages,
    }
    if settings.llm_reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": False,
            "schema": schema.model_json_schema(),
        },
    }
    try:
        body = _post(payload)
    except RuntimeError as exc:
        if getattr(exc, "status_code", None) != 400:
            raise
        payload.pop("response_format", None)
        body = _post(payload)
    message = body["choices"][0]["message"]
    text = _message_text(message)
    usage = body.get("usage") or {}
    return _parse_model(schema, text), int(usage.get("total_tokens") or 0), text


def _post(payload: dict) -> dict:
    settings = get_settings()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.llm_request_timeout_seconds,
        )
    except requests.Timeout as exc:
        error = RuntimeError("OpenRouter request timed out")
        error.status_code = 504
        raise error from exc
    if response.status_code >= 400:
        if response.status_code in {401, 403}:
            raise NonRetryableError(f"OpenRouter rejected the API key ({response.status_code})")
        error = RuntimeError(response.text[:500])
        error.status_code = response.status_code
        raise error
    return response.json()


def _message_text(message: dict) -> str:
    content = message.get("content")
    text = _content_to_text(content)
    if text.strip():
        return text
    details = message.get("reasoning_details") or message.get("reasoning")
    if isinstance(details, list):
        parts = []
        for item in details:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        text = "".join(parts)
    elif isinstance(details, str):
        text = details
    else:
        text = ""
    return text.strip()


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return ""


def _parse_model(schema: type[T], text: str) -> T | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    for candidate in (cleaned, _json_object(cleaned)):
        if not candidate:
            continue
        try:
            return schema.model_validate_json(candidate)
        except Exception:
            try:
                return schema.model_validate(json.loads(candidate))
            except Exception:
                continue
    return None


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return ""


def _repair(schema, system: str, user: str, raw_text: str) -> tuple[object, int]:
    settings = get_settings()

    def call():
        parsed, _tokens, _text = _openrouter(
            schema,
            [
                {
                    "role": "system",
                    "content": system
                    + " Return only valid JSON for the schema. Use chunk_id exactly as given in the chunks list.",
                },
                {"role": "user", "content": f"{user}\n\nThe previous response was invalid:\n{raw_text[:4000]}"},
            ],
        )
        if parsed is None:
            raise NonRetryableError("model returned invalid structured output")
        return parsed

    try:
        parsed = retry_with_backoff(
            call,
            max_attempts=1,
            base=settings.backoff_base_seconds,
            cap=settings.backoff_cap_seconds,
        )
    except Exception as exc:
        raise NonRetryableError(f"model returned invalid structured output: {exc}") from exc
    return parsed, 0
