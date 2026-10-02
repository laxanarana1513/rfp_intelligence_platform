"""Exponential backoff with full jitter for transient failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

NON_RETRYABLE_STATUS = {400, 401, 403, 404}


class NonRetryableError(Exception):
    """Raised for failures that must not be retried, such as authentication errors."""


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    """Full jitter in ``[0, min(cap, base * 2^attempt)]``. ``attempt`` is zero-based."""
    ceiling = min(cap, base * (2**attempt))
    if ceiling <= 0:
        return 0.0
    return random.uniform(0, ceiling)


def sleep(seconds: float) -> None:
    time.sleep(seconds)


def status_code_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, NonRetryableError):
        return False
    status = status_code_of(exc)
    if status in NON_RETRYABLE_STATUS:
        return False
    return True


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base: float = 0.5,
    cap: float = 8.0,
    retryable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> T:
    """Call ``fn`` until it succeeds or attempts are exhausted.

    ``on_retry`` receives the zero-based attempt that just failed, the delay, and the error.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    check = retryable or is_retryable
    pause = sleeper or sleep
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not check(exc) or attempt >= max_attempts - 1:
                raise
            delay = backoff_delay(attempt, base, cap)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            pause(delay)
    assert last is not None
    raise last
