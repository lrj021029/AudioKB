"""一个处理短暂 Milvus 一致性/可用性错误的简单重试辅助工具."""
from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from app.shared.config.audiobook_config import (
    MILVUS_RETRY_ATTEMPTS,
    MILVUS_RETRY_BASE_DELAY_SECONDS,
)
from app.shared.runtime.logger import logger

P = ParamSpec("P")
R = TypeVar("R")

_TRANSIENT_MILVUS_ERRORS = (
    "timestamp lag too large",
    "unavailable",
    "deadline exceeded",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
    "timeout",
)


def retry_on_milvus_error(
    func: Callable[P, R] | None = None,
    *,
    attempts: int = MILVUS_RETRY_ATTEMPTS,
    base_delay: float = MILVUS_RETRY_BASE_DELAY_SECONDS,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry only recognized transient Milvus errors; propagate all other failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if base_delay < 0:
        raise ValueError("base_delay cannot be negative")

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(target)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, attempts + 1):
                try:
                    return target(*args, **kwargs)
                except Exception as exc:
                    message = str(exc).lower()
                    is_transient = any(token in message for token in _TRANSIENT_MILVUS_ERRORS)
                    if not is_transient or attempt == attempts:
                        raise
                    delay = float(base_delay * (2 ** (attempt - 1)))
                    logger.warning(
                        f"Milvus 暂时性错误，第 {attempt}/{attempts} 次重试，"
                        f"{delay:.2f}s 后再次执行：{exc}"
                    )
                    time.sleep(delay)
            raise RuntimeError("unreachable retry state")

        return wrapped

    return decorate(func) if func is not None else decorate
