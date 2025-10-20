"""Minimal tenacity stub for testing."""

from __future__ import annotations

from typing import Any, Callable


def retry(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*f_args: Any, **f_kwargs: Any) -> Any:
            return func(*f_args, **f_kwargs)

        return wrapper

    return decorator


def stop_after_attempt(attempts: int) -> None:  # pragma: no cover - compatibility stub
    return None


def wait_exponential(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - compatibility stub
    return None


class RetryError(RuntimeError):
    pass
