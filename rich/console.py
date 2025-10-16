"""Minimal console implementation for testing."""

from __future__ import annotations

from typing import Any


class Console:
    def print(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        message = " ".join(str(arg) for arg in args)
        print(message)

    def log(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        self.print(*args, **kwargs)
