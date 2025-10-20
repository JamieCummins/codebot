"""Minimal stub of the requests module for offline testing."""

from __future__ import annotations

from typing import Any, Dict, Optional


class Response:
    def __init__(self, *, status_code: int = 200, text: str = "", json_data: Optional[Any] = None) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def json(self) -> Any:
        return self._json_data


def post(url: str, *, headers: Optional[Dict[str, str]] = None, json: Any = None, data: Any = None, timeout: int = 0) -> Response:
    raise RuntimeError("HTTP requests are not available in the test environment.")


def get(url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 0) -> Response:
    raise RuntimeError("HTTP requests are not available in the test environment.")
