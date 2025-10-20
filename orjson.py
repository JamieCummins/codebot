"""Minimal orjson-compatible interface using Python's json module."""

from __future__ import annotations

import json
from typing import Any

OPT_INDENT_2 = object()


def dumps(obj: Any, option: Any | None = None) -> bytes:
    indent = 2 if option is OPT_INDENT_2 else None
    return json.dumps(obj, indent=indent).encode("utf-8")


def loads(data: bytes | bytearray | str) -> Any:
    if isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8")
    else:
        text = data
    return json.loads(text)
