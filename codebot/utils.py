from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def take(lines: list[str], start: int, end: int) -> str:
    start = max(start, 0)
    end = min(end, len(lines))
    return "\n".join(lines[start:end])


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    denom = sum(exps) or 1.0
    return [v / denom for v in exps]


def to_compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


def parse_json_or_fallback(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
        return {"data": loaded}
    except Exception:
        return {"raw": text}


def build_tree_string(tree: list[dict[str, Any]]) -> str:
    return "\n".join(sorted([item.get("path", "") for item in tree if isinstance(item, dict) and "path" in item]))
