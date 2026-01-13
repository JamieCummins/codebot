import json
import math
import re
from typing import Any, List, Sequence


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def take(lines: List[str], start: int, end: int) -> str:
    start = max(start, 0)
    end = min(end, len(lines))
    return "\n".join(lines[start:end])


def softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs) if xs else 0.0
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def to_compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


def parse_json_or_fallback(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {"status": "unknown", "explanation": "Model returned non-JSON", "raw": text[:2000]}


def build_tree_string(tree: list[dict[str, str]]) -> str:
    """
    Formats a GitHub API tree response (list of dicts) into a simple newline-separated string.
    """
    return "\n".join(sorted([item.get("path", "") for item in tree if "path" in item]))

