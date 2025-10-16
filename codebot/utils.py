"""Utility helpers for CodeBot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import orjson
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ServiceConfig

JSON = Dict[str, Any]


class HttpError(RuntimeError):
    """Raised when HTTP requests fail irrecoverably."""


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def http_post(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[JSON] = None,
    data: Optional[bytes] = None,
    timeout: int = 60,
) -> requests.Response:
    """Perform a POST request with retries."""

    response = requests.post(url, headers=headers, json=json_body, data=data, timeout=timeout)
    if response.status_code >= 400:
        raise HttpError(f"POST {url} failed with status {response.status_code}: {response.text[:200]}")
    return response


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def http_get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> requests.Response:
    """Perform a GET request with retries."""

    response = requests.get(url, headers=headers, params=params, timeout=timeout)
    if response.status_code >= 400:
        raise HttpError(f"GET {url} failed with status {response.status_code}: {response.text[:200]}")
    return response


def load_json(path: Path | str) -> Any:
    """Load JSON data from disk using orjson."""

    with open(path, "rb") as file_handle:
        return orjson.loads(file_handle.read())


def save_json(path: Path | str, data: Any) -> None:
    """Save JSON data to disk using orjson with pretty formatting."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file_handle:
        file_handle.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


NOTEBOOK_CODE_CELL_PATTERN = re.compile(r"^code$", re.IGNORECASE)
SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_notebook_code_cells(content: JSON) -> str:
    """Extract concatenated code cells from a notebook JSON payload."""

    if not isinstance(content, dict):
        return ""
    cells: Iterable[JSON] = content.get("cells", [])
    snippets: List[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            snippets.append(source)
        elif isinstance(source, list):
            snippets.append("".join(source))
    return "\n".join(snippets)


def normalize_text(text: str) -> str:
    """Normalize whitespace in text for matching."""

    return re.sub(r"\s+", " ", text).strip()


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    """Compute Jaccard similarity between two iterables of strings."""

    set_a = {token.lower() for token in a if token}
    set_b = {token.lower() for token in b if token}
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def split_identifiers(text: str) -> List[str]:
    """Extract safe identifier-like tokens from text."""

    return SAFE_IDENTIFIER_PATTERN.findall(text)


def prepare_authorization_headers(config: ServiceConfig) -> Dict[str, str]:
    """Create headers for Landing AI requests."""

    return {
        "Authorization": f"Bearer {config.landing_ai_token}",
        "Content-Type": "application/octet-stream",
    }


def chunk_lines(text: str, start: int, end: int) -> str:
    """Return text chunk between given line bounds."""

    lines = text.splitlines()
    start_index = max(start - 1, 0)
    end_index = min(end, len(lines))
    return "\n".join(lines[start_index:end_index])


def safe_call(callable_: Callable[[], Any], fallback: Any) -> Any:
    """Execute callable and return fallback if it raises."""

    try:
        return callable_()
    except Exception:
        return fallback
