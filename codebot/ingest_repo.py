"""Repository ingestion utilities."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

from .config import ServiceConfig, load_service_config
from .schemas import CodeAnalysisIR
from .utils import extract_notebook_code_cells, http_get, normalize_text, split_identifiers

SUPPORTED_EXTENSIONS = {".r", ".rmd", ".qmd", ".py", ".ipynb", ".do"}
CODE_PATTERN_GROUPS = {
    "r": {
        "logistic": re.compile(r"glm\s*\(.*family\s*=\s*binomial", re.IGNORECASE | re.DOTALL),
        "logistic_glmer": re.compile(r"glmer\s*\(.*family\s*=\s*binomial", re.IGNORECASE | re.DOTALL),
        "cox": re.compile(r"coxph\s*\(", re.IGNORECASE),
        "t-test": re.compile(r"t\.test\s*\(", re.IGNORECASE),
        "chi-square": re.compile(r"chisq\.test\s*\(", re.IGNORECASE),
        "poisson": re.compile(r"glm\s*\(.*family\s*=\s*poisson", re.IGNORECASE | re.DOTALL),
        "psm": re.compile(r"matchit\s*\(", re.IGNORECASE),
    },
    "python": {
        "logistic": re.compile(r"sm\.Logit\s*\(", re.IGNORECASE),
        "logistic_glm": re.compile(r"sm\.GLM\s*\(.*Binomial", re.IGNORECASE | re.DOTALL),
        "poisson": re.compile(r"sm\.GLM\s*\(.*Poisson", re.IGNORECASE | re.DOTALL),
        "cox": re.compile(r"CoxPHFitter\s*\(", re.IGNORECASE),
        "t-test": re.compile(r"scipy\.stats\.ttest_|pingouin\.ttest", re.IGNORECASE),
        "chi-square": re.compile(r"scipy\.stats\.chi|chi2", re.IGNORECASE),
    },
    "stata": {
        "logistic": re.compile(r"^\s*(logit|logistic)\b", re.IGNORECASE | re.MULTILINE),
        "poisson": re.compile(r"^\s*poisson\b", re.IGNORECASE | re.MULTILINE),
        "linear": re.compile(r"^\s*regress\b", re.IGNORECASE | re.MULTILINE),
        "cox": re.compile(r"^\s*stcox\b", re.IGNORECASE | re.MULTILINE),
        "t-test": re.compile(r"^\s*ttest\b", re.IGNORECASE | re.MULTILINE),
        "chi-square": re.compile(r"^\s*tabulate.+,\s*chi2", re.IGNORECASE | re.MULTILINE),
    },
}
def _parse_repo_url(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid repository URL: {repo_url}")
    return parts[0], parts[1]


def fetch_repo_code_files(
    repo_url: str,
    *,
    branch: str = "main",
    config: Optional[ServiceConfig] = None,
) -> List[Dict[str, str]]:
    """Fetch code files from a GitHub repository using the Trees API."""

    config = config or load_service_config()
    owner, repo = _parse_repo_url(repo_url)
    base_headers = {"Accept": "application/vnd.github+json"}
    if config.github_token:
        base_headers["Authorization"] = f"Bearer {config.github_token}"

    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
    response = http_get(
        tree_url,
        params={"recursive": "1"},
        headers=base_headers,
        timeout=config.request_timeout,
    )
    payload = response.json()
    tree = payload.get("tree", [])
    files: List[Dict[str, str]] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        extension = Path(path).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        content_response = http_get(
            contents_url,
            params={"ref": branch},
            headers=base_headers,
            timeout=config.request_timeout,
        )
        content_payload = content_response.json()
        content_data = content_payload.get("content", "")
        if content_payload.get("encoding") == "base64":
            decoded = base64.b64decode(content_data).decode(errors="ignore")
        else:
            decoded = content_data
        if path.lower().endswith(".ipynb"):
            try:
                notebook_json = json.loads(decoded)
                decoded = extract_notebook_code_cells(notebook_json)
            except json.JSONDecodeError:
                decoded = ""
        files.append({"path": path, "content": decoded})
    return files


def _window_snippet(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines()
    start_index = max(start_line - 1, 0)
    end_index = min(end_line, len(lines))
    return "\n".join(lines[start_index:end_index])


def _line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _gather_variables(snippet: str) -> List[str]:
    return split_identifiers(snippet)


def mine_code_ir(files: Iterable[Dict[str, str]]) -> List[CodeAnalysisIR]:
    """Mine code analyses from repository files using regex heuristics."""

    analyses: List[CodeAnalysisIR] = []
    counter = 1
    for file_entry in files:
        path = file_entry.get("path", "")
        content = file_entry.get("content", "")
        normalized_path = path.lower()
        language = "python"
        if normalized_path.endswith(('.r', '.rmd', '.qmd')):
            language = "r"
        elif normalized_path.endswith('.do'):
            language = "stata"
        elif normalized_path.endswith('.ipynb'):
            language = "python"
        patterns = CODE_PATTERN_GROUPS.get(language, {})
        for family, pattern in patterns.items():
            for match in pattern.finditer(content):
                line_start = _line_number(content, match.start()) - 15
                line_end = _line_number(content, match.end()) + 15
                line_start = max(line_start, 1)
                line_end = max(line_end, line_start + 1)
                snippet = _window_snippet(content, line_start, line_end)
                formula_hint = normalize_text(match.group(0))[:200]
                variables_hint = _gather_variables(snippet)
                analysis = CodeAnalysisIR(
                    analysis_id=f"C-{counter:03d}",
                    file_path=path,
                    line_start=line_start,
                    line_end=line_end,
                    snippet=snippet,
                    model_family=family if family not in {"logistic_glmer", "logistic"} else "logistic",
                    formula_hint=formula_hint,
                    variables_hint=variables_hint,
                )
                analyses.append(analysis)
                counter += 1
    return analyses
