from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import urlparse

import requests

from codebot.config import DEFAULT_R_EXTENSIONS, GithubSettings, get_github_token
from codebot.models import RepoFile
from codebot.utils import build_tree_string


def parse_repo_url(repo_url: str) -> Tuple[str, str]:
    """
    Returns (owner, repo_name) for a GitHub URL.
    """
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse owner/repo from {repo_url}")
    owner, repo = parts[0], parts[1]
    repo = repo[:-4] if repo.endswith(".git") else repo
    return owner, repo


def _auth_headers(token: str | None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_repo_tree(
    repo_url: str,
    *,
    branch: str = "main",
    token: str | None = None,
) -> list[dict]:
    owner, repo = parse_repo_url(repo_url)
    token = token or get_github_token()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(api_url, headers=_auth_headers(token), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("tree", [])


def fetch_repo_files(
    repo_url: str,
    *,
    branch: str = "main",
    extensions: Iterable[str] | None = None,
    token: str | None = None,
) -> tuple[list[dict], list[RepoFile]]:
    """
    Fetches the GitHub tree and pulls down files matching `extensions`.
    """
    extensions = set(extensions or DEFAULT_R_EXTENSIONS)
    tree = fetch_repo_tree(repo_url, branch=branch, token=token)
    owner, repo = parse_repo_url(repo_url)
    token = token or get_github_token()
    files: list[RepoFile] = []

    for file in tree:
        path = file.get("path")
        if not path:
            continue
        _, ext = os.path.splitext(path)
        if ext not in extensions:
            continue
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        resp = requests.get(raw_url, headers=_auth_headers(token), timeout=60)
        if resp.status_code != 200:
            continue
        files.append(RepoFile(file_path=path, content=resp.text))
    return tree, files


def fetch_project_yaml(
    repo_url: str,
    *,
    branch: str = "main",
    token: str | None = None,
) -> str | None:
    owner, repo = parse_repo_url(repo_url)
    token = token or get_github_token()
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/project.yaml"
    resp = requests.get(raw_url, headers=_auth_headers(token), timeout=60)
    if resp.status_code == 200:
        return resp.text
    return None


def save_files_to_disk(files: list[RepoFile], output_dir: str | Path) -> None:
    """
    Utility to write fetched files locally if needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rf in files:
        dest = output_dir / rf.file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rf.content, encoding="utf-8")


def tree_string_from_repo(repo_url: str, branch: str = "main", token: str | None = None) -> str:
    tree = fetch_repo_tree(repo_url, branch=branch, token=token)
    return build_tree_string(tree)

