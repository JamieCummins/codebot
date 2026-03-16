from __future__ import annotations

from typing import Iterable

from .config import DEFAULT_R_EXTENSIONS
from .github_client import GithubClient, load_repo_context, parse_repo_url
from .models import CodeFile


def fetch_repo_tree(repo_url: str, *, branch: str = "main", token: str | None = None) -> list[dict]:
    client = GithubClient(token=token)
    ref = client.repo_ref(repo_url, branch=branch)
    return client.fetch_repo_tree(ref)


def fetch_repo_files(
    repo_url: str,
    *,
    branch: str = "main",
    extensions: Iterable[str] | None = None,
    token: str | None = None,
) -> tuple[list[dict], list[CodeFile]]:
    ctx = load_repo_context(
        repo_url,
        branch=branch,
        extensions=extensions or DEFAULT_R_EXTENSIONS,
        token=token,
    )
    return ctx.tree, ctx.files


def fetch_project_yaml(repo_url: str, *, branch: str = "main", token: str | None = None) -> str | None:
    client = GithubClient(token=token)
    ref = client.repo_ref(repo_url, branch=branch)
    return client.fetch_project_yaml(ref)


def tree_string_from_repo(repo_url: str, branch: str = "main", token: str | None = None) -> str:
    ctx = load_repo_context(repo_url, branch=branch, extensions=set(), token=token)
    return ctx.tree_string
