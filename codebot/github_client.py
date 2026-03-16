from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import requests

from .config import DEFAULT_R_EXTENSIONS, get_github_token
from .models import CodeFile
from .utils import build_tree_string


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    branch: str = "main"


@dataclass(frozen=True)
class RepoContext:
    ref: RepoRef
    tree: list[dict]
    tree_string: str
    files: list[CodeFile]
    project_yaml: str | None = None


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse owner/repo from {repo_url}")
    owner, repo = parts[0], parts[1]
    repo = repo[:-4] if repo.endswith(".git") else repo
    return owner, repo


class GithubClient:
    def __init__(self, *, token: str | None = None, timeout: int = 60):
        self.token = token or get_github_token()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _get(self, url: str) -> requests.Response:
        return self.session.get(url, timeout=self.timeout)

    def repo_ref(self, repo_url: str, branch: str = "main") -> RepoRef:
        owner, repo = parse_repo_url(repo_url)
        return RepoRef(owner=owner, repo=repo, branch=branch)

    def fetch_repo_tree(self, ref: RepoRef) -> list[dict]:
        api_url = f"https://api.github.com/repos/{ref.owner}/{ref.repo}/git/trees/{ref.branch}?recursive=1"
        resp = self._get(api_url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("tree", [])

    def fetch_raw_file(self, ref: RepoRef, path: str) -> str | None:
        raw_url = f"https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/{ref.branch}/{path}"
        resp = self._get(raw_url)
        if resp.status_code != 200:
            return None
        return resp.text

    def fetch_project_yaml(self, ref: RepoRef) -> str | None:
        return self.fetch_raw_file(ref, "project.yaml")

    def fetch_repo_files(self, ref: RepoRef, tree: list[dict], extensions: Iterable[str] | None = None) -> list[CodeFile]:
        ext_set = set(extensions or DEFAULT_R_EXTENSIONS)
        files: list[CodeFile] = []
        for item in tree:
            path = item.get("path")
            if not path:
                continue
            _, ext = os.path.splitext(path)
            if ext_set and ext not in ext_set:
                continue
            content = self.fetch_raw_file(ref, path)
            if content is None:
                continue
            files.append(CodeFile(path=path, content=content))
        return files

    def load_repo_context(
        self,
        repo_url: str,
        *,
        branch: str = "main",
        extensions: Iterable[str] | None = None,
    ) -> RepoContext:
        ref = self.repo_ref(repo_url, branch=branch)
        tree = self.fetch_repo_tree(ref)
        files = self.fetch_repo_files(ref, tree, extensions=extensions)
        project_yaml = self.fetch_project_yaml(ref)
        return RepoContext(
            ref=ref,
            tree=tree,
            tree_string=build_tree_string(tree),
            files=files,
            project_yaml=project_yaml,
        )


def load_repo_context(
    repo_url: str,
    *,
    branch: str = "main",
    extensions: Iterable[str] | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> RepoContext:
    client = GithubClient(token=token, timeout=timeout)
    return client.load_repo_context(repo_url, branch=branch, extensions=extensions)

