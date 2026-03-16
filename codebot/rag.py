from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from openai import OpenAI

from .models import CodeFile


@dataclass(frozen=True)
class RAGChunk:
    chunk_id: str
    text: str
    source: str  # "paper" | "code"
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    @property
    def location(self) -> str:
        if self.path and self.start_line is not None and self.end_line is not None:
            return f"{self.path}:{self.start_line}-{self.end_line}"
        if self.start_line is not None and self.end_line is not None:
            return f"lines {self.start_line}-{self.end_line}"
        return self.chunk_id


@dataclass(frozen=True)
class RetrievalHit:
    chunk: RAGChunk
    score: float


class EmbeddingIndex:
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        chunks: Sequence[RAGChunk],
        embeddings: Sequence[Sequence[float]],
    ):
        self.client = client
        self.model = model
        self.chunks = list(chunks)
        self.embeddings = [list(e) for e in embeddings]

    @classmethod
    def build(
        cls,
        *,
        client: OpenAI,
        model: str,
        chunks: Sequence[RAGChunk],
        batch_size: int = 64,
    ) -> "EmbeddingIndex":
        if not chunks:
            return cls(client=client, model=model, chunks=[], embeddings=[])

        vectors: list[list[float]] = []
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        return cls(client=client, model=model, chunks=chunks, embeddings=vectors)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def _embed_query(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=query)
        return list(resp.data[0].embedding)

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if self.is_empty():
            return []
        q = self._embed_query(query)
        scored: list[RetrievalHit] = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            scored.append(RetrievalHit(chunk=chunk, score=self._cosine(q, emb)))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(top_k, 0)]


def build_paper_chunks(
    paper_text: str,
    *,
    chunk_chars: int = 1400,
    overlap_chars: int = 250,
) -> list[RAGChunk]:
    text = (paper_text or "").strip()
    if not text:
        return []
    chunks: list[RAGChunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        snippet = text[start:end].strip()
        if snippet:
            start_line = text[:start].count("\n") + 1
            end_line = text[:end].count("\n") + 1
            chunks.append(
                RAGChunk(
                    chunk_id=f"paper_chunk_{idx}",
                    text=snippet,
                    source="paper",
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            idx += 1
        if end >= len(text):
            break
        next_start = max(end - overlap_chars, start + 1)
        start = next_start
    return chunks


def build_code_chunks(
    code_files: Iterable[CodeFile],
    *,
    chunk_lines: int = 80,
    overlap_lines: int = 20,
) -> list[RAGChunk]:
    chunks: list[RAGChunk] = []
    stride = max(chunk_lines - overlap_lines, 1)
    for cf in code_files:
        lines = cf.content.splitlines()
        if not lines:
            continue
        for i in range(0, len(lines), stride):
            section = lines[i : i + chunk_lines]
            if not section:
                continue
            text = "\n".join(section).strip()
            if not text:
                continue
            start_line = i + 1
            end_line = i + len(section)
            chunks.append(
                RAGChunk(
                    chunk_id=f"{cf.path}:{start_line}-{end_line}",
                    text=text,
                    source="code",
                    path=cf.path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
    return chunks


def looks_like_code(text: str) -> bool:
    if not text.strip():
        return False
    code_markers = [
        r"<-",
        r"\bif\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\bfunction\b",
        r"\bglm\s*\(",
        r"\bcoxph\s*\(",
        r"\bselect\s*\(",
        r"\bmutate\s*\(",
        r"=",
    ]
    return any(re.search(pattern, text) for pattern in code_markers)

