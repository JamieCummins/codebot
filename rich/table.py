"""Minimal table implementation for testing."""

from __future__ import annotations

from typing import List


class Table:
    def __init__(self, title: str | None = None) -> None:
        self.title = title
        self.columns: List[str] = []
        self.rows: List[List[str]] = []

    def add_column(self, header: str, **_: object) -> None:
        self.columns.append(header)

    def add_row(self, *values: str) -> None:
        self.rows.append(list(values))

    def __str__(self) -> str:  # pragma: no cover - trivial
        output = []
        if self.title:
            output.append(self.title)
        output.extend(
            ", ".join(row) for row in self.rows
        )
        return "\n".join(output)
