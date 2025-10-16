"""Pydantic schemas for CodeBot intermediate representations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BaseIRModel(BaseModel):
    """Base class with convenience helpers."""

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }

    def model_dump_json(self, **kwargs) -> str:  # type: ignore[override]
        """Serialize model to JSON using orjson under the hood."""

        import orjson

        return orjson.dumps(self.model_dump(**kwargs)).decode()

    @classmethod
    def json_schema(cls) -> Dict[str, object]:  # type: ignore[override]
        """Return JSON schema for the model."""

        return cls.model_json_schema()


class PaperAnalysisIR(BaseIRModel):
    analysis_id: str = Field(..., description="Identifier such as P-001")
    analysis_description: str = Field(..., description="Description of the analysis")
    location: str = Field(..., description="Where in the paper the analysis is located")
    model_family_hint: Optional[str] = Field(None, description="Heuristic model family hint")
    variables_hint: List[str] = Field(default_factory=list, description="Variable tokens")


class CodeAnalysisIR(BaseIRModel):
    analysis_id: str = Field(..., description="Identifier such as C-001")
    file_path: str
    line_start: int
    line_end: int
    snippet: str
    model_family: Optional[str] = None
    formula_hint: Optional[str] = None
    variables_hint: List[str] = Field(default_factory=list)


class MatchEdge(BaseIRModel):
    paper_id: str
    code_id: str
    score: float
    reasons: Dict[str, float | str]


class DimensionDiff(BaseIRModel):
    dimension: str
    status: Literal["match", "minor", "major", "unknown"]
    explanation: str
    evidence: Dict[str, object]


class RunResults(BaseIRModel):
    meta: Dict[str, object] = Field(default_factory=lambda: {
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policies": [],
    })
    paper_analyses: List[PaperAnalysisIR] = Field(default_factory=list)
    code_analyses: List[CodeAnalysisIR] = Field(default_factory=list)
    matches: List[MatchEdge] = Field(default_factory=list)
    comparisons: List[Dict[str, object]] = Field(default_factory=list)


__all__ = [
    "PaperAnalysisIR",
    "CodeAnalysisIR",
    "MatchEdge",
    "DimensionDiff",
    "RunResults",
]
