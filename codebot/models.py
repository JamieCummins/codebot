from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperAnalysisIR:
    analysis_id: str                        # "P-001"
    analysis_description: str               # human-readable
    location: str                           # "methods ¶12", "table 2", etc.
    model_family_hint: Optional[str] = None # "logistic", "cox", "t-test", etc.
    variables_hint: List[str] = field(default_factory=list)


@dataclass
class CodeAnalysisIR:
    analysis_id: str                        # "C-001"
    file_path: str
    line_start: int
    line_end: int
    snippet: str
    model_family: Optional[str] = None      # detected; "glm[binomial]", "coxph", "t.test", etc.
    formula_hint: Optional[str] = None      # if found
    variables_hint: List[str] = field(default_factory=list)


@dataclass
class MatchEdge:
    paper_id: str
    code_id: str
    score: float
    reasons: Dict[str, Any]


@dataclass
class DimensionDiff:
    dimension: str
    status: str           # "match" | "minor" | "major" | "unknown"
    explanation: str
    evidence: Dict[str, Any]


@dataclass
class RepoFile:
    file_path: str
    content: str

