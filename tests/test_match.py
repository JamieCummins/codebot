"""Tests for matching algorithms."""

from codebot.match import greedy_unique_bipartite, match_paper_to_code
from codebot.schemas import CodeAnalysisIR, MatchEdge, PaperAnalysisIR


def _paper(model_family: str, variables: list[str]) -> PaperAnalysisIR:
    return PaperAnalysisIR(
        analysis_id="P-001",
        analysis_description="Logistic regression on age and sex",
        location="Section 3",
        model_family_hint=model_family,
        variables_hint=variables,
    )


def _code(model_family: str, variables: list[str]) -> CodeAnalysisIR:
    return CodeAnalysisIR(
        analysis_id="C-001",
        file_path="analysis.R",
        line_start=1,
        line_end=10,
        snippet="glm(outcome ~ age + sex, family = binomial)",
        model_family=model_family,
        formula_hint="glm(outcome ~ age + sex)",
        variables_hint=variables,
    )


def test_match_paper_to_code_scores_positive() -> None:
    paper = _paper("logistic", ["age", "sex"])
    code = _code("logistic", ["age", "sex"])
    edges = match_paper_to_code([paper], [code])
    assert edges and edges[0].score > 0.7


def test_greedy_unique_bipartite_selects_highest() -> None:
    paper = _paper("logistic", ["age", "sex"])
    code_primary = _code("logistic", ["age", "sex"])
    code_secondary = _code("logistic", ["age", "income"])
    code_secondary.analysis_id = "C-002"
    edges = match_paper_to_code([paper], [code_primary, code_secondary])
    selected = greedy_unique_bipartite(edges, min_score=0.1)
    assert len(selected) == 1
    assert selected[0].code_id == "C-001"
