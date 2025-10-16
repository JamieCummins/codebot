"""Matching paper analyses to code analyses."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .schemas import CodeAnalysisIR, MatchEdge, PaperAnalysisIR
from .utils import jaccard_similarity


def _family_score(paper: PaperAnalysisIR, code: CodeAnalysisIR) -> float:
    if paper.model_family_hint and code.model_family:
        return 1.0 if paper.model_family_hint == code.model_family else 0.0
    if paper.model_family_hint or code.model_family:
        return 0.2
    return 0.0


def _formula_bonus(paper: PaperAnalysisIR, code: CodeAnalysisIR) -> float:
    snippet = (code.formula_hint or "") + " " + code.snippet
    snippet_lower = snippet.lower()
    for token in paper.variables_hint:
        if token.lower() in snippet_lower:
            return 1.0
    return 0.0


def match_paper_to_code(
    papers: Iterable[PaperAnalysisIR],
    codes: Iterable[CodeAnalysisIR],
    *,
    top_k: int = 3,
) -> List[MatchEdge]:
    """Compute candidate match edges between papers and code."""

    codes_list = list(codes)
    edges: List[MatchEdge] = []
    for paper in papers:
        scored: List[MatchEdge] = []
        for code in codes_list:
            family_score = _family_score(paper, code)
            variables_score = jaccard_similarity(paper.variables_hint, code.variables_hint)
            formula_bonus = _formula_bonus(paper, code)
            score = 0.6 * family_score + 0.4 * variables_score + 0.2 * formula_bonus
            if score <= 0:
                continue
            reasons: Dict[str, float | str] = {
                "family_score": family_score,
                "variables_score": variables_score,
                "formula_bonus": formula_bonus,
            }
            edge = MatchEdge(
                paper_id=paper.analysis_id,
                code_id=code.analysis_id,
                score=round(float(score), 4),
                reasons=reasons,
            )
            scored.append(edge)
        scored.sort(key=lambda edge: edge.score, reverse=True)
        edges.extend(scored[:top_k])
    return edges


def greedy_unique_bipartite(
    edges: Iterable[MatchEdge],
    *,
    min_score: float = 0.35,
) -> List[MatchEdge]:
    """Select unique matches greedily with a score threshold."""

    chosen: List[MatchEdge] = []
    used_papers: set[str] = set()
    used_codes: set[str] = set()
    for edge in sorted(edges, key=lambda edge: edge.score, reverse=True):
        if edge.score < min_score:
            continue
        if edge.paper_id in used_papers or edge.code_id in used_codes:
            continue
        used_papers.add(edge.paper_id)
        used_codes.add(edge.code_id)
        chosen.append(edge)
    return chosen
