from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List

from codebot.config import OpenAISettings
from codebot.models import CodeAnalysisIR, DimensionDiff, MatchEdge, PaperAnalysisIR, RepoFile
from codebot.utils import normalize_space, to_compact_json


def compare_one_dimension(
    pa: PaperAnalysisIR,
    ca: CodeAnalysisIR,
    dimension_name: str,
    dimension_def: str,
    client,
    settings: OpenAISettings | None = None,
) -> DimensionDiff:
    settings = settings or OpenAISettings()
    prompt = f"""
You are comparing one dimension between a paper-described analysis and a code snippet.

Dimension: {dimension_name}
Definition: {dimension_def}

Task: Decide if the paper and code MATCH, MINORLY DEVIATE, MAJORLY DEVIATE, or UNKNOWN for this dimension.
- "match": substantively equivalent
- "minor": small difference unlikely to affect inference
- "major": substantive mismatch (e.g., different model family, wrong link, missing random effects)
- "unknown": insufficient info

Paper analysis description:
{pa.analysis_description}

Code snippet (file {ca.file_path}, lines {ca.line_start}-{ca.line_end}):
{ca.snippet}

Return STRICT JSON:
{{
  "status": "match" | "minor" | "major" | "unknown",
  "explanation": "short reason (<=2 sentences)"
}}
"""
    resp = client.chat.completions.create(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        messages=[
            {"role": "system", "content": "Return only valid JSON with fields 'status' and 'explanation'."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        obj = json.loads(raw)
        status = obj.get("status", "unknown").lower()
        explanation = normalize_space(obj.get("explanation", ""))
    except Exception:
        status, explanation = "unknown", "LLM response could not be parsed."
    return DimensionDiff(
        dimension=dimension_name,
        status=status,
        explanation=explanation,
        evidence={
            "paper_id": pa.analysis_id,
            "code_id": ca.analysis_id,
            "file": ca.file_path,
            "lines": [ca.line_start, ca.line_end],
        },
    )


def compare_one_dimension_full_context(
    pa: PaperAnalysisIR,
    dimension_name: str,
    dimension_def: str,
    *,
    paper_text: str,
    repo_tree: str,
    project_yaml: str | None,
    repo_files: List[RepoFile],
    client,
    settings: OpenAISettings | None = None,
) -> DimensionDiff:
    """
    Full-context comparison: send the entire paper text plus all code files to the LLM.
    Mirrors the original notebook behavior (no pre-filtering).
    """
    settings = settings or OpenAISettings()
    prompt = (
        "You are comparing an analysis described in a paper to its implementation in code.\n"
        "Decide if the paper and code MATCH, MINORLY DEVIATE, MAJORLY DEVIATE, or UNKNOWN "
        "for the given dimension.\n"
        f"Dimension: {dimension_name}\n"
        f"Definition: {dimension_def}\n\n"
        f"Paper analysis description:\n{pa.analysis_description}\n\n"
        "Full paper text:\n"
        f"{paper_text}\n\n"
        "Here is the repository structure:\n"
        f"{repo_tree}\n\n"
        "Here is the project YAML (if present):\n"
        f"{project_yaml or '<missing>'}\n\n"
        "Here are the code files (path + content):\n"
        f"{to_compact_json([asdict(rf) for rf in repo_files])}\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        '  "status": "match" | "minor" | "major" | "unknown",\n'
        '  "explanation": "short reason (<=2 sentences)"\n'
        "}\n"
    )
    resp = client.chat.completions.create(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        messages=[
            {"role": "system", "content": "Return only valid JSON with fields 'status' and 'explanation'."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        obj = json.loads(raw)
        status = obj.get("status", "unknown").lower()
        explanation = normalize_space(obj.get("explanation", ""))
    except Exception:
        status, explanation = "unknown", "LLM response could not be parsed."
    return DimensionDiff(
        dimension=dimension_name,
        status=status,
        explanation=explanation,
        evidence={
            "paper_id": pa.analysis_id,
            "code_id": None,
            "file": None,
            "lines": None,
        },
    )


def compare_dimensions_for_matches(
    papers: Dict[str, PaperAnalysisIR],
    codes: Dict[str, CodeAnalysisIR],
    matches: Iterable[MatchEdge],
    dimensions: Dict[str, str],
    client,
    settings: OpenAISettings | None = None,
    progress: Callable | None = None,
) -> List[dict]:
    """
    Runs dimension-wise comparisons for matched pairs.
    Returns a list of serialisable dicts.
    """
    rows: List[dict] = []
    matches = list(matches)
    for idx, edge in enumerate(matches, start=1):
        if progress:
            progress(f"Comparing match {idx}/{len(matches)}: {edge.paper_id} vs {edge.code_id}")
        pa = papers[edge.paper_id]
        ca = codes[edge.code_id]
        diffs = [
            asdict(compare_one_dimension(pa, ca, dim_name, dim_def, client, settings=settings))
            for dim_name, dim_def in dimensions.items()
        ]
        rows.append(
            {
                "paper_id": edge.paper_id,
                "code_id": edge.code_id,
                "match_score": edge.score,
                "match_reasons": edge.reasons,
                "dimension_diffs": diffs,
            }
        )
    return rows


def compare_dimensions_full_context(
    papers: Dict[str, PaperAnalysisIR],
    relevance: Dict[str, str],
    dimensions: Dict[str, str],
    *,
    paper_text: str,
    repo_tree: str,
    project_yaml: str | None,
    repo_files: List[RepoFile],
    client,
    settings: OpenAISettings | None = None,
    progress: Callable | None = None,
) -> List[dict]:
    """
    Full-context path: send all code files to the LLM (no pre-filtering).
    Only analyses marked relevant are compared.
    """
    rows: List[dict] = []
    relevant_papers = [p for p in papers.values() if relevance.get(p.analysis_id) == "relevant"]
    total = len(relevant_papers)
    for idx, pa in enumerate(relevant_papers, start=1):
        if progress:
            progress(f"Analysis {idx}/{total}: {pa.analysis_id}")
        diffs = []
        for dim_name, dim_def in dimensions.items():
            diffs.append(
                asdict(
                    compare_one_dimension_full_context(
                        pa,
                        dim_name,
                        dim_def,
                        paper_text=paper_text,
                        repo_tree=repo_tree,
                        project_yaml=project_yaml,
                        repo_files=repo_files,
                        client=client,
                        settings=settings,
                    )
                )
            )
        rows.append(
            {
                "paper_id": pa.analysis_id,
                "code_id": None,
                "match_score": None,
                "match_reasons": {},
                "dimension_diffs": diffs,
            }
        )
    return rows
