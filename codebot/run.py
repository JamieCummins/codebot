"""Workflow orchestration for CodeBot."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from .classify import classify_relevance
from .compare import compare_dimensions
from .config import ServiceConfig, load_service_config
from .ingest_paper import extract_paper_analyses, parse_pdf_with_dpt2
from .ingest_repo import fetch_repo_code_files, mine_code_ir
from .match import greedy_unique_bipartite, match_paper_to_code
from .schemas import CodeAnalysisIR, MatchEdge, PaperAnalysisIR, RunResults
from .utils import save_json

console = Console()


class WorkflowError(RuntimeError):
    """Raised when the workflow cannot complete."""


def _build_lookup(items: List[PaperAnalysisIR | CodeAnalysisIR]) -> Dict[str, PaperAnalysisIR | CodeAnalysisIR]:
    return {item.analysis_id: item for item in items}


def run_workflow(
    *,
    pdf_path: str,
    repo_url: str,
    out_path: str,
    branch: str = "main",
    min_score: float = 0.35,
    model_name: Optional[str] = None,
    config: Optional[ServiceConfig] = None,
) -> RunResults:
    """Execute the full CodeBot workflow."""

    config = config or load_service_config()

    console.log("Parsing paper with Landing AI DPT-2...")
    paper_text = parse_pdf_with_dpt2(pdf_path, config=config)

    console.log("Extracting analyses from paper text...")
    paper_analyses = extract_paper_analyses(paper_text, config=config)

    console.log("Fetching repository files from GitHub...")
    repo_files = fetch_repo_code_files(repo_url, branch=branch, config=config)
    console.log(f"Fetched {len(repo_files)} candidate files")

    console.log("Mining code analyses...")
    code_analyses = mine_code_ir(repo_files)
    if not code_analyses:
        raise WorkflowError("No code analyses mined from repository.")

    console.log("Classifying paper relevance...")
    relevance = classify_relevance(paper_analyses, config=config)

    console.log("Matching paper analyses to code snippets...")
    candidate_edges = match_paper_to_code(paper_analyses, code_analyses)
    selected_matches = greedy_unique_bipartite(candidate_edges, min_score=min_score)
    if not selected_matches:
        raise WorkflowError("No matches met the minimum score threshold.")

    paper_lookup: Dict[str, PaperAnalysisIR] = _build_lookup(paper_analyses)  # type: ignore[arg-type]
    code_lookup: Dict[str, CodeAnalysisIR] = _build_lookup(code_analyses)  # type: ignore[arg-type]

    console.log("Comparing matched analyses dimension-by-dimension...")
    comparisons: List[Dict[str, object]] = []
    for match in selected_matches:
        if relevance.get(match.paper_id) != "relevant":
            continue
        diffs = compare_dimensions(
            match,
            paper_lookup=paper_lookup,
            code_lookup=code_lookup,
            config=config,
            model_name=model_name,
        )
        comparisons.append(
            {
                "paper_id": match.paper_id,
                "code_id": match.code_id,
                "dimension_diffs": [diff.model_dump() for diff in diffs],
            }
        )

    run_results = RunResults(
        meta={
            "version": "0.1.0",
            "model": model_name or config.openai_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        paper_analyses=paper_analyses,
        code_analyses=code_analyses,
        matches=selected_matches,
        comparisons=comparisons,
    )

    output_path = Path(out_path)
    save_json(output_path, run_results.model_dump())

    artifacts_dir = output_path.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_json(artifacts_dir / "paper_analyses.json", [paper.model_dump() for paper in paper_analyses])
    save_json(artifacts_dir / "code_analyses.json", [code.model_dump() for code in code_analyses])
    save_json(artifacts_dir / "matches.json", [match.model_dump() for match in selected_matches])
    save_json(artifacts_dir / "comparisons.json", comparisons)

    _print_summary(paper_analyses, code_analyses, selected_matches, comparisons)

    return run_results


def _print_summary(
    paper_analyses: List[PaperAnalysisIR],
    code_analyses: List[CodeAnalysisIR],
    matches: List[MatchEdge],
    comparisons: List[Dict[str, object]],
) -> None:
    table = Table(title="CodeBot Summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Paper analyses", str(len(paper_analyses)))
    table.add_row("Code analyses", str(len(code_analyses)))
    table.add_row("Matches", str(len(matches)))
    table.add_row("Comparisons", str(len(comparisons)))
    console.print(table)
