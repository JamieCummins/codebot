"""
CLI entrypoint for the modular CodeBot pipeline extracted from CodeBot_flow.ipynb.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from codebot.analysis import classify_paper_relevance, extract_paper_analyses_as_json
from codebot.comparison import greedy_unique_bipartite, match_paper_to_code, mine_code_ir
from codebot.comparison.dimension_compare import (
    compare_dimensions_for_matches,
    compare_dimensions_full_context,
)
from codebot.config import DEFAULT_COMPARISON_DIMENSIONS, DEFAULT_R_EXTENSIONS, OpenAISettings, get_openai_client
from codebot.ingestion import fetch_project_yaml, fetch_repo_files
from codebot.parsing import parse_pdf_with_dpt2, parse_pdf_with_grobid
from codebot.reporting import write_csv_from_comparisons, write_json
from codebot.utils import build_tree_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CodeBot pipeline.")
    parser.add_argument("--paper-path", required=True, help="Path to the PDF to parse.")
    parser.add_argument(
        "--parser",
        choices=["grobid", "dpt2"],
        default="grobid",
        help="PDF parser to use.",
    )
    parser.add_argument("--grobid-url", default="https://kermitt2-grobid.hf.space/api/processFulltextDocument")
    parser.add_argument("--dpt2-endpoint", default=None, help="Optional override for DPT-2 endpoint.")
    parser.add_argument("--dpt2-model", default=None, help="Optional override for DPT-2 model name.")
    parser.add_argument("--parser-token", default=None, help="Explicit token for DPT-2 (otherwise env vars).")

    parser.add_argument("--repo-url", required=True, help="GitHub repository URL to ingest.")
    parser.add_argument("--branch", default="main", help="Repository branch to read from.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="File extensions to pull from the repo (default: R-centric set).",
    )
    parser.add_argument("--github-token", default=None, help="GitHub token (otherwise env vars).")

    parser.add_argument("--model", default=OpenAISettings().model, help="OpenAI model to use.")
    parser.add_argument("--reasoning", default=OpenAISettings().reasoning_effort, help="OpenAI reasoning effort.")
    parser.add_argument("--openai-key", default=None, help="Explicit OpenAI API key (otherwise env vars).")

    parser.add_argument(
        "--use-matching",
        action="store_true",
        help="Enable paper↔code matching to filter code snippets before LLM comparison (default: OFF).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k code candidates per paper analysis.")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum score for greedy matching.")

    parser.add_argument(
        "--dimensions-path",
        default=None,
        help="Path to JSON file with comparison dimensions (defaults to notebook dimensions).",
    )
    parser.add_argument("--output-json", default="codebot_run_results.json", help="Where to write the JSON results.")
    parser.add_argument("--output-csv", default="codebot_report.csv", help="Where to write the flattened CSV report.")
    parser.add_argument("--skip-csv", action="store_true", help="Skip writing the CSV report.")
    return parser.parse_args()


def load_dimensions(path: str | None) -> dict:
    if not path:
        return DEFAULT_COMPARISON_DIMENSIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dimensions JSON must be a mapping of dimension -> definition.")
    return data


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def log_progress(prefix: str) -> callable:
    def _inner(msg: str) -> None:
        log(f"{prefix}: {msg}")
    return _inner


def main() -> None:
    args = parse_args()
    dimensions = load_dimensions(args.dimensions_path)
    settings = OpenAISettings(model=args.model, reasoning_effort=args.reasoning)
    client = get_openai_client(api_key=args.openai_key, settings=settings)

    log("Starting CodeBot pipeline")
    log(f"PDF parser: {args.parser}")
    log(f"Repo: {args.repo_url} (branch {args.branch})")

    # Parse paper
    log("Parsing paper...")
    if args.parser == "dpt2":
        paper_text = parse_pdf_with_dpt2(
            args.paper_path,
            model=args.dpt2_model,
            endpoint=args.dpt2_endpoint,
            token=args.parser_token,
        )
    else:
        paper_text = parse_pdf_with_grobid(args.paper_path, grobid_url=args.grobid_url)
    log(f"Parsed paper text length: {len(paper_text):,} characters")

    # Extract analyses and classify relevance
    log("Extracting analyses from paper (LLM)...")
    paper_analyses = extract_paper_analyses_as_json(paper_text, client, settings=settings)
    log(f"Extracted {len(paper_analyses)} analyses from paper")

    log("Classifying relevance...")
    paper_relevance = classify_paper_relevance(
        paper_analyses,
        client,
        settings=settings,
        progress=log_progress("Relevance"),
    )
    relevant_count = sum(1 for v in paper_relevance.values() if v == "relevant")
    log(f"Marked {relevant_count}/{len(paper_relevance)} analyses as relevant")

    # Ingest repo + project yaml
    log("Fetching repository tree and code files...")
    extensions = args.extensions or list(DEFAULT_R_EXTENSIONS)
    repo_tree, repo_files = fetch_repo_files(
        args.repo_url,
        branch=args.branch,
        extensions=extensions,
        token=args.github_token,
    )
    project_yaml = fetch_project_yaml(args.repo_url, branch=args.branch, token=args.github_token)
    log(f"Fetched {len(repo_files)} code files matching extensions: {', '.join(sorted(extensions))}")

    repo_tree_str = build_tree_string(repo_tree)
    comparisons = []
    matched_pairs = []
    code_analyses = []

    if args.use_matching:
        log("Matching path enabled: mining code analyses and pre-filtering before LLM comparisons.")
        code_analyses = mine_code_ir(repo_files)
        log(f"Detected {len(code_analyses)} candidate code analyses")

        log("Scoring paper↔code matches...")
        candidate_matches = match_paper_to_code(paper_analyses, code_analyses, top_k=args.top_k)
        matched_pairs = greedy_unique_bipartite(candidate_matches, min_score=args.min_score)
        log(f"Kept {len(matched_pairs)} greedy unique matches (min_score={args.min_score})")
        relevant_matches = [m for m in matched_pairs if paper_relevance.get(m.paper_id) == "relevant"]
        log(f"{len(relevant_matches)} matches remain after relevance filter")

        paper_by_id = {p.analysis_id: p for p in paper_analyses}
        code_by_id = {c.analysis_id: c for c in code_analyses}

        log("Running dimension-wise comparisons (LLM) with filtered snippets...")
        comparisons = compare_dimensions_for_matches(
            paper_by_id,
            code_by_id,
            relevant_matches,
            dimensions,
            client,
            settings=settings,
            progress=log_progress("Compare"),
        )
        log(f"Completed {len(comparisons)} comparison rows")
    else:
        log("Full-context path enabled (default): sending entire codebase to LLM for each relevant analysis.")
        paper_by_id = {p.analysis_id: p for p in paper_analyses}
        comparisons = compare_dimensions_full_context(
            paper_by_id,
            paper_relevance,
            dimensions,
            paper_text=paper_text,
            repo_tree=repo_tree_str,
            project_yaml=project_yaml,
            repo_files=repo_files,
            client=client,
            settings=settings,
            progress=log_progress("Compare"),
        )
        log(f"Completed {len(comparisons)} comparison rows (full-context)")

    run_results = {
        "meta": {
            "version": "0.1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parser": args.parser,
            "repo_url": args.repo_url,
            "branch": args.branch,
            "use_matching": args.use_matching,
            "num_paper_analyses": len(paper_analyses),
            "num_code_analyses": len(code_analyses),
            "num_matches": len(matched_pairs),
            "num_comparisons": len(comparisons),
        },
        "paper_analyses": [asdict(x) for x in paper_analyses],
        "code_analyses": [asdict(x) for x in code_analyses],
        "paper_relevance": paper_relevance,
        "repo_tree": repo_tree_str,
        "project_yaml": project_yaml,
        "matches": [asdict(x) for x in matched_pairs],
        "comparisons": comparisons,
    }

    write_json(run_results, args.output_json)
    if not args.skip_csv:
        write_csv_from_comparisons(comparisons, args.output_csv)

    log(f"Wrote JSON results to {args.output_json}")
    if not args.skip_csv:
        log(f"Wrote CSV report to {args.output_csv}")
    log("Pipeline complete")


if __name__ == "__main__":
    main()
