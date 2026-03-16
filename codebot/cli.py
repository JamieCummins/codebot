from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

from .dimensions import DEFAULT_DIMENSIONS
from .runner import DEFAULT_OUTPUT_DIR, RunConfig, default_client, report_stem_from_paper_path, run_multi, run_single
from .comparison import RAGComparisonStrategy, to_comparison_records
from .extraction import extract_analysis_summaries
from .classification import classify_analyses
from .github_client import load_repo_context, parse_repo_url
from .text_parser import parse_pdf
from .writer import ensure_dir, write_intermediates, write_per_paper
from .models import AnalysisDetail

LOG = logging.getLogger("codebot_flow")
DEFAULT_PAPERS_DIR = Path("papers")
DEFAULT_PAIRS_SPEC_PATH = DEFAULT_PAPERS_DIR / "pairs-specification.csv"


# ----------------- CLI helpers -----------------

def _add_common_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-id", help="Identifier for the paper (defaults to stem of paper path)")
    parser.add_argument("--parser", default="grobid", choices=["grobid", "dpt2", "pypdf", "text"], help="Paper parsing backend")
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI model to use")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="OpenAI reasoning_effort setting",
    )
    parser.add_argument("--mode", default="combined", choices=["combined", "staged"], help="Comparison mode")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument("--no-intermediates", action="store_true", help="Skip saving stage artifacts")


def _add_repo_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--branch", default="main", help="GitHub branch to ingest (when --repo-url is used)")
    parser.add_argument("--github-token", help="GitHub token (optional; falls back to env vars)")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="File extensions to include (defaults to R-centric extensions)",
    )


def _resolve_pairs_csv_path(pairs_csv_arg: str | None) -> Path:
    if pairs_csv_arg:
        return Path(pairs_csv_arg)
    if DEFAULT_PAIRS_SPEC_PATH.exists():
        return DEFAULT_PAIRS_SPEC_PATH
    if DEFAULT_PAPERS_DIR.exists():
        matches = sorted(DEFAULT_PAPERS_DIR.rglob("pairs-specification.csv"))
        if matches:
            return matches[0]
    return DEFAULT_PAIRS_SPEC_PATH


def _resolve_paper_pdf_path(paper_id_value: str, *, papers_dir: Path) -> Path:
    raw = (paper_id_value or "").strip()
    if not raw:
        raise ValueError("paper_id is empty")
    direct = papers_dir / raw
    if direct.exists():
        return direct
    if not raw.lower().endswith(".pdf"):
        with_pdf = papers_dir / f"{raw}.pdf"
        if with_pdf.exists():
            return with_pdf
    raise FileNotFoundError(f"Could not find paper PDF for paper_id '{raw}' under {papers_dir}")


# ----------------- Stage commands -----------------

def stage_paper(args) -> None:
    LOG.info("Stage: paper | parse + extract + classify")
    client = default_client()
    paper = parse_pdf(args.paper, method=args.parser)
    explicit_paper_id = (args.paper_id or "").strip()
    if explicit_paper_id:
        paper.paper_id = explicit_paper_id
    else:
        paper.paper_id = report_stem_from_paper_path(args.paper)

    LOG.info("Stage: paper | extract analysis summaries")
    summaries = extract_analysis_summaries(paper, client=client, model=args.model, reasoning_effort=args.reasoning_effort)
    LOG.info("Stage: paper | prepare analysis details from summaries")
    details = [
        AnalysisDetail(
            analysis_id=s.analysis_id,
            brief_description=s.brief_description,
            location=s.location,
        )
        for s in summaries
    ]
    LOG.info("Stage: paper | classify analyses")
    relevant = classify_analyses(details, client=client, model=args.model, reasoning_effort=args.reasoning_effort)

    state_dir = Path(args.state_dir)
    ensure_dir(state_dir)
    (state_dir / "analyses.json").write_text(json.dumps([a.__dict__ for a in relevant], ensure_ascii=False, indent=2), encoding="utf-8")
    (state_dir / "paper_meta.json").write_text(json.dumps({"paper_id": paper.paper_id}, indent=2), encoding="utf-8")

    # paper evidence per dimension
    LOG.info("Stage: paper | extract dimension evidence from paper")
    strategy = RAGComparisonStrategy(
        client=client, model=args.model, dimensions=DEFAULT_DIMENSIONS, reasoning_effort=args.reasoning_effort
    )
    if hasattr(strategy, "prepare_context"):
        strategy.prepare_context(paper_text=paper.text, code_files=[])
    paper_evidence = []
    for analysis in relevant:
        for dim, definition in DEFAULT_DIMENSIONS.items():
            ev = strategy.extract_paper_dimension(analysis, dim, definition, paper.text)
            ev.paper_id = paper.paper_id
            paper_evidence.append(ev)

    (state_dir / "paper_evidence.json").write_text(
        json.dumps([e.__dict__ for e in paper_evidence], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved analyses + paper evidence to {state_dir}")


def stage_code(args) -> None:
    LOG.info("Stage: code | load analyses + paper evidence")
    client = default_client()
    state_dir = Path(args.state_dir)
    analyses_path = state_dir / "analyses.json"
    paper_ev_path = state_dir / "paper_evidence.json"
    if not analyses_path.exists() or not paper_ev_path.exists():
        raise SystemExit("Run stage paper first to produce analyses.json and paper_evidence.json")

    analyses = [AnalysisDetail(**item) for item in json.loads(analyses_path.read_text(encoding="utf-8"))]
    paper_evidence = { (item["analysis_id"], item["dimension"]): item["value"] for item in json.loads(paper_ev_path.read_text(encoding="utf-8")) }

    LOG.info("Stage: code | load code files from GitHub")
    ctx = load_repo_context(
        args.repo_url,
        branch=args.branch,
        extensions=args.extensions,
        token=args.github_token,
    )
    code_files = ctx.files
    repo_tree = ctx.tree_string
    project_yaml = ctx.project_yaml

    meta = json.loads((state_dir / "paper_meta.json").read_text(encoding="utf-8")) if (state_dir / "paper_meta.json").exists() else {}
    owner, repo = parse_repo_url(args.repo_url)
    repo_fallback = f"{owner}_{repo}"
    explicit_paper_id = (args.paper_id or "").strip() or None
    paper_id = meta.get("paper_id") or explicit_paper_id or repo_fallback

    LOG.info("Stage: code | extract dimension evidence from code")
    strategy = RAGComparisonStrategy(
        client=client, model=args.model, dimensions=DEFAULT_DIMENSIONS, reasoning_effort=args.reasoning_effort
    )
    if hasattr(strategy, "prepare_context"):
        strategy.prepare_context(code_files=code_files)
    code_evidence = []
    for analysis in analyses:
        for dim, definition in DEFAULT_DIMENSIONS.items():
            paper_val = paper_evidence.get((analysis.analysis_id, dim))
            ev = strategy.extract_code_dimension(
                analysis,
                dim,
                definition,
                code_files,
                paper_value=paper_val,
                repo_tree=repo_tree,
                project_yaml=project_yaml,
            )
            ev.paper_id = paper_id
            code_evidence.append(ev)

    (state_dir / "code_evidence.json").write_text(
        json.dumps([e.__dict__ for e in code_evidence], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved code evidence to {state_dir}")


def stage_judge(args) -> None:
    LOG.info("Stage: judge | load analyses + evidence")
    client = default_client()
    state_dir = Path(args.state_dir)
    analyses_path = state_dir / "analyses.json"
    paper_ev_path = state_dir / "paper_evidence.json"
    code_ev_path = state_dir / "code_evidence.json"
    if not (analyses_path.exists() and paper_ev_path.exists() and code_ev_path.exists()):
        raise SystemExit("Run stage paper and code first to produce analyses.json/paper_evidence.json/code_evidence.json")

    analyses = [AnalysisDetail(**item) for item in json.loads(analyses_path.read_text(encoding="utf-8"))]
    paper_evidence = { (item["analysis_id"], item["dimension"]): item for item in json.loads(paper_ev_path.read_text(encoding="utf-8")) }
    code_evidence = { (item["analysis_id"], item["dimension"]): item for item in json.loads(code_ev_path.read_text(encoding="utf-8")) }

    meta = json.loads((state_dir / "paper_meta.json").read_text(encoding="utf-8")) if (state_dir / "paper_meta.json").exists() else {}
    explicit_paper_id = (args.paper_id or "").strip() or None
    paper_id = meta.get("paper_id") or explicit_paper_id or report_stem_from_paper_path(args.state_dir)

    LOG.info("Stage: judge | adjudicate matches")
    strategy = RAGComparisonStrategy(
        client=client, model=args.model, dimensions=DEFAULT_DIMENSIONS, reasoning_effort=args.reasoning_effort
    )
    matches = []
    for analysis in analyses:
        for dim, definition in DEFAULT_DIMENSIONS.items():
            paper_item = paper_evidence.get((analysis.analysis_id, dim)) or {}
            code_item = code_evidence.get((analysis.analysis_id, dim)) or {}
            paper_val = paper_item.get("value", "")
            code_val = code_item.get("value", "")
            m = strategy.judge_dimension(analysis, dim, definition, paper_val, code_val)
            merged_evidence = m.evidence if isinstance(m.evidence, dict) else {}
            paper_meta = paper_item.get("evidence") if isinstance(paper_item.get("evidence"), dict) else {}
            code_meta = code_item.get("evidence") if isinstance(code_item.get("evidence"), dict) else {}
            if not merged_evidence.get("paper_span"):
                merged_evidence["paper_span"] = paper_meta.get("paper_span", "") or paper_val
            if not merged_evidence.get("location"):
                merged_evidence["location"] = paper_meta.get("location", "")
            if not merged_evidence.get("code_path"):
                merged_evidence["code_path"] = code_meta.get("code_path", "")
            if not merged_evidence.get("code_lines"):
                merged_evidence["code_lines"] = code_meta.get("code_lines", "")
            if not merged_evidence.get("code_line_ranges"):
                merged_evidence["code_line_ranges"] = code_meta.get("code_line_ranges", "")
            if not merged_evidence.get("informative_code_lines"):
                merged_evidence["informative_code_lines"] = code_meta.get("informative_code_lines", [])
            if not merged_evidence.get("screener_output"):
                merged_evidence["screener_output"] = code_meta.get("screener_output", {})
            m.evidence = merged_evidence
            m.paper_id = paper_id
            matches.append(m)

    LOG.info("Stage: judge | write reports")
    comparisons = to_comparison_records(analyses, matches)
    out_dir = Path(args.output_dir)
    write_per_paper(comparisons, paper_id, out_dir)
    write_intermediates(paper_id, paper_evidence=list(paper_evidence.values()), code_evidence=list(code_evidence.values()), matches=matches, output_dir=out_dir)
    print(f"Wrote adjudications + reports to {out_dir}")


# ----------------- Main CLI -----------------

def run_single_cmd(args) -> None:
    LOG.info("Run: single | pipeline start")
    cfg = RunConfig(
        paper_path=Path(args.paper),
        repo_url=args.repo_url,
        branch=args.branch,
        github_token=args.github_token,
        extensions=args.extensions,
        paper_id=args.paper_id,
        parser=args.parser,
        mode=args.mode,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=Path(args.output_dir),
        keep_intermediates=not args.no_intermediates,
    )
    run_single(cfg)
    LOG.info("Run: single | pipeline complete")


def run_multi_cmd(args) -> None:
    LOG.info("Run: multi | pipeline start")
    pairs_csv_path = _resolve_pairs_csv_path(args.pairs_csv)
    if not pairs_csv_path.exists():
        raise SystemExit(f"pairs specification CSV not found: {pairs_csv_path}")

    rows: List[dict] = []
    with open(pairs_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        required = {"paper_id", "github_url"}
        if not required.issubset(columns):
            raise SystemExit("pairs specification CSV must include columns: paper_id, github_url")

        for row in reader:
            paper_id_val = (row.get("paper_id") or "").strip()
            repo_url = (row.get("github_url") or "").strip() or None
            if not repo_url:
                raise SystemExit("Each row in pairs specification CSV must include github_url")
            try:
                paper_path = _resolve_paper_pdf_path(paper_id_val, papers_dir=DEFAULT_PAPERS_DIR)
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            rows.append(
                {
                    "paper_path": str(paper_path),
                    "repo_url": repo_url,
                    # Use PDF-derived report names by default.
                    "paper_id": report_stem_from_paper_path(paper_path),
                    "branch": (row.get("branch") or args.branch),
                }
            )
    run_multi(
        rows,
        parser=args.parser,
        mode=args.mode,
        branch=args.branch,
        github_token=args.github_token,
        extensions=args.extensions,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=Path(args.output_dir),
        keep_intermediates=not args.no_intermediates,
        parallelism=args.parallelism,
    )
    LOG.info("Run: multi | pipeline complete")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CodeBot modular CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # run-single
    single = sub.add_parser("run-single", help="Run pipeline for one paper + GitHub repository")
    single.add_argument("--paper", required=True, help="Path to PDF or text file")
    _add_repo_source_args(single)
    _add_common_parser_args(single)
    single.set_defaults(func=run_single_cmd)

    # run-multi
    multi = sub.add_parser("run-multi", help="Run pipeline for multiple pairs from CSV")
    multi.add_argument(
        "--pairs-csv",
        default=None,
        help="Optional CSV path (otherwise searches papers/ for pairs-specification.csv) with columns paper_id,github_url[,branch]",
    )
    multi.add_argument("--branch", default="main", help="Default branch for rows using repo_url (row value overrides)")
    multi.add_argument("--github-token", help="GitHub token (optional; falls back to env vars)")
    multi.add_argument("--extensions", nargs="*", default=None, help="Extensions to include from repositories")
    multi.add_argument("--parallelism", type=int, default=2, help="Number of paper-code pairs to run in parallel")
    _add_common_parser_args(multi)
    multi.set_defaults(mode="staged", func=run_multi_cmd)

    # staged commands
    stage = sub.add_parser("stage", help="Run an individual stage (paper/code/judge)")
    stage.add_argument("stage", choices=["paper", "code", "judge"], help="Stage to execute")
    stage.add_argument("--paper", help="Path to paper (required for stage paper)")
    stage.add_argument("--repo-url", help="GitHub repository URL (required for stage code)")
    stage.add_argument("--branch", default="main", help="GitHub branch to ingest (for stage code)")
    stage.add_argument("--github-token", help="GitHub token (optional; falls back to env vars)")
    stage.add_argument("--extensions", nargs="*", default=None, help="Extensions to include from code files")
    stage.add_argument("--state-dir", default=".codebot_state", help="Where to persist intermediate JSON between stages")
    stage.add_argument("--paper-id", help="Paper identifier")
    stage.add_argument("--parser", default="grobid", choices=["grobid", "dpt2", "pypdf", "text"], help="Paper parsing backend")
    stage.add_argument("--model", default="gpt-5.1", help="OpenAI model to use")
    stage.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="OpenAI reasoning_effort setting",
    )
    stage.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Final report destination (judge stage)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="CLI log level")

    def stage_dispatch(args):
        if args.stage == "paper":
            if not args.paper:
                raise SystemExit("--paper is required for stage paper")
            stage_paper(args)
        elif args.stage == "code":
            if not args.repo_url:
                raise SystemExit("--repo-url is required for stage code")
            stage_code(args)
        else:
            stage_judge(args)

    stage.set_defaults(func=stage_dispatch)

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
