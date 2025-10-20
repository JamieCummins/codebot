"""Command-line interface for CodeBot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from .config import ServiceConfig, load_service_config
from .run import WorkflowError, run_workflow
from .schemas import (
    CodeAnalysisIR,
    DimensionDiff,
    MatchEdge,
    PaperAnalysisIR,
    RunResults,
)

console = Console()


def _load_config() -> ServiceConfig:
    return load_service_config()


@click.group()
@click.version_option("0.1.0")
def main() -> None:
    """CodeBot CLI entry point."""


@main.command("run")
@click.option("--pdf", "pdf_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--repo", "repo_url", required=True, type=str)
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False))
@click.option("--branch", default="main", show_default=True)
@click.option("--min-score", default=0.35, show_default=True, type=float)
@click.option("--model", "model_name", default=None)
def run_command(
    pdf_path: str,
    repo_url: str,
    out_path: str,
    branch: str,
    min_score: float,
    model_name: Optional[str],
) -> None:
    """Run the full CodeBot workflow."""

    try:
        run_workflow(
            pdf_path=pdf_path,
            repo_url=repo_url,
            out_path=out_path,
            branch=branch,
            min_score=min_score,
            model_name=model_name,
        )
    except WorkflowError as error:
        raise click.ClickException(str(error)) from error


@main.command("schema")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False))
def schema_command(out_dir: str) -> None:
    """Export JSON schemas for CodeBot models."""

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schemas = {
        "paper_analysis.json": PaperAnalysisIR.json_schema(),
        "code_analysis.json": CodeAnalysisIR.json_schema(),
        "match_edge.json": MatchEdge.json_schema(),
        "dimension_diff.json": DimensionDiff.json_schema(),
        "run_results.json": RunResults.json_schema(),
    }
    for filename, schema in schemas.items():
        (output_path / filename).write_text(json.dumps(schema, indent=2))
    console.print(f"Wrote {len(schemas)} schemas to {output_path}")


@main.command("ping")
def ping_command() -> None:
    """Verify that required API keys are available."""

    try:
        config = _load_config()
    except Exception as error:
        raise click.ClickException(f"Configuration error: {error}") from error

    status_lines = [
        f"Landing AI token: {'set' if config.landing_ai_token else 'missing'}",
        f"OpenAI API key: {'set' if config.openai_api_key else 'missing'}",
        f"GitHub token: {'set' if config.github_token else 'missing'}",
    ]
    for line in status_lines:
        console.print(line)


if __name__ == "__main__":
    main()
