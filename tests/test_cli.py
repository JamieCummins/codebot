"""CLI tests with mocked services."""

import json
from pathlib import Path

from click.testing import CliRunner

from codebot import RunResults
from codebot.cli import main


def test_cli_run_invokes_workflow(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"dummy")
    output_path = tmp_path / "output.json"

    def fake_run_workflow(**kwargs):
        Path(kwargs["out_path"]).write_text(json.dumps({"meta": {}}))
        return RunResults(
            meta={"version": "0.1.0"},
            paper_analyses=[],
            code_analyses=[],
            matches=[],
            comparisons=[],
        )

    monkeypatch.setattr("codebot.cli.run_workflow", fake_run_workflow)

    result = runner.invoke(
        main,
        [
            "run",
            "--pdf",
            str(pdf_path),
            "--repo",
            "https://github.com/org/repo",
            "--out",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_schema_exports(tmp_path) -> None:
    runner = CliRunner()
    schema_dir = tmp_path / "schemas"
    result = runner.invoke(main, ["schema", "--out", str(schema_dir)])
    assert result.exit_code == 0
    assert any(schema_dir.iterdir())
