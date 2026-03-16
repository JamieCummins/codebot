# CodeBot (canonical)

Canonical package combining:
- `codebot_new` workflow/architecture improvements (modular runner + CLI, staged and combined flows, multi-run).
- `codebot_old` domain-specific behavior (medical-analysis focus and GitHub repo tree/path ingestion).

## Core behavior

- Analysis relevance is focused on medical/biomedical statistical analyses (e.g., logistic/Cox/survival, propensity matching, Poisson, chi-square, t-tests, descriptive central-tendency summaries).
- Expected inputs are manuscript PDF + GitHub URL (`--repo-url` + optional `--branch`).
- GitHub tree paths and `project.yaml` (if present) are included in code comparison context.
- Quote extraction is RAG-based: paper/code are embedded once per run, retrieved by analysis+dimension query, and code snippets are screened by an LLM appropriateness check before use.
- Default report output directory is `codebot-reports/`.
- Per-paper report files are named from manuscript PDF filename stems (e.g., `my_manuscript.json`, `my_manuscript.csv`).

## Single run

```bash
python -m codebot.cli run-single \
  --paper /abs/path/paper.pdf \
  --repo-url https://github.com/org/repo \
  --branch main \
  --parser grobid \
  --mode staged
```

## Multi run

By default, `run-multi` searches `papers/` for `pairs-specification.csv` (using `papers/pairs-specification.csv` when present).

The CSV must include:
- `paper_id`: PDF filename in `papers/` (with or without `.pdf`)
- `github_url`: GitHub repository URL
- Optional: `branch`

```csv
paper_id,github_url,branch
paper_a.pdf,https://github.com/org/repo-a,main
paper_b,https://github.com/org/repo-b,dev
```

```bash
python -m codebot.cli run-multi
```

Default parallelism is `2`. You can override it with `--parallelism`.
Default comparison mode for `run-multi` is `staged` (override with `--mode combined`).

## Staged run

```bash
python -m codebot.cli stage paper --paper /abs/path/paper.pdf --state-dir .codebot_state/p1
python -m codebot.cli stage code --state-dir .codebot_state/p1 --repo-url https://github.com/org/repo
python -m codebot.cli stage judge --state-dir .codebot_state/p1
```
