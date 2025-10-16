# CodeBot

CodeBot aligns statistical analyses described in research papers with the code that implements them. Given a PDF and a GitHub repository URL, CodeBot parses the paper, mines the repository for analytical code, matches analyses, evaluates relevance, and compares implementations dimension-by-dimension.

## Features

- Landing AI DPT-2 integration for PDF parsing
- OpenAI GPT-5 integration for structured extraction, relevance classification, and comparison
- Regex-based mining for R, Python, and Stata analytical patterns
- Greedy bipartite matching with explainable scoring
- JSON and Rich CLI outputs with artifact exports

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Create a `.env` file (see `.env.example`) and export the required environment variables:

```bash
export LANDING_AI_TOKEN=...  # Landing AI ADE token
export OPENAI_API_KEY=...    # OpenAI API key with GPT-5 access
export GITHUB_TOKEN=...      # Optional GitHub token for higher rate limits
```

Run CodeBot:

```bash
codebot run --pdf paper.pdf --repo https://github.com/org/repo --out codebot_run_results.json --branch main --min-score 0.35
```

This command generates:

- `codebot_run_results.json` – consolidated run results
- `artifacts/paper_analyses.json`
- `artifacts/code_analyses.json`
- `artifacts/matches.json`
- `artifacts/comparisons.json`

Use the schema export for downstream integrations:

```bash
codebot schema --out schemas/
```

Verify configuration:

```bash
codebot ping
```

## JSON Report Structure

- `meta`: run metadata (version, model)
- `paper_analyses`: list of `PaperAnalysisIR`
- `code_analyses`: list of `CodeAnalysisIR`
- `matches`: selected `MatchEdge` entries with scoring reasons
- `comparisons`: per-match comparison results with `DimensionDiff` entries

See exported schemas for field-level details.

## Development

- Formatting relies on standard library modules
- Tests rely on mocked HTTP requests to avoid network access: `pytest`
- The CLI prints a summary table and exits non-zero when no analyses or matches are found

## License

MIT. See [LICENSE](LICENSE).
