# CodeBot

An LLM-powered pipeline for comparing statistical analyses described in academic papers with their implementations in GitHub repositories.

CodeBot extracts analyses from papers, mines code for statistical patterns, and uses LLM reasoning to identify matches and compare them across multiple dimensions.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `openai>=1.3.0` - OpenAI API client
- `requests>=2.31.0` - HTTP library for GitHub API and PDF parser endpoints

### Environment Variables

Set API keys via environment variables (checked in order of preference):

| Service | Environment Variables |
|---------|----------------------|
| OpenAI | `CODEBOT_OPENAI_API_KEY` → `OPENAI_API_KEY` |
| GitHub | `GITHUB_TOKEN` → `CODEBOT_GITHUB_TOKEN` |
| Landing.ai (DPT-2) | `DPT2_API_KEY` → `LANDINGAI_API_TOKEN` → `CODEBOT_LANDING_TOKEN` |

## Usage

### Basic Usage

```bash
python main.py \
  --paper-path paper.pdf \
  --repo-url https://github.com/user/repo
```

### With Pre-filtering (Faster)

```bash
python main.py \
  --paper-path paper.pdf \
  --repo-url https://github.com/user/repo \
  --use-matching \
  --min-score 0.3
```

### Full-context Mode (More Thorough)

```bash
python main.py \
  --paper-path paper.pdf \
  --repo-url https://github.com/user/repo \
  --parser dpt2 \
  --model gpt-5
```

### Command-line Options

#### PDF Parsing
| Option | Default | Description |
|--------|---------|-------------|
| `--paper-path` | *required* | Path to the PDF to parse |
| `--parser` | `grobid` | PDF parser: `grobid` or `dpt2` |
| `--grobid-url` | HuggingFace endpoint | Grobid API endpoint URL |
| `--dpt2-endpoint` | Landing.ai default | Override DPT-2 endpoint |
| `--dpt2-model` | - | Override DPT-2 model name |
| `--parser-token` | - | Explicit parser API token |

#### Repository
| Option | Default | Description |
|--------|---------|-------------|
| `--repo-url` | *required* | GitHub repository URL |
| `--branch` | `main` | Repository branch to read from |
| `--extensions` | R-centric set | File extensions to include |
| `--github-token` | - | GitHub authentication token |

#### LLM Configuration
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-5` | OpenAI model to use |
| `--reasoning` | `medium` | Reasoning effort level |
| `--openai-key` | - | Explicit OpenAI API key |

#### Matching Strategy
| Option | Default | Description |
|--------|---------|-------------|
| `--use-matching` | off | Enable paper↔code matching before LLM comparison |
| `--top-k` | `3` | Top-k code candidates per paper analysis |
| `--min-score` | `0.35` | Minimum score for greedy matching |

#### Output
| Option | Default | Description |
|--------|---------|-------------|
| `--dimensions-path` | built-in | Path to JSON file with comparison dimensions |
| `--output-json` | `codebot_run_results.json` | JSON results output path |
| `--output-csv` | `codebot_report.csv` | CSV report output path |
| `--skip-csv` | off | Skip writing CSV report |

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PDF Parsing   │────▶│ Paper Analysis  │────▶│   Relevance     │
│ (Grobid/DPT-2)  │     │   Extraction    │     │ Classification  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│   Repository    │────▶│   Code Mining   │              │
│   Ingestion     │     │  (8 patterns)   │              │
└─────────────────┘     └────────┬────────┘              │
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────────────────────┐
                        │      Paper ↔ Code Matching      │
                        │   (optional, --use-matching)    │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Dimension-wise Comparison     │
                        │          (LLM-based)            │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │     JSON + CSV Output           │
                        └─────────────────────────────────┘
```

### Two Comparison Modes

1. **Full-context mode** (default): Sends entire paper text and codebase to the LLM for each relevant analysis. More thorough but slower and more expensive.

2. **Pre-filtered mode** (`--use-matching`): Mines code for statistical patterns, scores similarity between paper and code analyses, then only compares matched pairs. Faster and cheaper.

## Comparison Dimensions

CodeBot compares paper and code across five dimensions:

| Dimension | Description |
|-----------|-------------|
| **Test Specification** | Statistical test type (e.g., logistic regression, Hazard Ratio) |
| **Variable Specification** | Variables and their roles (outcome, predictor, control) |
| **Parameter Specification** | Analysis parameters (e.g., equal groups assumption) |
| **Inference Specification** | Pre-specified criteria (e.g., alpha=0.05, confidence intervals) |
| **Coding Specification** | Variable encoding schemes (e.g., contrast coding) |

Custom dimensions can be provided via `--dimensions-path`.

## Detected Statistical Patterns

The code mining step detects these R statistical functions:

- `glmer(..., family=binomial)` - Mixed-effects logistic regression
- `glm(..., family=binomial)` - Logistic regression
- `glm(..., family=poisson)` - Poisson regression
- `coxph()` - Cox proportional hazards
- `t.test()` - T-tests
- `chisq.test()` - Chi-square tests
- `mean()`, `median()`, `sd()` - Descriptive statistics
- `matchit()` - Propensity score matching

## Output

### JSON Results

```json
{
  "meta": {
    "version": "0.1",
    "timestamp": "2024-01-15T10:30:00Z",
    "parser": "grobid",
    "repo_url": "https://github.com/user/repo",
    "use_matching": true,
    "num_paper_analyses": 12,
    "num_code_analyses": 8,
    "num_matches": 5,
    "num_comparisons": 25
  },
  "paper_analyses": [...],
  "code_analyses": [...],
  "paper_relevance": {"P-001": "relevant", ...},
  "repo_tree": "...",
  "matches": [...],
  "comparisons": [
    {
      "paper_id": "P-001",
      "code_id": "C-005",
      "match_score": 0.68,
      "dimension_diffs": [
        {
          "dimension": "Test Specification",
          "status": "match",
          "explanation": "Both use logistic regression...",
          "evidence": {...}
        }
      ]
    }
  ]
}
```

### CSV Report

Flattened format with one row per paper × code × dimension:

| paper_id | code_id | dimension | status | explanation | code_file | code_lines |
|----------|---------|-----------|--------|-------------|-----------|------------|
| P-001 | C-005 | Test Specification | match | Both use... | analysis.R | 45-60 |

## Project Structure

```
codebot/
├── main.py                 # CLI entrypoint
├── requirements.txt
├── CodeBot_flow.ipynb      # Original notebook implementation
└── codebot/
    ├── analysis/
    │   ├── extraction.py   # Paper analysis extraction (LLM)
    │   └── classification.py # Relevance classification
    ├── comparison/
    │   ├── matchers.py     # Code mining and paper↔code matching
    │   └── dimension_compare.py # Dimension-wise LLM comparison
    ├── ingestion/
    │   └── github_repo.py  # GitHub repository fetching
    ├── parsing/
    │   ├── grobid.py       # Grobid PDF parser
    │   └── landing_ai.py   # DPT-2 PDF parser
    ├── reporting/
    │   └── export.py       # JSON and CSV output
    ├── config.py           # Configuration and defaults
    ├── models.py           # Data structures
    └── utils.py            # Utility functions
```

## License

[Add license information here]
