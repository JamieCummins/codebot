"""Dimension comparison between paper analyses and code."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from .config import ServiceConfig, load_service_config
from .schemas import CodeAnalysisIR, DimensionDiff, MatchEdge, PaperAnalysisIR
from .utils import http_post

DIMENSION_DEFINITIONS = {
    "Test Specification": "Does the statistical test or model align between paper and code?",
    "Variable Specification": "Are the described variables present and used similarly?",
    "Parameter Specification": "Are key parameters (e.g., covariates, interactions) consistent?",
    "Inference Specification": "Are inference procedures (confidence intervals, p-values) aligned?",
    "Coding Specification": "Do implementation details (packages, functions) match the description?",
}


class ComparisonError(RuntimeError):
    """Raised when LLM comparison fails."""


def _call_openai(
    *,
    config: ServiceConfig,
    dimension: str,
    paper: PaperAnalysisIR,
    code: CodeAnalysisIR,
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_name or config.openai_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You compare research paper descriptions with code implementations.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "dimension": dimension,
                        "definition": DIMENSION_DEFINITIONS.get(dimension, ""),
                        "paper_analysis": paper.analysis_description,
                        "code_snippet": code.snippet,
                        "code_location": {
                            "file_path": code.file_path,
                            "line_start": code.line_start,
                            "line_end": code.line_end,
                        },
                    }
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    response = http_post(
        config.openai_endpoint,
        headers=headers,
        json_body=body,
        timeout=config.request_timeout,
    )
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise ComparisonError("Malformed OpenAI response") from error
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ComparisonError("OpenAI response not object")
    return data


def compare_dimensions(
    match: MatchEdge,
    paper_lookup: Dict[str, PaperAnalysisIR],
    code_lookup: Dict[str, CodeAnalysisIR],
    *,
    config: Optional[ServiceConfig] = None,
    model_name: Optional[str] = None,
) -> List[DimensionDiff]:
    """Compare matched analyses across dimensions."""

    config = config or load_service_config()
    paper = paper_lookup[match.paper_id]
    code = code_lookup[match.code_id]
    diffs: List[DimensionDiff] = []
    for dimension in DIMENSION_DEFINITIONS.keys():
        try:
            llm_response = _call_openai(
                config=config,
                dimension=dimension,
                paper=paper,
                code=code,
                model_name=model_name,
            )
            status = llm_response.get("status", "unknown")
            explanation = llm_response.get("explanation", "No explanation provided.")
        except Exception:
            status = "unknown"
            explanation = "Comparison failed; marking as unknown."
        diff = DimensionDiff(
            dimension=dimension,
            status=status if status in {"match", "minor", "major", "unknown"} else "unknown",
            explanation=explanation,
            evidence={
                "file_path": code.file_path,
                "line_start": code.line_start,
                "line_end": code.line_end,
            },
        )
        diffs.append(diff)
    return diffs
