"""Relevance classification via LLM."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from .config import MODEL_FAMILY_ALLOWLIST, ServiceConfig, load_service_config
from .schemas import PaperAnalysisIR
from .utils import http_post


class ClassificationError(RuntimeError):
    """Raised when relevance classification fails."""


def _call_openai(papers: List[PaperAnalysisIR], config: ServiceConfig) -> Dict[str, str]:
    prompt_items = [
        {
            "id": paper.analysis_id,
            "description": paper.analysis_description,
            "model_family_hint": paper.model_family_hint,
        }
        for paper in papers
    ]
    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.openai_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify whether each paper analysis is relevant to statistical models "
                    "from the allowlist: logistic, cox, psm, t-test, chi-square, poisson, counts/ct."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_items),
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
        raise ClassificationError("Malformed OpenAI response") from error
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ClassificationError("OpenAI returned non-JSON content") from error
    if not isinstance(data, dict):
        raise ClassificationError("OpenAI response must be an object")
    return {str(key): str(value) for key, value in data.items()}


def classify_relevance(
    papers: Iterable[PaperAnalysisIR],
    *,
    config: Optional[ServiceConfig] = None,
) -> Dict[str, str]:
    """Classify analyses as relevant or irrelevant."""

    config = config or load_service_config()
    papers_list = list(papers)
    try:
        llm_results = _call_openai(papers_list, config)
    except Exception:
        llm_results = {}

    classifications: Dict[str, str] = {}
    for paper in papers_list:
        label = llm_results.get(paper.analysis_id)
        if label is None:
            if paper.model_family_hint and paper.model_family_hint in MODEL_FAMILY_ALLOWLIST:
                label = "relevant"
            else:
                label = "irrelevant"
        classifications[paper.analysis_id] = label
    return classifications
