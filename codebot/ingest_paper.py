"""Paper ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from .config import ServiceConfig, load_service_config
from .schemas import PaperAnalysisIR
from .utils import http_post, normalize_text, split_identifiers

PAPER_EXTRACTION_PROMPT = (
    "You are an expert research assistant. Given the paper content, "
    "identify every statistical analysis described. Respond with STRICT JSON "
    "formatted as a list of objects with keys analysis_id (integer starting at 1), "
    "analysis_description, and location."
)


class PaperExtractionError(RuntimeError):
    """Raised when paper extraction fails."""


def parse_pdf_with_dpt2(
    path_or_bytes: str | Path | bytes,
    *,
    config: Optional[ServiceConfig] = None,
) -> str:
    """Parse a PDF into text using Landing AI's DPT-2 service."""

    config = config or load_service_config()

    if isinstance(path_or_bytes, (str, Path)):
        pdf_bytes = Path(path_or_bytes).read_bytes()
    else:
        pdf_bytes = path_or_bytes

    headers = {
        "Authorization": f"Bearer {config.landing_ai_token}",
        "Content-Type": "application/pdf",
    }
    response = http_post(
        config.landing_ai_endpoint,
        headers=headers,
        data=pdf_bytes,
        timeout=config.request_timeout,
    )
    payload = response.json()
    try:
        return payload["text"]
    except KeyError as error:
        raise PaperExtractionError("Landing AI response missing 'text'") from error


def _call_openai_for_paper(text: str, config: ServiceConfig) -> List[dict]:
    """Invoke OpenAI Chat Completions to extract paper analyses."""

    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.openai_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You extract structured data from documents."},
            {"role": "user", "content": PAPER_EXTRACTION_PROMPT + "\n\n" + text},
        ],
        "response_format": {"type": "json_object"},
    }
    response = http_post(
        config.openai_endpoint,
        headers=headers,
        json_body=body,
        timeout=config.request_timeout,
    )
    parsed = response.json()
    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise PaperExtractionError("OpenAI response missing content") from error
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise PaperExtractionError("OpenAI returned non-JSON content") from error
    if isinstance(data, dict):
        # Response format json_object returns dictionary; look for list
        analyses = data.get("analyses") or data.get("data") or data.get("items")
        if analyses is None:
            # maybe dictionary is list? fallback
            if all(key.isdigit() for key in data.keys()):
                analyses = list(data.values())
        data = analyses if analyses is not None else []
    if not isinstance(data, list):
        raise PaperExtractionError("OpenAI response not list-like")
    return data


def _derive_model_family(description: str) -> Optional[str]:
    lowered = description.lower()
    mapping = {
        "logistic": ["logistic", "logit", "odds ratio", "or"],
        "cox": ["cox", "hazard", "survival"],
        "t-test": ["t-test", "ttest", "t test"],
        "chi-square": ["chi-square", "χ2", "chi2", "chisq"],
        "poisson": ["poisson"],
        "psm": ["propensity", "match"],
        "counts/ct": ["count", "mean", "median", "sd", "standard deviation"],
    }
    for family, keywords in mapping.items():
        if any(keyword in lowered for keyword in keywords):
            return family
    return None


def extract_paper_analyses(
    dpt2_text: str,
    *,
    config: Optional[ServiceConfig] = None,
) -> List[PaperAnalysisIR]:
    """Extract structured analyses from parsed paper text."""

    config = config or load_service_config()
    try:
        raw_items = _call_openai_for_paper(dpt2_text, config)
    except PaperExtractionError:
        raw_items = []

    paper_irs: List[PaperAnalysisIR] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        analysis_description = normalize_text(str(item.get("analysis_description") or item.get("description") or ""))
        if not analysis_description:
            continue
        location = normalize_text(str(item.get("location") or "")) or "unknown"
        analysis_id = f"P-{index:03d}"
        model_family_hint = _derive_model_family(analysis_description)
        variables_hint = split_identifiers(analysis_description)
        try:
            paper_ir = PaperAnalysisIR(
                analysis_id=analysis_id,
                analysis_description=analysis_description,
                location=location,
                model_family_hint=model_family_hint,
                variables_hint=variables_hint,
            )
            paper_irs.append(paper_ir)
        except ValidationError:
            continue

    return paper_irs
