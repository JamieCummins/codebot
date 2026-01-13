import json
import re
from typing import List, Optional

from codebot.config import OpenAISettings
from codebot.models import PaperAnalysisIR
from codebot.utils import normalize_space


def detect_model_family_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["logistic", "logit", "odds ratio", "or "]):
        return "logistic"
    if any(k in t for k in ["cox", "hazard ratio", "survival", "time-to-event"]):
        return "cox"
    if "t-test" in t or "t test" in t:
        return "t-test"
    if "chi-square" in t or "χ2" in t or "chi2" in t:
        return "chi-square"
    if "poisson" in t:
        return "poisson"
    if "propensity" in t:
        return "psm"
    if any(k in t for k in ["count", "mean", "median", "sd", "standard deviation", "central tendency"]):
        return "counts/ct"
    return None


def _parse_extraction_json(raw: str) -> list[dict]:
    try:
        return json.loads(raw)
    except Exception:
        arr = []
        for i, line in enumerate(raw.splitlines()):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0].isdigit():
                arr.append(
                    {
                        "analysis_id": int(parts[0]),
                        "analysis_description": parts[1],
                        "location": parts[2],
                    }
                )
        return arr


def extract_paper_analyses_as_json(
    paper_text: str,
    client,
    settings: OpenAISettings | None = None,
) -> List[PaperAnalysisIR]:
    """
    Preferred paper analysis extraction: asks the LLM for JSON rows.
    Falls back to light parsing if JSON is not returned.
    """
    settings = settings or OpenAISettings()
    extraction_prompt = (
        "You are an extraction assistant. Extract ALL distinct statistical analyses from the paper.\n"
        "Treat enumerations (A,B,C) × (X,Y,Z) as separate analyses.\n"
        "Return STRICT JSON (no prose) as a list of objects with keys:\n"
        "  analysis_id (int starting at 1),\n"
        "  analysis_description (string),\n"
        "  location (string, e.g., 'methods paragraph 12' or 'table 2').\n"
        "Do not include any keys other than these three. Output must be valid JSON.\n\n"
        f"Paper text:\n{paper_text}"
    )

    resp = client.chat.completions.create(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": extraction_prompt},
        ],
    )

    rows = _parse_extraction_json(resp.choices[0].message.content)

    paper_irs: List[PaperAnalysisIR] = []
    for obj in rows:
        pid = f"P-{int(obj['analysis_id']):03d}"
        desc = normalize_space(obj.get("analysis_description", ""))
        loc = normalize_space(obj.get("location", ""))
        family = detect_model_family_from_text(desc)
        vars_hint = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", desc)))[:12]
        paper_irs.append(
            PaperAnalysisIR(
                analysis_id=pid,
                analysis_description=desc,
                location=loc,
                model_family_hint=family,
                variables_hint=vars_hint,
            )
        )
    return paper_irs


def extract_analysis_details(
    paper_text: str,
    extracted_analyses: str,
    client,
    settings: OpenAISettings | None = None,
) -> dict:
    """
    Two-step detail extractor mirroring the notebook logic.
    """
    settings = settings or OpenAISettings()
    analysis_detail_prompt = (
        "Here is a list of analyses extracted from a research paper. "
        "Your task is to extract structured information about each analysis from the paper below, "
        "specifically direct quotes from the paper addressing each dimension.\n"
        f"{extracted_analyses}\n"
        "You should extract quotes related to the following information:\n"
        "analysis_specification: specific statistical test used;\n"
        "variable_specification: variables used and their designation (e.g., outcome, predictor, control);\n"
        "parameter_specification: parameters set for the analysis (e.g., assumptions);\n"
        "inference_specification: pre-specified inference criteria (e.g., alpha = 0.05).\n"
        "Output JSON with entries per analysis: "
        "analysis_id, analysis_specification, variable_specification, parameter_specification, inference_specification, location.\n"
        "If any information is missing state 'missing'. Also flag extra analyses not in the list.\n"
        "Return detailed direct quotes. Be comprehensive.\n"
        f"Here is the paper:\n{paper_text}"
    )

    resp = client.chat.completions.create(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        messages=[
            {
                "role": "system",
                "content": "You are an extraction assistant that specialises in identifying detailed information about analyses conducted within academic paper texts.",
            },
            {"role": "user", "content": analysis_detail_prompt},
        ],
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"raw": resp.choices[0].message.content}


def extract_onestep_analysis_details(
    paper_text: str,
    client,
    settings: OpenAISettings | None = None,
) -> str:
    """
    One-step extractor that returns a markdown table-like string, mirroring the notebook's alternate approach.
    """
    settings = settings or OpenAISettings()
    onestep_prompt = (
        "Below is the text from an academic paper."
        "Please extract all analyses. Treat enumerations as distinct analyses."
        "Extract analyses mentioned in-text (methods/results) and in tables."
        "For each analysis, extract: analysis_specification, variable_specification, parameter_specification, "
        "inference_specification, location."
        "Output a table with columns: analysis_id, analysis_specification, variable_specification, "
        "parameter_specification, inference_specification, location."
        "If any information is missing then state 'missing'."
        f"Here is the paper:\n{paper_text}"
    )

    resp = client.chat.completions.create(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        messages=[
            {
                "role": "system",
                "content": "You are an extraction assistant that specialises in identifying analyses conducted within academic paper texts.",
            },
            {"role": "user", "content": onestep_prompt},
        ],
    )

    return resp.choices[0].message.content.strip()

