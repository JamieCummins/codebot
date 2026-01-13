from __future__ import annotations

import json
from typing import Callable, Dict, List

from codebot.config import OpenAISettings
from codebot.models import PaperAnalysisIR

CLASSIFIER_FUNCTION_SPEC = [
    {
        "name": "classify_analysis",
        "description": (
            "Classify whether the analysis is 'relevant' or 'irrelevant'. Relevant analyses are any one of:"
            " (i) logistic regression, (ii) survival analysis, (iii) propensity score matching,"
            " (iv) t-tests, (v) chi-square tests, (vi) Poisson regression, and (vii) counts/central tendency measures."
        ),
        "parameters": {
            "type": "object",
            "properties": {"object": {"type": "string", "enum": ["relevant", "irrelevant"]}},
            "required": ["object"],
        },
    }
]


def classify_paper_relevance(
    paper_analyses: List[PaperAnalysisIR],
    client,
    settings: OpenAISettings | None = None,
    progress: Callable | None = None,
) -> Dict[str, str]:
    settings = settings or OpenAISettings()
    results: Dict[str, str] = {}
    for idx, pa in enumerate(paper_analyses, start=1):
        if progress:
            progress(f"Classifying relevance {idx}/{len(paper_analyses)} ({pa.analysis_id})")
        user_prompt_text = json.dumps(
            {"analysis_id": pa.analysis_id, "description": pa.analysis_description},
            ensure_ascii=False,
        )
        resp = client.chat.completions.create(
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            messages=[{"role": "user", "content": "Please classify the following analysis:\n" + user_prompt_text}],
            functions=CLASSIFIER_FUNCTION_SPEC,
            function_call={"name": "classify_analysis"},
        )
        fn_args_raw = resp.choices[0].message.function_call.arguments
        try:
            fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            classification = fn_args.get("object", "irrelevant")
        except Exception:
            classification = "irrelevant"
        results[pa.analysis_id] = classification
    return results
