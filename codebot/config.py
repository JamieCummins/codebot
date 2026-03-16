from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import openai

# Canonical medical-focused dimensions retained from legacy CodeBot.
DEFAULT_COMPARISON_DIMENSIONS: dict[str, str] = {
    "Test Specification": "The specific type of statistical test used for this analysis (e.g., logistic regression, Hazard Ratio).",
    "Variable Specification": "The variables used in the analysis, as well as their designation within the analysis (e.g., outcome, predictor, control, etc.).",
    "Parameter Specification": "Any parameters which are set for the analysis (e.g., assumptions of equal groups).",
    "Inference Specification": "Any pre-specified inference criteria used for the analysis (e.g., alpha = 0.05, confidence intervals excluding particular values).",
    "Coding Specification": "Any specifications related to how variables are coded within the analyses (e.g., contrast coding with Intervention = 1 and Control = 0).",
}

DEFAULT_R_EXTENSIONS: set[str] = {
    ".r",
    ".R",
    ".Rmd",
    ".rmd",
    ".qmd",
    ".Qmd",
    ".Rnw",
    ".rnw",
}


@dataclass(frozen=True)
class OpenAISettings:
    model: str = "gpt-5.1"
    reasoning_effort: str = "medium"
    api_key_envs: Tuple[str, ...] = ("CODEBOT_OPENAI_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True)
class GithubSettings:
    token_envs: Tuple[str, ...] = ("GITHUB_TOKEN", "CODEBOT_GITHUB_TOKEN")
    default_branch: str = "main"


@dataclass(frozen=True)
class LandingAISettings:
    endpoint: str = "https://api.va.eu-west-1.landing.ai/v1/ade/parse"
    model: str = "dpt-2-latest"
    token_envs: Tuple[str, ...] = (
        "CODEBOT_DPT2_API_KEY",
        "LANDING_AI_API_KEY",
        "DPT2_API_KEY",
        "LANDINGAI_API_TOKEN",
        "CODEBOT_LANDING_TOKEN",
    )


def first_env(keys: Iterable[str]) -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return ""


def get_openai_api_key(settings: OpenAISettings | None = None) -> str:
    settings = settings or OpenAISettings()
    return first_env(settings.api_key_envs)


def get_openai_client(api_key: str | None = None, settings: OpenAISettings | None = None) -> openai.OpenAI:
    settings = settings or OpenAISettings()
    key = api_key or get_openai_api_key(settings)
    if not key:
        raise RuntimeError(
            f"Missing OpenAI API key. Provide explicitly or set one of: {', '.join(settings.api_key_envs)}"
        )
    return openai.OpenAI(api_key=key)


def get_github_token(settings: GithubSettings | None = None) -> str:
    settings = settings or GithubSettings()
    return first_env(settings.token_envs)


def get_landingai_token(settings: LandingAISettings | None = None) -> str:
    settings = settings or LandingAISettings()
    return first_env(settings.token_envs)


def normalize_dimensions(dimensions: Mapping[str, str] | None) -> dict[str, str]:
    if dimensions is None:
        return dict(DEFAULT_COMPARISON_DIMENSIONS)
    return {str(k): str(v) for k, v in dimensions.items()}
