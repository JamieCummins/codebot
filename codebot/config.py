"""Configuration utilities for CodeBot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServiceConfig:
    """Service configuration for external APIs."""

    landing_ai_token: str
    openai_api_key: str
    github_token: Optional[str]
    landing_ai_endpoint: str = "https://api.va.eu-west-1.landing.ai/v1/ade/parse"
    openai_endpoint: str = "https://api.openai.com/v1/chat/completions"
    openai_model: str = "gpt-5"
    request_timeout: int = 60
    max_retries: int = 3


def load_service_config() -> ServiceConfig:
    """Load configuration from environment variables.

    Returns
    -------
    ServiceConfig
        Populated configuration dataclass.

    Raises
    ------
    EnvironmentError
        If required tokens are missing.
    """

    landing_ai_token = os.getenv("LANDING_AI_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")

    if not landing_ai_token:
        raise EnvironmentError("LANDING_AI_TOKEN environment variable is required.")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required.")

    return ServiceConfig(
        landing_ai_token=landing_ai_token,
        openai_api_key=openai_api_key,
        github_token=github_token,
    )


MODEL_FAMILY_ALLOWLIST = {
    "logistic",
    "cox",
    "psm",
    "t-test",
    "chi-square",
    "poisson",
    "counts/ct",
}


def get_optional_env(key: str) -> Optional[str]:
    """Retrieve optional environment variables.

    Parameters
    ----------
    key:
        Environment variable name.

    Returns
    -------
    Optional[str]
        Value if present, otherwise ``None``.
    """

    return os.getenv(key)
