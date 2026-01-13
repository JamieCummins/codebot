from pathlib import Path
from typing import Optional

import requests

from codebot.config import LandingAISettings, get_landingai_token


def parse_pdf_with_dpt2(
    pdf_path: str | Path,
    *,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """
    Calls the Landing.ai DPT-2 endpoint to parse a PDF into text.

    Note: this requires network access and a valid API token. The caller can pass
    `token` explicitly or set one of the env vars in LandingAISettings.token_envs.
    """
    settings = LandingAISettings()
    endpoint = endpoint or settings.endpoint
    model = model or settings.model
    token = token or get_landingai_token(settings)

    if not token:
        raise RuntimeError(
            f"Missing Landing.ai token. Provide explicitly or set one of: {', '.join(settings.token_envs)}"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    headers = {"Authorization": f"Bearer {token}"}
    data = {"model": model}

    with pdf_path.open("rb") as document:
        files = {"document": document}
        resp = requests.post(endpoint, files=files, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()

    try:
        payload = resp.json()
    except ValueError:
        payload = resp.text

    # Attempt to extract the textual payload; fall back to a stringified JSON object
    if isinstance(payload, dict):
        for key in ("text", "content", "document", "data"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return str(payload)

    if isinstance(payload, list):
        return "\n".join(str(item) for item in payload)

    return str(payload)

