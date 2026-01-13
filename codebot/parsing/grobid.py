import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests


def pdf_to_grobid_xml(pdf_path: str | Path, grobid_url: str) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    with pdf_path.open("rb") as file:
        files = {"input": file}
        response = requests.post(grobid_url, files=files, timeout=120)

    if response.status_code != 200:
        response.raise_for_status()

    return response.text


def extract_body_text(xml_content: str) -> str:
    namespace = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(xml_content)
    body = root.find(".//tei:body", namespace)
    if body is not None:
        return "".join(body.itertext()).strip()
    return "Body tag not found."


def remove_references(document_text: str) -> str:
    references_pattern = re.compile(
        r"(?:^|\n)([A-Z\\s]*\\bReferences\\b|Bibliography|Cited Works)[\\s]*\\n",
        re.IGNORECASE,
    )
    references_match = references_pattern.search(document_text)
    if references_match:
        return document_text[: references_match.start()]
    return document_text


def clean_document_text(document_text: str) -> str:
    introduction_pattern = re.compile(
        r"(?:^|\n)([A-Z\\s]*\\bIntroduction\\b)[\\s]*\\n", re.IGNORECASE
    )
    introduction_match = introduction_pattern.search(document_text)
    if introduction_match:
        document_text = document_text[introduction_match.start():]
    return remove_references(document_text)


def parse_pdf_with_grobid(
    pdf_path: str | Path,
    *,
    grobid_url: str = "https://kermitt2-grobid.hf.space/api/processFulltextDocument",
    timeout: int = 120,
    precomputed_xml: Optional[str] = None,
) -> str:
    """
    Parses a PDF using Grobid and returns cleaned body text.
    """
    xml_content = precomputed_xml or pdf_to_grobid_xml(pdf_path, grobid_url)
    body = extract_body_text(xml_content)
    return clean_document_text(body)

