from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import ComparisonRecord, DimensionEvidence, MatchDecision


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(records: Sequence[ComparisonRecord], path: Path) -> None:
    ensure_dir(path.parent)
    def _stringify(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [_stringify(v).strip() for v in value]
            parts = [p for p in parts if p]
            return "\n".join(parts)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _extract_evidence(evidence, key: str) -> str:
        if isinstance(evidence, dict):
            if evidence.get(key) is not None:
                return _stringify(evidence.get(key))
            if key == "code_lines":
                if evidence.get("code_line_ranges") is not None:
                    return _stringify(evidence.get("code_line_ranges"))
                if evidence.get("informative_code_lines") is not None:
                    return _stringify(evidence.get("informative_code_lines"))
            return ""
        if isinstance(evidence, list):
            vals = []
            for item in evidence:
                if isinstance(item, dict) and item.get(key):
                    vals.append(_stringify(item.get(key)))
            return "\n\n".join(v for v in vals if v.strip())
        return ""

    fieldnames = [
        "paper_id",
        "analysis_id",
        "brief_description",
        "dimension",
        "paper_text",
        "paper_location",
        "code_text",
        "code_file",
        "code_lines",
        "match_status",
        "explanation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "paper_id": r.paper_id,
                    "analysis_id": r.analysis_id,
                    "brief_description": r.brief_description,
                    "dimension": r.dimension,
                    "paper_text": r.paper_value,
                    "paper_location": _extract_evidence(r.evidence, "location"),
                    "code_text": r.code_value,
                    "code_file": _extract_evidence(r.evidence, "code_path"),
                    "code_lines": _extract_evidence(r.evidence, "code_lines"),
                    "match_status": r.match_status,
                    "explanation": r.explanation,
                }
            )


def write_per_paper(records: Sequence[ComparisonRecord], paper_id: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    write_json([r.to_dict() for r in records], output_dir / f"{paper_id}.json")
    write_csv(records, output_dir / f"{paper_id}.csv")


def write_intermediates(
    paper_id: str,
    *,
    paper_evidence: Sequence[DimensionEvidence] | None,
    code_evidence: Sequence[DimensionEvidence] | None,
    matches: Sequence[MatchDecision] | None,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    to_dict = lambda obj: obj.__dict__ if hasattr(obj, "__dict__") else obj
    if paper_evidence is not None:
        write_json([to_dict(e) for e in paper_evidence], output_dir / f"{paper_id}_paper_evidence.json")
    if code_evidence is not None:
        write_json([to_dict(e) for e in code_evidence], output_dir / f"{paper_id}_code_evidence.json")
    if matches is not None:
        write_json([to_dict(m) for m in matches], output_dir / f"{paper_id}_matches.json")


def write_aggregate(records: Iterable[ComparisonRecord], path: Path) -> None:
    records_list = list(records)
    write_json([r.to_dict() for r in records_list], path.with_suffix(".json"))
    write_csv(records_list, path.with_suffix(".csv"))


__all__ = [
    "write_per_paper",
    "write_aggregate",
    "write_intermediates",
    "write_json",
    "write_csv",
]
