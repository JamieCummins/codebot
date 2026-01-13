import csv
import json
from pathlib import Path
from typing import Iterable


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_from_comparisons(comparisons: Iterable[dict], path: str | Path) -> None:
    """
    Flattens comparison rows (paper_id/code_id/dimension_diffs) into a CSV.
    """
    rows = []
    for comp in comparisons:
        for diff in comp.get("dimension_diffs", []):
            evidence = diff.get("evidence", {})
            rows.append(
                {
                    "paper_id": comp.get("paper_id"),
                    "code_id": comp.get("code_id"),
                    "dimension": diff.get("dimension"),
                    "status": diff.get("status"),
                    "explanation": diff.get("explanation"),
                    "code_file": evidence.get("file"),
                    "code_lines": evidence.get("lines"),
                }
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["paper_id", "code_id", "dimension", "status", "explanation", "code_file", "code_lines"],
        )
        writer.writeheader()
        writer.writerows(rows)

