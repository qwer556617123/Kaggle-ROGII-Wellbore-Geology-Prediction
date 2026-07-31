"""Summarize pulled Kaggle notebooks for reproducibility and branch auditing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


INTERESTING = re.compile(
    r"(?:submission|public\s*lb|leaderboard|score|rmse|weight|blend|mha|hsmm|"
    r"viterbi|model.package|static|dataset|kernel|runtime|duration)",
    re.IGNORECASE,
)
STATIC_CSV = re.compile(r"/kaggle/input/[^'\"\s]+\.csv", re.IGNORECASE)


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _output_text(cell: dict) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        for key in ("text", "data"):
            value = output.get(key)
            if isinstance(value, dict):
                value = value.get("text/plain", "")
            if isinstance(value, list):
                value = "".join(str(item) for item in value)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def audit(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata_path = notebook_path.with_name("kernel-metadata.json")
    kernel_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    all_source = "\n".join(_source(cell) for cell in cells)
    interesting_lines: list[str] = []
    for line in all_source.splitlines():
        stripped = line.strip()
        if stripped and INTERESTING.search(stripped):
            interesting_lines.append(stripped[:300])
    output_lines: list[str] = []
    for cell in code_cells:
        for line in _output_text(cell).splitlines():
            stripped = line.strip()
            if stripped and INTERESTING.search(stripped):
                output_lines.append(stripped[:300])
    first_lines = []
    for index, cell in enumerate(code_cells):
        lines = [line.strip() for line in _source(cell).splitlines() if line.strip()]
        first_lines.append({"index": index, "first_line": lines[0][:200] if lines else ""})
    papermill = notebook.get("metadata", {}).get("papermill", {})
    return {
        "path": str(notebook_path),
        "kernel_id": kernel_metadata.get("id"),
        "title": kernel_metadata.get("title"),
        "dataset_sources": kernel_metadata.get("dataset_sources", []),
        "kernel_sources": kernel_metadata.get("kernel_sources", []),
        "competition_sources": kernel_metadata.get("competition_sources", []),
        "papermill_duration": papermill.get("duration"),
        "cells": len(cells),
        "code_cells": len(code_cells),
        "markdown_cells": len(markdown_cells),
        "source_chars": len(all_source),
        "static_csv_inputs": sorted(set(STATIC_CSV.findall(all_source))),
        "writes_submission": "submission.csv" in all_source,
        "interesting_source_lines": interesting_lines[:80],
        "interesting_output_lines": output_lines[:80],
        "code_cell_first_lines": first_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    notebooks = sorted(args.root.rglob("*.ipynb"))
    report = [audit(path) for path in notebooks]
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
