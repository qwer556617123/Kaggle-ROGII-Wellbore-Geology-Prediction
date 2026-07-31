"""Static integrity checks for generated orientation-field notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from build_orientation_field_notebooks import VARIANTS
from build_stopdose_breakthrough_notebooks import source


def main() -> None:
    checked = []
    for variant in VARIANTS:
        directory = Path("kaggle") / variant.slug
        notebook_path = directory / f"{variant.slug}.ipynb"
        metadata_path = directory / "kernel-metadata.json"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        joined = "\n".join(source(cell) for cell in notebook["cells"])
        assert metadata["id"] == f"qwer556617123/{variant.slug}"
        assert metadata["code_file"] == notebook_path.name
        assert metadata["competition_sources"] == ["rogii-wellbore-geology-prediction"]
        assert "ROGII_ORIENTATION_MODE" in joined
        assert "ROGII_ORIENTATION_K" in joined
        assert "ROGII_ORIENTATION_SEGMENT_ROWS" in joined
        assert "ROGII_ORIENTATION_DRIFT_SCALE" in joined
        assert "ROGII_TOOL_RESPONSE" in joined
        assert "orientation_field_audit.json" in joined
        assert "submission_before_orientation_field.csv" in joined
        assert "same_id_train_contacts_excluded" in joined
        assert "target_tail_tvt_used" in joined
        assert "fixed_public_ids_used" in joined
        assert "00e12e8b" not in joined
        for cell in notebook["cells"][-3:]:
            compile(source(cell), f"{variant.slug}:{cell.get('cell_type')}", "exec")
        checked.append(
            {
                "slug": variant.slug,
                "cells": len(notebook["cells"]),
                "push_eligible": variant.push_eligible,
            }
        )
    print(checked)


if __name__ == "__main__":
    main()
