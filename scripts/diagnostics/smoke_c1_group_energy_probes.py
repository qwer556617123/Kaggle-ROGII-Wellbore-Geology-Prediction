"""Static integrity checks for the single-bin C1 energy probes."""
from __future__ import annotations

import json
from pathlib import Path

from build_c1_group_energy_probes import VARIANTS
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
        assert metadata["competition_sources"] == [
            "rogii-wellbore-geology-prediction"
        ]
        assert f"_C1_BIN_ALPHAS = {variant.bin_alphas!r}" in joined
        assert "lexicographic_run_local_well_rank_mod_4" in joined
        assert "submission_before_c1_heel.csv" in joined
        assert "c1_heel_audit.json" in joined
        assert "fixed_public_ids_used" in joined
        assert "00e12e8b" not in joined

        nonzero = [index for index, value in enumerate(variant.bin_alphas) if value]
        assert nonzero == [variant.bin_index]
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile(source(cell), f"{variant.slug}:code", "exec")
        checked.append(
            {
                "slug": variant.slug,
                "bin_index": variant.bin_index,
                "bin_alphas": variant.bin_alphas,
                "cells": len(notebook["cells"]),
            }
        )
    print(checked)


if __name__ == "__main__":
    main()
