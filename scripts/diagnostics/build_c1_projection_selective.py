"""Build the C1 alpha-0.35 route with the harmful projection bin disabled."""
from __future__ import annotations

import json
from pathlib import Path

from build_hmm010_structural_tomography_batch import build_c1_bin_calibrated
from build_stopdose_breakthrough_notebooks import source


SLUG = "rogii-hmm010-c1bin1off035"
BIN_ALPHAS = (0.35, 0.0, 0.35, 0.35)


def main() -> None:
    output = build_c1_bin_calibrated(BIN_ALPHAS, SLUG)
    notebook = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in notebook["cells"])
    metadata = json.loads(
        (Path("kaggle") / SLUG / "kernel-metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["id"] == f"qwer556617123/{SLUG}"
    assert f"_C1_BIN_ALPHAS = {BIN_ALPHAS!r}" in joined
    assert "lexicographic_run_local_well_rank_mod_4" in joined
    assert "fixed_public_ids_used" in joined
    assert "00e12e8b" not in joined
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{SLUG}:code", "exec")
    print(f"built and checked {output}")


if __name__ == "__main__":
    main()
