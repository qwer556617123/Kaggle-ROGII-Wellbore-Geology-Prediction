"""Build rank-mod-8 probes that split the strongest C1 parent bins."""
from __future__ import annotations

import json
from pathlib import Path

from build_hmm010_structural_tomography_batch import (
    build_c1_partition_calibrated,
)
from build_stopdose_breakthrough_notebooks import source


PROBES = {
    "rogii-hmm010-c1r8-b1-split": (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "rogii-hmm010-c1r8-b1-child": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "rogii-hmm010-c1r8-b3-split": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    "rogii-hmm010-c1r8-b3-child": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
}


def main() -> None:
    for slug, alphas in PROBES.items():
        output = build_c1_partition_calibrated(alphas, slug)
        notebook = json.loads(output.read_text(encoding="utf-8"))
        joined = "\n".join(source(cell) for cell in notebook["cells"])
        metadata = json.loads(
            (Path("kaggle") / slug / "kernel-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["id"] == f"qwer556617123/{slug}"
        assert f"_C1_BIN_ALPHAS = {alphas!r}" in joined
        assert "_C1_PARTITION_MOD = 8" in joined
        assert "lexicographic_run_local_well_rank_mod_8" in joined
        assert "fixed_public_ids_used" in joined
        assert "00e12e8b" not in joined
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile(source(cell), f"{slug}:code", "exec")
        print(f"built and checked {output}")


if __name__ == "__main__":
    main()
