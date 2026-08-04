"""Build the aggressive cap-4 frontier of the calibrated b12 candidate."""
from __future__ import annotations

import json
from pathlib import Path

from build_c1_datum_b12_centered_b3_batch import contrasts
from build_c1_rank8_datum16_multistage_calibrated import multistage_cell
from build_c1_rank8_datum_codes import C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import source


OFFSET_CAP = 4.0


def build() -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8))
    notebook["cells"].append(multistage_cell((1, 2), contrasts(), OFFSET_CAP))
    slug = "rogii-c1r8-d16-b12-cap4"
    output = _write(
        notebook,
        slug,
        "c1_rank8_datum16_b12_cap4_aggressive",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_multistage'] = _DATUM16_MULTISTAGE_AUDIT",
    )
    generated = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in generated["cells"])
    assert "_DMC_CHILD_CODE_INDICES = (1, 2)" in joined
    assert "_DMC_OFFSET_CAP = 4.0" in joined
    assert "00e12e8b" not in joined
    for cell in generated["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{slug}:code", "exec")
    return output


def main() -> None:
    print(f"built and checked {build()}")


if __name__ == "__main__":
    main()
