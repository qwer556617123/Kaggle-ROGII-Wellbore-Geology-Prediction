"""Build competitive b3 probes centered on the calibrated b12 candidate."""
from __future__ import annotations

import json
from pathlib import Path

from build_c1_rank8_datum16_codes import datum16_cell
from build_c1_rank8_datum16_multistage_calibrated import multistage_cell
from build_c1_rank8_datum_codes import C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import source
from fit_c1_rank8_datum16_stage import decode_stage


B1_SCORES = (6.714, 6.394, 6.984, 6.867)
B2_SCORES = (6.988, 6.880, 6.443, 6.715)
ANCHOR_SCORE = 6.524
CODE_AMPLITUDE = 2.0
PROBE_AMPLITUDE = 1.0
PARENT_CODES = (0, 1, 2)


def contrasts() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(
        tuple(
            float(value)
            for value in decode_stage(
                ANCHOR_SCORE,
                scores,
                CODE_AMPLITUDE,
                child_code,
            )["parent_contrast_projection"]
        )
        for child_code, scores in ((1, B1_SCORES), (2, B2_SCORES))
    )


def build(parent_code: int) -> Path:
    decoded = contrasts()
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8))
    notebook["cells"].append(multistage_cell((1, 2), decoded))
    notebook["cells"].append(datum16_cell(parent_code, 3, PROBE_AMPLITUDE))
    slug = f"rogii-c1r8-d16-b12-a{parent_code}b3-probe"
    output = _write(
        notebook,
        slug,
        f"c1_rank8_datum16_b12_centered_a{parent_code}b3_probe",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_multistage'] = _DATUM16_MULTISTAGE_AUDIT\n"
        "_sd_audit['datum16_code'] = _DATUM16_CODE_AUDIT",
    )
    generated = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in generated["cells"])
    assert "_DMC_CHILD_CODE_INDICES = (1, 2)" in joined
    assert f"_D16_PARENT_CODE_INDEX = {parent_code!r}" in joined
    assert "_D16_CHILD_CODE_INDEX = 3" in joined
    assert "_D16_AMPLITUDE = 1.0" in joined
    assert "00e12e8b" not in joined
    for cell in generated["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{slug}:code", "exec")
    return output


def main() -> None:
    for parent_code in PARENT_CODES:
        print(f"built and checked {build(parent_code)}")


if __name__ == "__main__":
    main()
