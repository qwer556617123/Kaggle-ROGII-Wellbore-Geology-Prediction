"""Build a final-hour diagonal-extrapolation candidate targeting sub-6 RMSE."""
from __future__ import annotations

import json
import math
from pathlib import Path

from build_c1_datum_b12_centered_b3_batch import contrasts
from build_c1_datum_b12_centered_b3_final import centered_final_cell
from build_c1_rank8_datum16_multistage_calibrated import multistage_cell
from build_c1_rank8_datum_codes import C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import source


B12_SCORE = 6.186
MEASURED_PROBE_SCORES = (6.245, 6.304, 6.187)
MEASURED_AMPLITUDE = 1.0
ASSUMED_A3_PROJECTION = -1.50


def projections() -> tuple[float, float, float, float]:
    measured = tuple(
        float(
            (score * score - B12_SCORE * B12_SCORE - MEASURED_AMPLITUDE**2)
            / (2.0 * MEASURED_AMPLITUDE)
        )
        for score in MEASURED_PROBE_SCORES
    )
    return measured + (ASSUMED_A3_PROJECTION,)


def nominal_score() -> float:
    values = projections()
    return math.sqrt(B12_SCORE * B12_SCORE - sum(value * value for value in values))


def build() -> Path:
    decoded = contrasts()
    inferred = projections()
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8))
    notebook["cells"].append(multistage_cell((1, 2), decoded))
    notebook["cells"].append(centered_final_cell(inferred, (0, 1, 2, 3)))
    slug = "rogii-c1r8-d16-b12-b3diag150"
    output = _write(
        notebook,
        slug,
        "c1_rank8_datum16_b12_b3_diagonal_minus150",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_multistage'] = _DATUM16_MULTISTAGE_AUDIT\n"
        "_sd_audit['b3_diagonal_extrapolation'] = _B3_CENTERED_FINAL_AUDIT",
    )
    generated = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in generated["cells"])
    assert f"_BCF_PROJECTIONS = {inferred!r}" in joined
    assert "_BCF_PARENT_CODE_INDICES = (0, 1, 2, 3)" in joined
    assert "00e12e8b" not in joined
    for cell in generated["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{slug}:code", "exec")
    return output


def main() -> None:
    print(
        json.dumps(
            {
                "b12_score": B12_SCORE,
                "measured_probe_scores": MEASURED_PROBE_SCORES,
                "projections": projections(),
                "optimal_coefficients_under_assumption": tuple(-value for value in projections()),
                "nominal_score": nominal_score(),
                "assumption": "unmeasured a3b3 projection follows negative diagonal dominance",
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"built and checked {build()}")


if __name__ == "__main__":
    main()
