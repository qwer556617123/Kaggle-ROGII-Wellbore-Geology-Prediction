"""Build runtime-safe sequence refinements on the verified MHA140SEP4 anchor."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_stopdose_breakthrough_notebooks import (
    clean_code_cells,
    code_cell,
    hmm_runner,
    integrity_audit,
    source,
    validate_notebook,
    write_metadata,
)


MHA_SOURCE = Path(
    "kaggle/external_reviews/frontier_20260720b/mha140sep4/"
    "rogii-det-mha140sep4.ipynb"
)
A10_SOURCE = Path(
    "kaggle/external_reviews/frontier_20260720b/yusuke_latest/"
    "rogii-another-approach.ipynb"
)
HMM_SOURCE = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")

A10_OUTPUT = Path("kaggle/rogii-mha140sep4-a10-rebuild")
HMM_OUTPUT = Path("kaggle/rogii-mha140sep4-student-hmm025")

# These source cells are explicitly read-only diagnostics or an inactive probe.
# Prediction-changing anchor cells remain byte-identical.
MHA_SKIPPED = {43, 44, 45, 46, 47}
A10_SKIPPED = {44, 45, 46, 47, 51}


def _assert_anchor(joined: str) -> None:
    required = (
        'os.environ["ROGII_GOLD_PROFILE"] = "conservative"',
        "_BC_SHIFT = -0.40",
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "1.4, 0.22, 4.0, 40.0, 4.0",
        "Gold visible-prefix calibration overlay",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"verified MHA140SEP4 fragments missing: {missing}")
    forbidden = (
        "Legal heel-cal predictor v2",
        "Smoother SALVAGE sweep",
        "Smoother-vs-STACK OOF blend test",
        "LB PROBE v4",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"read-only/probe cells leaked into scoring rebuild: {leaked}")


def _start_cell() -> dict:
    return code_cell("import time as _sd_time\n_SD_STARTED = _sd_time.time()\n")


def _write_notebook(
    notebook: dict,
    cells: list[dict],
    output: Path,
    slug: str,
    source_path: Path,
    strategy: str,
    audit_extra: str = "",
) -> Path:
    notebook["cells"] = [
        _start_cell(),
        *cells,
        code_cell(integrity_audit(strategy, "_SD_STARTED", audit_extra)),
    ]
    validate_notebook(notebook, slug)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(source_path, output, slug)
    return path


def build_a10() -> Path:
    notebook, cells = clean_code_cells(A10_SOURCE, skipped=A10_SKIPPED)
    joined = "\n".join(source(cell) for cell in cells)
    _assert_anchor(joined)
    required = (
        "A10: slope-aware local sequence refinement",
        "move=_a10_np.clip(0.42*raw*sparse*taper,-0.14,0.14)",
        "submission_before_a10.csv",
        "submission_a10_slope_aware.csv",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"A10 scoring fragments missing: {missing}")
    extra = (
        "_sd_audit['anchor'] = 'verified_public_6979_mha140sep4'\n"
        "_sd_audit['a10_applied_wells'] = int(sum("
        "item.get('status') == 'applied' for item in _A10_REPORT))\n"
        "_sd_audit['a10_report'] = _A10_REPORT"
    )
    return _write_notebook(
        notebook,
        cells,
        A10_OUTPUT,
        "rogii-mha140sep4-a10-rebuild",
        A10_SOURCE,
        "verified_6979_mha140sep4_plus_a10_slope_sequence",
        extra,
    )


def build_hmm() -> Path:
    notebook, cells = clean_code_cells(MHA_SOURCE, skipped=MHA_SKIPPED)
    joined = "\n".join(source(cell) for cell in cells)
    _assert_anchor(joined)

    hmm_notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    hmm_core = copy.deepcopy(hmm_notebook["cells"][58])
    hmm_core["execution_count"] = None
    hmm_core["outputs"] = []
    hmm_core.get("metadata", {}).pop("execution", None)
    hmm_core.get("metadata", {}).pop("papermill", None)
    core_source = source(hmm_core)
    if "class HMMParams" not in core_source or "def run_hmm2" not in core_source:
        raise RuntimeError("exact HMM core was not found in EXP087 source cell 58")

    runner = hmm_runner()
    replacements = {
        "_HMM_WEIGHT = 0.15": "_HMM_WEIGHT = 0.25",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "verified_6979_mha140sep4_plus_student_t_second_order_hmm"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t second-order HMM posterior-mean blend over the verified "
            "6.979 MHA140SEP4 anchor."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one HMM runner replacement: {old}")
        runner = runner.replace(old, new)

    cells.extend([hmm_core, code_cell(runner)])
    extra = (
        "_sd_audit['anchor'] = 'verified_public_6979_mha140sep4'\n"
        "_sd_audit['hmm'] = _HMM_AUDIT"
    )
    return _write_notebook(
        notebook,
        cells,
        HMM_OUTPUT,
        "rogii-mha140sep4-student-hmm025",
        MHA_SOURCE,
        "verified_6979_mha140sep4_plus_student_t_hmm_weight_025",
        extra,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()
    targets = args.targets or ["a10", "hmm"]
    unknown = sorted(set(targets) - {"a10", "hmm"})
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}")
    for target in targets:
        path = build_a10() if target == "a10" else build_hmm()
        print(f"built {path}")


if __name__ == "__main__":
    main()
