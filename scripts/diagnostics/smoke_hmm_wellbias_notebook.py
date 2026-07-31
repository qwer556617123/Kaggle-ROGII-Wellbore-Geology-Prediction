"""Replay the HMM well-bias overlay on a scored visible output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-name", default="rogii-pkadopt-hmm-wellbias-rf")
    parser.add_argument("--source-output", default="pkadopt-hmm-wellcal-650505")
    args = parser.parse_args()
    notebook_path = ROOT / "kaggle" / args.notebook_name / f"{args.notebook_name}.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    overlay = next(
        _source(cell)
        for cell in notebook["cells"]
        if "Predictor-driven per-well datum correction" in _source(cell)
    )
    anti_audit_cell = next(
        (
            _source(cell)
            for cell in notebook["cells"]
            if "Exact anti-well-bias rerun audit" in _source(cell)
        ),
        None,
    )
    source = ROOT / "kaggle" / "external_outputs" / args.source_output / "submission.csv"

    class CFG:
        dataset_path = ROOT

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        shutil.copy2(source, tmp / "submission.csv")
        previous = Path.cwd()
        try:
            os.chdir(tmp)
            namespace = {"CFG": CFG, "_SD_STARTED": time.time()}
            exec(compile(overlay, "<well_bias_overlay>", "exec"), namespace)
            if anti_audit_cell is not None:
                localized = anti_audit_cell.replace(
                    "_awb_work = _AwbPath('/kaggle/working') if _AwbPath('/kaggle/working').exists() else _AwbPath('.')",
                    "_awb_work = _AwbPath('.')",
                    1,
                )
                localized = localized.replace(
                    "_awb_data = _AwbPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')\n"
                    "if not (_awb_data / 'sample_submission.csv').exists():\n"
                    "    _awb_data = _AwbPath('/kaggle/input/rogii-wellbore-geology-prediction')",
                    f"_awb_data = _AwbPath({str(ROOT)!r})",
                    1,
                )
                exec(compile(localized, "<anti_well_bias_audit>", "exec"), namespace)
        finally:
            os.chdir(previous)
        before = pd.read_csv(tmp / "submission_before_well_bias.csv")
        final = pd.read_csv(tmp / "submission.csv")
        audit = pd.read_csv(tmp / "well_bias_audit_prefix_gr_rf_leaf70.csv")
        assert before["id"].astype(str).equals(final["id"].astype(str))
        assert np.isfinite(final["tvt"].to_numpy(float)).all()
        move = final["tvt"].to_numpy(float) - before["tvt"].to_numpy(float)
        expected = dict(
            zip(audit["well"].astype(str), audit["applied_tvt_shift"].astype(float))
        )
        wells = final["id"].astype(str).str.rsplit("_", n=1).str[0]
        expected_move = wells.map(expected).to_numpy(float)
        np.testing.assert_allclose(move, expected_move, rtol=0.0, atol=2e-9)
        if anti_audit_cell is not None:
            anti_audit = json.loads(
                (tmp / "hmm_anti_well_bias_audit.json").read_text(encoding="utf-8")
            )
            assert anti_audit["bound_valid"] is False
            assert (
                anti_audit["naive_predicted_score_lower"]
                < anti_audit["naive_predicted_score_upper"]
            )
        print(
            json.dumps(
                {
                    "rows": len(final),
                    "wells": audit.to_dict(orient="records"),
                    "mean_move_ft": float(np.mean(move)),
                    "rms_move_ft": float(np.sqrt(np.mean(move**2))),
                    "max_abs_move_ft": float(np.max(np.abs(move))),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
