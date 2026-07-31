"""Replay the orthogonal structure/well-bias fusion against the scored anchor."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NAME = "rogii-pkadopt-hmm-structure-wellbias-fusion"
NOTEBOOK = ROOT / "kaggle" / NAME / f"{NAME}.ipynb"
ANCHOR = (
    ROOT
    / "kaggle"
    / "external_outputs"
    / "pkadopt-hmm-wellcal-650505"
    / "submission.csv"
)
WORK = ROOT / "tmp" / "structure_wellbias_fusion_smoke"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _localize_data(source: str, prefix: str) -> str:
    upper = prefix.upper()
    old = (
        f"_{prefix}_data = _{upper.title()}Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction')\n"
        f"if not (_{prefix}_data / 'sample_submission.csv').exists():\n"
        f"    _{prefix}_data = _{upper.title()}Path('/kaggle/input/rogii-wellbore-geology-prediction')"
    )
    # The generated cells use mixed-case aliases (_NsPath and _FwPath).
    alias = {"ns": "_NsPath", "fw": "_FwPath"}[prefix]
    old = (
        f"_{prefix}_data = {alias}('/kaggle/input/competitions/rogii-wellbore-geology-prediction')\n"
        f"if not (_{prefix}_data / 'sample_submission.csv').exists():\n"
        f"    _{prefix}_data = {alias}('/kaggle/input/rogii-wellbore-geology-prediction')"
    )
    return source.replace(old, f"_{prefix}_data = {alias}({str(ROOT)!r})", 1)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ANCHOR, WORK / "submission.csv")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    structure = next(
        _source(cell)
        for cell in notebook["cells"]
        if "Cross-well structural-path transfer" in _source(cell)
    )
    well_bias = next(
        _source(cell)
        for cell in notebook["cells"]
        if "Predictor-driven per-well datum correction" in _source(cell)
    )
    final_audit = next(
        _source(cell)
        for cell in notebook["cells"]
        if "Final orthogonal structure/well-bias fusion audit" in _source(cell)
    )
    structure = structure.replace(
        "_ns_work = _NsPath('/kaggle/working') if _NsPath('/kaggle/working').exists() else _NsPath('.')",
        "_ns_work = _NsPath('.')",
        1,
    )
    structure = _localize_data(structure, "ns")
    final_audit = final_audit.replace(
        "_fw_work = _FwPath('/kaggle/working') if _FwPath('/kaggle/working').exists() else _FwPath('.')",
        "_fw_work = _FwPath('.')",
        1,
    )
    final_audit = _localize_data(final_audit, "fw")

    class CFG:
        dataset_path = ROOT

    namespace = {"_SD_STARTED": time.time(), "CFG": CFG}
    previous = Path.cwd()
    try:
        os.chdir(WORK)
        exec(compile(structure, "<neighbor_structure>", "exec"), namespace, namespace)
        exec(compile(well_bias, "<well_bias>", "exec"), namespace, namespace)
        exec(compile(final_audit, "<fusion_audit>", "exec"), namespace, namespace)
    finally:
        os.chdir(previous)

    anchor = pd.read_csv(ANCHOR)
    structure_frame = pd.read_csv(WORK / "submission_before_well_bias.csv")
    final = pd.read_csv(WORK / "submission.csv")
    audit = json.loads(
        (WORK / "hmm_structure_wellbias_fusion_audit.json").read_text(encoding="utf-8")
    )
    assert anchor["id"].astype(str).equals(final["id"].astype(str))
    assert np.isfinite(final["tvt"].to_numpy(dtype=float)).all()
    assert audit["same_id_excluded"]
    structure_move = structure_frame["tvt"].to_numpy(float) - anchor["tvt"].to_numpy(float)
    bias_move = final["tvt"].to_numpy(float) - structure_frame["tvt"].to_numpy(float)
    np.testing.assert_allclose(
        np.sqrt(np.mean(structure_move**2)), audit["structure_move_rms_ft"], atol=1e-12
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(bias_move**2)), audit["well_bias_move_rms_ft"], atol=1e-12
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
