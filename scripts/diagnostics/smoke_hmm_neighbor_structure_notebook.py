"""Replay the generated neighbor-structure overlay against the scored anchor."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT
    / "kaggle"
    / "rogii-pkadopt-hmm-neighbor-structure010"
    / "rogii-pkadopt-hmm-neighbor-structure010.ipynb"
)
ANCHOR = (
    ROOT
    / "kaggle"
    / "external_outputs"
    / "pkadopt-hmm-wellcal-650505"
    / "submission.csv"
)
WORK = ROOT / "tmp" / "neighbor_structure_smoke"


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ANCHOR, WORK / "submission.csv")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = notebook["cells"][-1]["source"]
    source = "".join(source) if isinstance(source, list) else str(source)
    source = source.replace(
        "_ns_work = _NsPath('/kaggle/working') if _NsPath('/kaggle/working').exists() else _NsPath('.')",
        f"_ns_work = _NsPath({str(WORK)!r})",
        1,
    )
    source = source.replace(
        "_ns_data = _NsPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')\n"
        "if not (_ns_data / 'sample_submission.csv').exists():\n"
        "    _ns_data = _NsPath('/kaggle/input/rogii-wellbore-geology-prediction')",
        f"_ns_data = _NsPath({str(ROOT)!r})",
        1,
    )
    namespace = {"_SD_STARTED": time.time()}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace, namespace)

    before = pd.read_csv(WORK / "submission_before_neighbor_structure.csv")
    expected = pd.read_csv(ANCHOR)
    final = pd.read_csv(WORK / "submission.csv")
    if not before["id"].astype(str).equals(expected["id"].astype(str)):
        raise RuntimeError("smoke anchor id mismatch")
    if not np.allclose(before["tvt"], expected["tvt"], rtol=0.0, atol=0.0):
        raise RuntimeError("smoke anchor vector mismatch")
    if not final["id"].astype(str).equals(expected["id"].astype(str)):
        raise RuntimeError("smoke final id mismatch")
    if not np.isfinite(final["tvt"].to_numpy(dtype=float)).all():
        raise RuntimeError("smoke final contains non-finite values")
    audit = json.loads((WORK / "neighbor_structure_audit.json").read_text(encoding="utf-8"))
    if not audit["native_mask_gate"]["same_id_excluded"]:
        raise RuntimeError("same-ID exclusion audit failed")
    if not all(row.get("well") not in row.get("neighbors", []) for row in audit["wells"]):
        raise RuntimeError("same-ID neighbor leaked into overlay")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
