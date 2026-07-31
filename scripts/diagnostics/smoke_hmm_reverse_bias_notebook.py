"""Replay the generated reverse-bias notebook cell on audited public outputs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-name", default="rogii-pkadopt-hmm-student-reverse1750"
    )
    parser.add_argument("--source-output", default="pkadopt-hmm-student025")
    parser.add_argument("--dose", type=float, default=-1.75)
    args = parser.parse_args()
    notebook_path = ROOT / "kaggle" / args.notebook_name / f"{args.notebook_name}.ipynb"
    source_dir = ROOT / "kaggle" / "external_outputs" / args.source_output
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    with tempfile.TemporaryDirectory(prefix="rogii_reverse_smoke_") as temp_name:
        temp = Path(temp_name)
        for name in (
            "submission_before_model_package.csv",
            "submission_before_exact_hmm.csv",
            "submission_model_package_gated_020.csv",
        ):
            shutil.copyfile(source_dir / name, temp / name)
        shutil.copyfile(source_dir / "submission.csv", temp / "submission.csv")

        previous = Path.cwd()
        try:
            os.chdir(temp)
            exec(compile(source, str(notebook_path), "exec"), {})
        finally:
            os.chdir(previous)

        audit = json.loads((temp / "reverse_bias_audit.json").read_text(encoding="utf-8"))
        before = pd.read_csv(temp / "submission_before_reverse_bias.csv")
        after = pd.read_csv(temp / "submission.csv")
        package_base = pd.read_csv(temp / "submission_before_model_package.csv")
        positive = pd.read_csv(temp / "submission_model_package_gated_020.csv")
        expected = (
            before["tvt"].to_numpy(dtype=float)
            + args.dose
            * (
                positive["tvt"].to_numpy(dtype=float)
                - package_base["tvt"].to_numpy(dtype=float)
            )
            / 0.020
        )
        assert before["id"].astype(str).equals(after["id"].astype(str))
        assert np.isfinite(after["tvt"].to_numpy(dtype=float)).all()
        assert np.allclose(after["tvt"].to_numpy(dtype=float), expected, rtol=0.0, atol=1e-10)
        assert audit["reverse_dose"] == args.dose
        assert audit["rows"] == len(after) == 14151
        assert audit["rms_move_ft"] > 0
        print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
