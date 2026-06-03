"""
Evaluate rough alignment between local validation signals and Public LB.

The sample is intentionally small. This script is a lightweight guardrail: it
shows whether current local scores are trustworthy enough to spend submissions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
LB_PATH = ROOT_DIR / "docs" / "lb_history.csv"
OUT_PATH = ROOT_DIR / "docs" / "cv_lb_alignment_summary.csv"


def main() -> None:
    df = pd.read_csv(LB_PATH)
    scored = df[df["public_lb"].notna()].copy()
    val_scored = scored[scored["local_signal_value"].notna()].copy()

    rows = []
    if len(val_scored) >= 2:
        pearson = val_scored["local_signal_value"].corr(val_scored["public_lb"], method="pearson")
        spearman = val_scored["local_signal_value"].corr(val_scored["public_lb"], method="spearman")
        rows.append(
            {
                "comparison": "all_rows_with_local_signal_and_lb",
                "n": len(val_scored),
                "pearson": pearson,
                "spearman": spearman,
                "takeaway": "too few and mixed signals; do not trust exact LB prediction",
            }
        )

    v23 = scored[scored["method"].str.startswith("v23")].copy()
    if len(v23) >= 2:
        rows.append(
            {
                "comparison": "v23_variants",
                "n": len(v23),
                "pearson": None,
                "spearman": None,
                "takeaway": (
                    "v23 variants range from 12.044 to 17.482 LB, so implementation/data-pool "
                    "details dominate simple validation notes"
                ),
            }
        )

    rows.append(
        {
            "comparison": "current_gate",
            "n": len(df),
            "pearson": None,
            "spearman": None,
            "takeaway": (
                "CV is useful as a rejection gate, but not sufficient as an LB predictor. "
                "Candidates should beat CV gates and reproduce the exact v23 12.044 path."
            ),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
