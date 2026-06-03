"""
Evaluate rank alignment between local validation signals and Public LB.

The sample is intentionally small. This script is a lightweight guardrail: it
checks whether local validation can sort candidates in roughly the same order as
Public LB. Lower local score and lower LB are both treated as better.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
LB_PATH = ROOT_DIR / "docs" / "lb_history.csv"
OUT_PATH = ROOT_DIR / "docs" / "cv_lb_alignment_summary.csv"
RANK_DETAIL_PATH = ROOT_DIR / "docs" / "cv_lb_rank_alignment.csv"


def summarize_rank_alignment(name: str, frame: pd.DataFrame, takeaway: str) -> dict[str, object]:
    work = frame[frame["local_signal_value"].notna() & frame["public_lb"].notna()].copy()
    row = {
        "comparison": name,
        "n": len(work),
        "pearson": None,
        "spearman": None,
        "kendall": None,
        "avg_abs_rank_error": None,
        "exact_rank_match": None,
        "takeaway": takeaway,
    }
    if len(work) < 2:
        return row

    work["local_rank"] = work["local_signal_value"].rank(method="min", ascending=True)
    work["lb_rank"] = work["public_lb"].rank(method="min", ascending=True)
    row["pearson"] = work["local_signal_value"].corr(work["public_lb"], method="pearson")
    row["spearman"] = work["local_signal_value"].corr(work["public_lb"], method="spearman")
    row["kendall"] = work["local_signal_value"].corr(work["public_lb"], method="kendall")
    row["avg_abs_rank_error"] = (work["local_rank"] - work["lb_rank"]).abs().mean()
    row["exact_rank_match"] = bool((work["local_rank"] == work["lb_rank"]).all())
    return row


def main() -> None:
    df = pd.read_csv(LB_PATH)
    scored = df[df["public_lb"].notna()].copy()

    comparable = scored[
        scored["local_signal_name"].isin(["val_rmse", "cv_row_rmse", "cv_rank_replay_row_rmse"])
    ].copy()
    val_rmse = scored[scored["local_signal_name"] == "val_rmse"].copy()
    all_local = scored[scored["local_signal_value"].notna()].copy()

    rows = [
        summarize_rank_alignment(
            "val_rmse_only",
            val_rmse,
            "closest current proxy for CV/LB ordering, but n is too small to tune parameters hard",
        ),
        summarize_rank_alignment(
            "comparable_cv_like_signals",
            comparable,
            "rank signal after excluding visible-ID oracle; useful directionally, still under-sampled",
        ),
        summarize_rank_alignment(
            "all_local_signals",
            all_local,
            "includes mixed signal types such as visible-ID oracle, so treat as a contamination check",
        ),
    ]

    v23 = scored[scored["method"].str.startswith("v23")].copy()
    if len(v23) >= 2:
        rows.append(
            {
                "comparison": "v23_variants",
                "n": len(v23),
                "pearson": None,
                "spearman": None,
                "kendall": None,
                "avg_abs_rank_error": None,
                "exact_rank_match": None,
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
            "kendall": None,
            "avg_abs_rank_error": None,
            "exact_rank_match": None,
            "takeaway": (
                "Use CV primarily to rank candidates. If repeated LB-backed rows fail to align, "
                "fall back to standard well-level CV settings instead of over-tuning."
            ),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    detail = all_local.copy()
    detail["local_rank"] = detail["local_signal_value"].rank(method="min", ascending=True)
    detail["lb_rank"] = detail["public_lb"].rank(method="min", ascending=True)
    detail["abs_rank_error"] = (detail["local_rank"] - detail["lb_rank"]).abs()
    detail[
        [
            "version",
            "method",
            "local_signal_name",
            "local_signal_value",
            "public_lb",
            "local_rank",
            "lb_rank",
            "abs_rank_error",
            "notes",
        ]
    ].to_csv(RANK_DETAIL_PATH, index=False)

    print(out.to_string(index=False))
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {RANK_DETAIL_PATH}")


if __name__ == "__main__":
    main()
