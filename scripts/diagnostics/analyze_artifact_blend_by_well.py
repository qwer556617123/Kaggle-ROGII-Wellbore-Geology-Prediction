from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = [0.90, 0.85, 0.825, 0.80, 0.79, 0.75, 0.70]


def load_submission(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(df.columns) != {"id", "tvt"}:
        raise ValueError(f"{path} must contain id,tvt columns; got {list(df.columns)}")
    df = df.rename(columns={"tvt": value_name})
    df["well_id"] = df["id"].str.rsplit("_", n=1).str[0]
    df["row_idx"] = df["id"].str.rsplit("_", n=1).str[1].astype(int)
    return df


def summarize_profile(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    values = df[value_col].to_numpy(float)
    diffs = np.diff(values)
    return {
        "n_rows": float(len(values)),
        "start": float(values[0]),
        "end": float(values[-1]),
        "net": float(values[-1] - values[0]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "std": float(np.std(values)),
        "mean_step": float(np.mean(diffs)) if len(diffs) else 0.0,
        "std_step": float(np.std(diffs)) if len(diffs) else 0.0,
        "min_step": float(np.min(diffs)) if len(diffs) else 0.0,
        "max_step": float(np.max(diffs)) if len(diffs) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=DEFAULT_WEIGHTS,
        help="PF weights to summarize; artifact weight is 1 - PF weight.",
    )
    args = parser.parse_args()

    pf = load_submission(args.pf, "pf_tvt")
    artifact = load_submission(args.artifact, "artifact_tvt")
    merged = pf.merge(
        artifact[["id", "artifact_tvt"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(pf) or len(merged) != len(artifact):
        raise ValueError("PF and artifact submissions do not align one-to-one")

    merged["artifact_minus_pf"] = merged["artifact_tvt"] - merged["pf_tvt"]

    profile_rows: list[dict[str, float | str]] = []
    delta_rows: list[dict[str, float | str]] = []
    for well_id, well_df in merged.groupby("well_id", sort=True):
        well_df = well_df.sort_values("row_idx")
        delta = well_df["artifact_minus_pf"].to_numpy(float)
        delta_rows.append(
            {
                "well_id": well_id,
                "n_rows": len(well_df),
                "artifact_minus_pf_mean": float(np.mean(delta)),
                "artifact_minus_pf_std": float(np.std(delta)),
                "artifact_minus_pf_min": float(np.min(delta)),
                "artifact_minus_pf_max": float(np.max(delta)),
                "artifact_minus_pf_start": float(delta[0]),
                "artifact_minus_pf_end": float(delta[-1]),
                "artifact_minus_pf_net": float(delta[-1] - delta[0]),
            }
        )

        for pf_weight in args.weights:
            value_col = f"blend_{pf_weight:.3f}"
            well_df = well_df.copy()
            well_df[value_col] = (
                pf_weight * well_df["pf_tvt"]
                + (1.0 - pf_weight) * well_df["artifact_tvt"]
            )
            summary = summarize_profile(well_df, value_col)
            profile_rows.append(
                {
                    "well_id": well_id,
                    "pf_weight": pf_weight,
                    "artifact_weight": 1.0 - pf_weight,
                    **summary,
                }
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(profile_rows).to_csv(
        args.outdir / "artifact_blend_profile_by_well.csv", index=False
    )
    pd.DataFrame(delta_rows).to_csv(
        args.outdir / "artifact_minus_pf_by_well.csv", index=False
    )

    print("Wrote", args.outdir / "artifact_blend_profile_by_well.csv")
    print("Wrote", args.outdir / "artifact_minus_pf_by_well.csv")


if __name__ == "__main__":
    main()
