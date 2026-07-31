"""Evaluate fault-aware residual smoothing on existing native-mask anchors."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fault_aware_smoother import FaultSmootherConfig, smooth_fault_residual


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in details.groupby("variant", sort=False):
        total_rows = int(group["rows"].sum())
        pooled = float(np.sqrt(group["sse"].sum() / max(total_rows, 1)))
        scores = group["rmse"].to_numpy(dtype=float)
        worst_n = max(1, int(np.ceil(0.10 * len(group))))
        worst = group.nlargest(worst_n, "sse")
        rows.append(
            {
                "variant": variant,
                "wells": int(group["well"].nunique()),
                "rows": total_rows,
                "row_rmse": pooled,
                "well_mean": float(np.mean(scores)),
                "well_median": float(np.median(scores)),
                "well_p90": float(np.quantile(scores, 0.90)),
                "well_max": float(np.max(scores)),
                "worst10_sse_share": float(worst["sse"].sum() / max(group["sse"].sum(), 1e-12)),
            }
        )
    return pd.DataFrame(rows).sort_values("row_rmse")


def profile_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (profile, variant), group in details.groupby(
        ["nearest_public_profile", "variant"], sort=True
    ):
        rows.append(
            {
                "profile": profile,
                "variant": variant,
                "wells": int(group["well"].nunique()),
                "rows": int(group["rows"].sum()),
                "row_rmse": float(
                    np.sqrt(group["sse"].sum() / max(group["rows"].sum(), 1))
                ),
                "well_median": float(group["rmse"].median()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--anchor", type=Path, default=Path("reports/native100_pf_anchor.csv")
    )
    parser.add_argument(
        "--anchor-details", type=Path, default=Path("reports/native100_pf_details.csv")
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("reports/fault_native100")
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-residual", type=float, default=40.0)
    parser.add_argument("--fault-hazard", type=float, default=0.0008)
    parser.add_argument("--anchor-prior-strength", type=float, default=0.004)
    parser.add_argument("--emission-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    anchor = pd.read_csv(args.anchor)[["id", "tvt"]]
    anchor["id"] = anchor["id"].astype(str)
    anchor["well"] = anchor["id"].str.rsplit("_", n=1).str[0]
    anchor["row"] = anchor["id"].str.rsplit("_", n=1).str[1].astype(int)
    metadata = pd.read_csv(args.anchor_details).drop_duplicates("well")
    profiles = metadata.set_index("well")["nearest_public_profile"].to_dict()
    wells = sorted(anchor["well"].unique())
    if args.limit > 0:
        wells = wells[: args.limit]
    config = FaultSmootherConfig(
        max_residual=args.max_residual,
        fault_hazard=args.fault_hazard,
        anchor_prior_strength=args.anchor_prior_strength,
        emission_temperature=args.emission_temperature,
    )
    records = []
    mode_records = []
    for index, well in enumerate(wells, 1):
        well_started = time.time()
        hw = pd.read_csv(args.data_dir / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(args.data_dir / "train" / f"{well}__typewell.csv")
        group = anchor[anchor["well"].eq(well)].sort_values("row")
        full_anchor = hw["TVT_input"].to_numpy(dtype=float).copy()
        rows = group["row"].to_numpy(dtype=int)
        full_anchor[rows] = group["tvt"].to_numpy(dtype=float)
        if not np.isfinite(full_anchor).all():
            raise RuntimeError(f"anchor does not cover all rows for {well}")
        eval_mask = hw["TVT_input"].isna().to_numpy()
        truth = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        if not np.isfinite(truth).all():
            raise RuntimeError(f"non-finite truth in eval rows for {well}")
        no_fault = smooth_fault_residual(
            hw, tw, full_anchor, config=config, allow_faults=False
        )
        fault = smooth_fault_residual(
            hw, tw, full_anchor, config=config, allow_faults=True
        )
        variants = {
            "anchor": full_anchor[eval_mask],
            "no_fault_posterior": no_fault.posterior[eval_mask],
            "fault_map": fault.map_path[eval_mask],
            "fault_posterior": fault.posterior[eval_mask],
            "fault_blend_010": (
                0.90 * full_anchor[eval_mask] + 0.10 * fault.posterior[eval_mask]
            ),
            "fault_blend_025": (
                0.75 * full_anchor[eval_mask] + 0.25 * fault.posterior[eval_mask]
            ),
        }
        for variant, prediction in variants.items():
            score = rmse(truth, prediction)
            records.append(
                {
                    "well": well,
                    "nearest_public_profile": profiles.get(well, "unknown"),
                    "variant": variant,
                    "rows": int(eval_mask.sum()),
                    "rmse": score,
                    "sse": float(np.sum((truth - prediction) ** 2)),
                    "mean_error": float(np.mean(prediction - truth)),
                    "mean_abs_move": float(np.mean(np.abs(prediction - full_anchor[eval_mask]))),
                }
            )
        mode_records.append(
            {
                "well": well,
                "runtime_sec": float(time.time() - well_started),
                **{f"no_fault_{key}": value for key, value in no_fault.metadata.items() if key != "config"},
                **{f"fault_{key}": value for key, value in fault.metadata.items() if key != "config"},
            }
        )
        print(
            f"[{index:03d}/{len(wells):03d}] {well} "
            f"anchor={records[-6]['rmse']:.3f} "
            f"no_fault={records[-5]['rmse']:.3f} "
            f"fault={records[-3]['rmse']:.3f} "
            f"blend10={records[-2]['rmse']:.3f} "
            f"sec={time.time() - well_started:.1f}",
            flush=True,
        )

    details = pd.DataFrame(records)
    summary = summarize(details)
    profiles_df = profile_summary(details)
    modes = pd.DataFrame(mode_records)
    base = summary.set_index("variant").loc["anchor"]
    candidate_name = str(summary.iloc[0]["variant"])
    if candidate_name == "anchor":
        candidate_name = "fault_blend_010"
    candidate = summary.set_index("variant").loc[candidate_name]
    profile_pivot = profiles_df.pivot(index="profile", columns="variant", values="row_rmse")
    matched = profile_pivot.dropna(subset=["anchor", candidate_name])
    profiles_improved = int((matched[candidate_name] < matched["anchor"]).sum())
    required_profiles = int(np.ceil(2.0 * len(matched) / 3.0)) if len(matched) else 0
    gate = {
        "verdict": "PASS",
        "wells": int(len(wells)),
        "runtime_sec": float(time.time() - started),
        "candidate": candidate_name,
        "row_gain": float(base["row_rmse"] - candidate["row_rmse"]),
        "well_median_gain": float(base["well_median"] - candidate["well_median"]),
        "well_p90_delta": float(candidate["well_p90"] - base["well_p90"]),
        "worst10_sse_share_delta": float(
            candidate["worst10_sse_share"] - base["worst10_sse_share"]
        ),
        "profiles_improved": profiles_improved,
        "profiles_required": required_profiles,
        "config": {
            "max_residual": config.max_residual,
            "grid_step": config.grid_step,
            "fault_hazard": config.fault_hazard,
            "anchor_prior_strength": config.anchor_prior_strength,
            "emission_temperature": config.emission_temperature,
        },
    }
    checks = [
        gate["row_gain"] >= 0.60,
        gate["well_median_gain"] >= 0.40,
        gate["well_p90_delta"] <= 0.25,
        gate["worst10_sse_share_delta"] < 0.0,
        profiles_improved >= required_profiles,
    ]
    if not all(checks):
        gate["verdict"] = "STOP"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), index=False)
    summary.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), index=False)
    profiles_df.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_profiles.csv"), index=False
    )
    modes.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_modes.csv"), index=False)
    args.output_prefix.with_name(args.output_prefix.name + "_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
