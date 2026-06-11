"""Evaluate learned residual correction on top of PF baseline predictions.

The goal is to test a larger jump than hold-weight tuning: keep the strong PF
path as a baseline, then learn a lightweight residual model from pseudo-hidden
train wells using only features that are also available for test wells.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_pf_module():
    path = Path("scripts/diagnostics/evaluate_pf_variants.py")
    spec = importlib.util.spec_from_file_location("pf_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(dtype=float)
    return tw_tvt, tw_gr


def row_features(hw: pd.DataFrame, tw: pd.DataFrame, pred_base: np.ndarray, pred_alt: np.ndarray) -> pd.DataFrame:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    known = hw[hw["TVT_input"].notna()]
    eval_idx = np.where(eval_mask)[0]
    last = known.iloc[-1]
    tw_tvt, tw_gr = typewell_arrays(tw)
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(dtype=float)
    pred_gr = np.interp(pred_base[eval_mask], tw_tvt, tw_gr)

    md_eval = hw.loc[eval_mask, "MD"].to_numpy(dtype=float)
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    denom = max(len(eval_idx) - 1, 1)
    row_frac = np.arange(len(eval_idx), dtype=float) / denom
    pred_eval = pred_base[eval_mask]
    slope = np.gradient(pred_eval) if len(pred_eval) > 1 else np.zeros_like(pred_eval)

    return pd.DataFrame(
        {
            "row_idx": eval_idx,
            "row_frac": row_frac,
            "md_delta": md_eval - float(last["MD"]),
            "z_delta": z_eval - float(last["Z"]),
            "z_eval": z_eval,
            "gr_interp": gr_all[eval_mask],
            "gr_isna": hw.loc[eval_mask, "GR"].isna().astype(float).to_numpy(),
            "pred_tvt": pred_eval,
            "pred_delta_last": pred_eval - float(last["TVT_input"]),
            "pred_slope": slope,
            "pred_gr": pred_gr,
            "gr_residual": gr_all[eval_mask] - pred_gr,
            "alt_gap": pred_alt[eval_mask] - pred_eval,
        }
    )


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    pf = load_pf_module()
    cfg = pf.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    wells = pf.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    rows = []
    for i, wid in enumerate(wells, 1):
        hw, tw = pf.load_well(args.data_dir, wid, "train")
        eval_mask = hw["TVT_input"].isna().to_numpy()
        if not eval_mask.any():
            continue
        print(f"[{i:03d}/{len(wells):03d}] {wid}", flush=True)
        preds, meta = pf.predict_variants(hw, tw, ["grid_s3_b0_h0p18", "grid_s3_b0_h0p2"], cfg)
        feats = row_features(hw, tw, preds["grid_s3_b0_h0p18"], preds["grid_s3_b0_h0p2"])
        profile = pf.well_profile(hw)
        feats["well"] = wid
        feats["y_true"] = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        feats["base_pred"] = preds["grid_s3_b0_h0p18"][eval_mask]
        feats["residual"] = feats["y_true"] - feats["base_pred"]
        for key, value in profile.items():
            feats[key] = value
        for key, value in meta.items():
            feats[key] = value
        rows.append(feats)
    return pd.concat(rows, ignore_index=True)


def summarize_by_well(df: pd.DataFrame, pred_col: str) -> tuple[float, float, float]:
    row = rmse(df["y_true"].to_numpy(), df[pred_col].to_numpy())
    per_well = df.groupby("well").apply(lambda g: rmse(g["y_true"].to_numpy(), g[pred_col].to_numpy()))
    return row, float(per_well.mean()), float(per_well.max())


def evaluate_models(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    feature_cols = [
        "row_frac",
        "md_delta",
        "z_delta",
        "gr_interp",
        "gr_isna",
        "pred_delta_last",
        "pred_slope",
        "gr_residual",
        "alt_gap",
        "n_eval",
        "known_frac",
        "z_span",
        "gr_nan",
        "pf_entropy",
        "pf_path_std",
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["residual"].to_numpy(dtype=float)
    groups = df["well"].to_numpy()
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "huber": make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=300)),
        "histgb": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.2, random_state=7),
        "rf_small": RandomForestRegressor(n_estimators=160, max_depth=7, min_samples_leaf=80, n_jobs=-1, random_state=7),
    }

    out = df[["well", "y_true", "base_pred"]].copy()
    out["pred_base"] = out["base_pred"]
    n_splits = min(args.folds, df["well"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    for name, model in models.items():
        pred_resid = np.zeros(len(df), dtype=float)
        for tr, te in gkf.split(X, y, groups):
            model.fit(X[tr], y[tr])
            pred_resid[te] = np.clip(model.predict(X[te]), -args.clip, args.clip)
        out[f"pred_{name}"] = out["base_pred"] + pred_resid

    rows = []
    for col in [c for c in out.columns if c.startswith("pred_")]:
        row_rmse, well_mean, well_max = summarize_by_well(out.rename(columns={col: "pred"}), "pred")
        rows.append(
            {
                "model": col.replace("pred_", ""),
                "row_rmse": row_rmse,
                "well_rmse_mean": well_mean,
                "well_rmse_max": well_max,
            }
        )
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--particles", type=int, default=160)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--clip", type=float, default=12.0)
    parser.add_argument("--row-output", type=Path, default=Path("docs/pf_residual_rows.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("docs/pf_residual_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(args)
    args.row_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.row_output, index=False)
    summary = evaluate_models(df, args)
    summary.to_csv(args.summary_output, index=False)
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.row_output}")
    print(f"Saved {args.summary_output}")


if __name__ == "__main__":
    main()
