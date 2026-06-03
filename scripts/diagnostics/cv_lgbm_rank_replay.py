"""
Replay candidate families on pseudo-test wells for CV/LB rank checks.

This uses train_features_v2.parquet, which contains all 773 training wells with
post-PS target_correction. Wells are split as groups, so each validation fold
simulates the official setting: unseen wells with post-PS TVT hidden.

The goal is rank evidence, not exact LB prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold


ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT_DIR / "features" / "train_features_v2.parquet"
REPORT_DIR = ROOT_DIR / "reports"

TARGET = "target_correction"
TRUE_TVT = "target_tvt_reconstructed"
ANCHOR = "last_known_tvt"
PHYSICS = "anchored_physics_col"

EXCLUDE_COLS = {
    "well_id",
    "is_post_ps",
    "target_correction",
    "anchored_physics_col",
    TRUE_TVT,
}

V13_EXCLUDE_PREFIXES = ("knn_",)
V13_EXCLUDE_EXACT = {
    "tw_gradient_at_pos",
    "gr_tw_at_physics",
    "gr_dev_from_tw",
    "tw_curvature_at_pos",
    "tw_local_gr_range",
}

V8_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.05,
    "lambda_l2": 0.05,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

V13_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.3,
    "lambda_l2": 0.3,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class ModelSpec:
    method: str
    params: dict[str, object]
    feature_set: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped pseudo-test CV for candidate rank checks.")
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--row-stride", type=int, default=1, help="Keep every Nth row per well for faster pilot runs.")
    parser.add_argument("--limit-wells", type=int, help="Optional smoke-test limit after sorting well ids.")
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument(
        "--blend-betas",
        default="0.6,0.8,1.0,1.1",
        help="Comma-separated betas for anchor + beta * (prediction - anchor).",
    )
    parser.add_argument("--out-prefix", type=Path, default=REPORT_DIR / "cv_lgbm_rank_replay")
    return parser.parse_args()


def numeric_features(df: pd.DataFrame) -> list[str]:
    features = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def v13_features(features: list[str]) -> list[str]:
    return [
        col
        for col in features
        if col not in V13_EXCLUDE_EXACT and not col.startswith(V13_EXCLUDE_PREFIXES)
    ]


def load_frame(path: Path, row_stride: int, limit_wells: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["is_post_ps"] == 1].copy()
    df[TRUE_TVT] = df[PHYSICS] + df[TARGET]

    if limit_wells:
        keep = sorted(df["well_id"].unique())[:limit_wells]
        df = df[df["well_id"].isin(keep)].copy()

    if row_stride > 1:
        df["_row_in_well"] = df.groupby("well_id").cumcount()
        df = df[df["_row_in_well"] % row_stride == 0].drop(columns=["_row_in_well"]).copy()

    return df.reset_index(drop=True)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    err = pred - true
    return float(np.sqrt(np.mean(err * err)))


def score_predictions(frame: pd.DataFrame, pred: np.ndarray, method: str, fold: int) -> tuple[dict, pd.DataFrame]:
    scored = frame[["well_id", TRUE_TVT]].copy()
    scored["pred"] = pred
    scored["se"] = (scored["pred"] - scored[TRUE_TVT]) ** 2

    per_well = (
        scored.groupby("well_id", as_index=False)
        .agg(sse=("se", "sum"), n_rows=("se", "size"))
        .assign(rmse=lambda x: np.sqrt(x["sse"] / x["n_rows"]))
    )
    row = {
        "fold": fold,
        "method": method,
        "row_rmse": rmse(pred, frame[TRUE_TVT].to_numpy()),
        "per_well_rmse": float(per_well["rmse"].mean()),
        "n_wells": int(per_well.shape[0]),
        "n_rows": int(len(frame)),
    }
    per_well["fold"] = fold
    per_well["method"] = method
    return row, per_well


def smooth_by_well(frame: pd.DataFrame, values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values
    out = np.empty_like(values, dtype=float)
    tmp = pd.DataFrame({"well_id": frame["well_id"].to_numpy(), "pred": values})
    for _wid, idx in tmp.groupby("well_id", sort=False).indices.items():
        out[idx] = gaussian_filter1d(values[idx], sigma=sigma)
    return out


def train_predict_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    spec: ModelSpec,
    max_rounds: int,
    early_stopping: int,
    smooth_sigma: float,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    params = dict(spec.params)
    params["seed"] = seed
    lgb_train = lgb.Dataset(
        train[features].to_numpy(dtype=np.float32),
        label=train[TARGET].to_numpy(dtype=np.float32),
        feature_name=features,
    )
    lgb_valid = lgb.Dataset(
        valid[features].to_numpy(dtype=np.float32),
        label=valid[TARGET].to_numpy(dtype=np.float32),
        feature_name=features,
        reference=lgb_train,
    )
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=max_rounds,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    corr = model.predict(valid[features].to_numpy(dtype=np.float32), num_iteration=model.best_iteration)
    pred = valid[PHYSICS].to_numpy(dtype=float) + corr
    pred = smooth_by_well(valid, pred, smooth_sigma)
    return pred, int(model.best_iteration), float(model.best_score["valid_0"]["rmse"])


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("method", as_index=False)
        .agg(
            row_rmse_mean=("row_rmse", "mean"),
            row_rmse_std=("row_rmse", "std"),
            per_well_rmse_mean=("per_well_rmse", "mean"),
            per_well_rmse_std=("per_well_rmse", "std"),
            folds=("fold", "count"),
            n_rows_mean=("n_rows", "mean"),
        )
        .sort_values("row_rmse_mean")
    )


def parse_betas(raw: str) -> list[float]:
    betas = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            betas.append(float(part))
    return list(dict.fromkeys(betas))


def blend_name(method: str, beta: float) -> str:
    text = f"{beta:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return method if abs(beta - 1.0) < 1e-12 else f"{method}_anchor_beta_{text}"


def main() -> None:
    args = parse_args()
    blend_betas = parse_betas(args.blend_betas)
    df = load_frame(args.feature_path, args.row_stride, args.limit_wells)
    wells = np.array(sorted(df["well_id"].unique()))
    if len(wells) < args.folds:
        raise RuntimeError(f"Need at least {args.folds} wells, found {len(wells)}")

    all_features = numeric_features(df)
    feature_sets = {
        "v13_base": v13_features(all_features),
        "all_features": all_features,
    }
    specs = [
        ModelSpec("lgbm_v8_like_base", V8_PARAMS, "v13_base"),
        ModelSpec("lgbm_v13_like_base", V13_PARAMS, "v13_base"),
        ModelSpec("lgbm_extra_features", V13_PARAMS, "all_features"),
    ]

    print(f"rows={len(df):,} wells={len(wells)} row_stride={args.row_stride}")
    print(f"features: v13_base={len(feature_sets['v13_base'])}, all_features={len(all_features)}")

    rows = []
    per_well_frames = []
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(wells), start=1):
        train_wells = set(wells[tr_idx])
        valid_wells = set(wells[vl_idx])
        train = df[df["well_id"].isin(train_wells)].copy()
        valid = df[df["well_id"].isin(valid_wells)].copy()
        print(f"\nfold={fold} train_wells={len(train_wells)} valid_wells={len(valid_wells)}")

        deterministic = {
            "anchor": valid[ANCHOR].to_numpy(dtype=float),
            "physics": valid[PHYSICS].to_numpy(dtype=float),
            "alpha_0p07": valid[ANCHOR].to_numpy(dtype=float)
            + 0.07 * (valid[PHYSICS].to_numpy(dtype=float) - valid[ANCHOR].to_numpy(dtype=float)),
            "pseudo_oracle_lookup": valid[TRUE_TVT].to_numpy(dtype=float),
        }
        for method, pred in deterministic.items():
            row, per_well = score_predictions(valid, pred, method, fold)
            rows.append(row)
            per_well_frames.append(per_well)
            print(f"  {method:<22} row_rmse={row['row_rmse']:.4f} per_well={row['per_well_rmse']:.4f}")

        for spec in specs:
            features = feature_sets[spec.feature_set]
            pred, best_iter, val_corr_rmse = train_predict_lgbm(
                train,
                valid,
                features,
                spec,
                args.max_rounds,
                args.early_stopping,
                args.smooth_sigma,
                args.seed + fold,
            )
            anchor = valid[ANCHOR].to_numpy(dtype=float)
            for beta in blend_betas:
                blended = anchor + beta * (pred - anchor)
                method = blend_name(spec.method, beta)
                row, per_well = score_predictions(valid, blended, method, fold)
                row["best_iter"] = best_iter
                row["valid_correction_rmse"] = val_corr_rmse
                row["n_features"] = len(features)
                row["blend_beta"] = beta
                rows.append(row)
                per_well_frames.append(per_well)
                print(
                    f"  {method:<38} row_rmse={row['row_rmse']:.4f} "
                    f"per_well={row['per_well_rmse']:.4f} iter={best_iter}"
                )

    summary = pd.DataFrame(rows)
    aggregate_df = aggregate(summary)
    per_well_df = pd.concat(per_well_frames, ignore_index=True)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    folds_path = args.out_prefix.with_name(args.out_prefix.name + "_folds.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    per_well_path = args.out_prefix.with_name(args.out_prefix.name + "_per_well.csv")
    summary.to_csv(folds_path, index=False)
    aggregate_df.to_csv(summary_path, index=False)
    per_well_df.to_csv(per_well_path, index=False)

    print("\nAggregate:")
    print(aggregate_df.to_string(index=False))
    print(f"\nSaved: {folds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_well_path}")


if __name__ == "__main__":
    main()
