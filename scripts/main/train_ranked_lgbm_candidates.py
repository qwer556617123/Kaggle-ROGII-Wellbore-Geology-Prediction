"""
Train final ranked LightGBM candidates and generate submission CSVs.

The candidate definitions mirror scripts/diagnostics/cv_lgbm_rank_replay.py:
train on train_features_v2.parquet, predict correction on test_features_v2,
then blend the resulting TVT curve back toward the anchor:

    final_tvt = anchor + beta * (lgbm_tvt - anchor)

The current defaults are the top two pseudo-test CV candidates from
reports/cv_lgbm_rank_replay_blend_confirm_summary.csv.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FEATURES = ROOT_DIR / "features" / "train_features_v2.parquet"
TEST_FEATURES = ROOT_DIR / "features" / "test_features_v2.parquet"
SUBMISSION_TEMPLATE = ROOT_DIR / "sample_submission.csv"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
REPORT_DIR = ROOT_DIR / "reports"

TARGET = "target_correction"
TRAIN_PHYSICS = "anchored_physics_col"
TEST_PHYSICS = "physics_tvt"
ANCHOR = "last_known_tvt"

EXCLUDE_COLS = {
    "well_id",
    "is_post_ps",
    "target_correction",
    "anchored_physics_col",
}

V13_EXCLUDE_PREFIXES = ("knn_",)
V13_EXCLUDE_EXACT = {
    "tw_gradient_at_pos",
    "gr_tw_at_physics",
    "gr_dev_from_tw",
    "tw_curvature_at_pos",
    "tw_local_gr_range",
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
class Candidate:
    name: str
    beta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final LGBM candidates and write submissions.")
    parser.add_argument("--train-features", type=Path, default=TRAIN_FEATURES)
    parser.add_argument("--test-features", type=Path, default=TEST_FEATURES)
    parser.add_argument("--sample-submission", type=Path, default=SUBMISSION_TEMPLATE)
    parser.add_argument("--num-rounds", type=int, default=1254)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidates",
        default="v13_beta_0p80:0.80,v13_beta_0p75:0.75",
        help="Comma-separated name:beta candidate definitions.",
    )
    parser.add_argument("--model-out", type=Path, default=MODELS_DIR / "ranked_lgbm_v13_base.pkl")
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSIONS_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_DIR / "ranked_lgbm_candidates_report.csv")
    return parser.parse_args()


def parse_candidates(raw: str) -> list[Candidate]:
    candidates = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, beta = item.split(":", 1)
        candidates.append(Candidate(name=name.strip(), beta=float(beta)))
    return candidates


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


def smooth_by_well(frame: pd.DataFrame, values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values
    out = np.empty_like(values, dtype=float)
    tmp = pd.DataFrame({"well_id": frame["well_id"].to_numpy(), "pred": values})
    for _wid, idx in tmp.groupby("well_id", sort=False).indices.items():
        out[idx] = gaussian_filter1d(values[idx], sigma=sigma)
    return out


def submission_ids(test: pd.DataFrame) -> pd.Series:
    rows = test["ps_idx"].astype(int) + test["rows_since_ps"].round().astype(int)
    return test["well_id"].astype(str) + "_" + rows.astype(str)


def train_model(train: pd.DataFrame, features: list[str], num_rounds: int, seed: int) -> lgb.Booster:
    params = dict(V13_PARAMS)
    params["seed"] = seed
    dataset = lgb.Dataset(
        train[features].to_numpy(dtype=np.float32),
        label=train[TARGET].to_numpy(dtype=np.float32),
        feature_name=features,
    )
    return lgb.train(
        params,
        dataset,
        num_boost_round=num_rounds,
        callbacks=[lgb.log_evaluation(250)],
    )


def write_submission(template_path: Path, test: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    sub = pd.read_csv(template_path)
    pred_map = pd.Series(pred, index=submission_ids(test)).to_dict()
    missing = set(sub["id"]) - set(pred_map)
    extra = set(pred_map) - set(sub["id"])
    if missing or extra:
        raise RuntimeError(f"Submission id mismatch: missing={len(missing)} extra={len(extra)}")

    sub["tvt"] = sub["id"].map(pred_map)
    if sub["tvt"].isna().any():
        raise RuntimeError("NaN predictions in submission.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)


def candidate_report(test: pd.DataFrame, name: str, beta: float, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    tmp = test[["well_id", ANCHOR, TEST_PHYSICS]].copy()
    tmp["pred"] = pred
    for well_id, g in tmp.groupby("well_id", sort=True):
        rows.append(
            {
                "candidate": name,
                "beta": beta,
                "well_id": well_id,
                "n_rows": len(g),
                "pred_min": float(g["pred"].min()),
                "pred_max": float(g["pred"].max()),
                "pred_start": float(g["pred"].iloc[0]),
                "pred_end": float(g["pred"].iloc[-1]),
                "pred_trend": float(g["pred"].iloc[-1] - g["pred"].iloc[0]),
                "anchor": float(g[ANCHOR].iloc[0]),
                "physics_start": float(g[TEST_PHYSICS].iloc[0]),
                "physics_end": float(g[TEST_PHYSICS].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = parse_candidates(args.candidates)

    train = pd.read_parquet(args.train_features)
    train = train[train["is_post_ps"] == 1].copy()
    test = pd.read_parquet(args.test_features)
    test = test[test["is_post_ps"] == 1].copy()

    features = v13_features(numeric_features(train))
    missing_test_features = [c for c in features if c not in test.columns]
    if missing_test_features:
        raise RuntimeError(f"Test features missing columns: {missing_test_features[:10]}")

    print(f"train rows={len(train):,} test rows={len(test):,} features={len(features)}")
    print(f"training final v13-like model for {args.num_rounds} rounds")
    model = train_model(train, features, args.num_rounds, args.seed)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": features, "num_rounds": args.num_rounds}, args.model_out)
    print(f"Saved model: {args.model_out}")

    corr = model.predict(test[features].to_numpy(dtype=np.float32), num_iteration=args.num_rounds)
    lgbm_tvt = test[TEST_PHYSICS].to_numpy(dtype=float) + corr
    lgbm_tvt = smooth_by_well(test, lgbm_tvt, args.smooth_sigma)
    anchor = test[ANCHOR].to_numpy(dtype=float)

    reports = []
    for candidate in candidates:
        pred = anchor + candidate.beta * (lgbm_tvt - anchor)
        out_path = args.submission_dir / f"{candidate.name}.csv"
        write_submission(args.sample_submission, test, pred, out_path)
        reports.append(candidate_report(test, candidate.name, candidate.beta, pred))
        print(f"Saved submission: {out_path}")

    report = pd.concat(reports, ignore_index=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_out, index=False)
    print(f"Saved report: {args.report_out}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
