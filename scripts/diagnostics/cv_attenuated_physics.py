"""
Cross-validation for attenuated anchored physics.

This simulates the test-time setting on training wells:

  - use only pre-PS TVT_input to fit anchored physics;
  - predict post-PS TVT with anchor + alpha * (physics - anchor);
  - evaluate against the hidden training TVT post-PS.

It reports anchor, raw physics, fixed alpha values, and a fold-trained global
alpha chosen only from the training folds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
REPORT_DIR = DATA_DIR / "reports"


@dataclass
class WellCurve:
    well_id: str
    n_post: int
    anchor: float
    physics: np.ndarray
    true: np.ndarray
    z_net: float
    post_gr_std: float
    pre_gr_std: float


def get_ps(hw: pd.DataFrame) -> int:
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def tvt_input(hw: pd.DataFrame) -> pd.Series:
    return hw["TVT_input"].astype(str).replace("", np.nan).astype(float)


def fill_float(values) -> np.ndarray:
    return pd.Series(values).ffill().bfill().astype(float).to_numpy()


def load_curve(well_id: str) -> WellCurve | None:
    hw_path = TRAIN_DIR / f"{well_id}__horizontal_well.csv"
    if not hw_path.exists():
        return None
    hw = pd.read_csv(hw_path)
    if "TVT" not in hw.columns:
        return None

    ps = get_ps(hw)
    if ps >= len(hw) - 5:
        return None

    z = hw["Z"].astype(float).to_numpy()
    ti = tvt_input(hw)
    known = ti.iloc[:ps].dropna()
    if len(known) < 8:
        return None

    anchor = float(ti.ffill().iloc[max(0, ps - 1)])
    z_anchor = float(z[max(0, ps - 1)])
    pre_z = z[known.index.to_numpy()]
    reg = LinearRegression().fit(pre_z.reshape(-1, 1), known.to_numpy())
    slope = float(reg.coef_[0])
    physics = anchor + slope * (z[ps:] - z_anchor)
    true = hw["TVT"].astype(float).to_numpy()[ps:]

    gr = fill_float(hw["GR"])
    return WellCurve(
        well_id=well_id,
        n_post=len(true),
        anchor=anchor,
        physics=physics,
        true=true,
        z_net=float(z[-1] - z_anchor),
        post_gr_std=float(np.std(gr[ps:])),
        pre_gr_std=float(np.std(gr[:ps])),
    )


def all_well_ids(limit: int | None = None) -> list[str]:
    ids = sorted(p.name.split("__")[0] for p in TRAIN_DIR.glob("*__horizontal_well.csv"))
    if limit:
        return ids[:limit]
    return ids


def rmse_from_sse(sse: float, n: int) -> float:
    return float(np.sqrt(sse / max(1, n)))


def score_alpha(curves: list[WellCurve], alpha: float) -> tuple[float, float, pd.DataFrame]:
    rows = []
    sse = 0.0
    n = 0
    for c in curves:
        pred = c.anchor + alpha * (c.physics - c.anchor)
        err = pred - c.true
        well_rmse = float(np.sqrt(np.mean(err * err)))
        rows.append(
            {
                "well_id": c.well_id,
                "alpha": alpha,
                "rmse": well_rmse,
                "n_post": c.n_post,
                "z_net": c.z_net,
                "pre_gr_std": c.pre_gr_std,
                "post_gr_std": c.post_gr_std,
                "true_net": float(c.true[-1] - c.true[0]),
                "physics_net": float(c.physics[-1] - c.physics[0]),
            }
        )
        sse += float(np.sum(err * err))
        n += c.n_post
    df = pd.DataFrame(rows)
    return rmse_from_sse(sse, n), float(df["rmse"].mean()), df


def best_global_alpha(curves: list[WellCurve], alphas: np.ndarray) -> tuple[float, float]:
    best_alpha = float(alphas[0])
    best_score = float("inf")
    for alpha in alphas:
        row_rmse, _pw, _df = score_alpha(curves, float(alpha))
        if row_rmse < best_score:
            best_score = row_rmse
            best_alpha = float(alpha)
    return best_alpha, best_score


def run_cv(curves: list[WellCurve], alphas: np.ndarray, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    summary_rows = []
    detail_frames = []
    curves_arr = np.array(curves, dtype=object)

    fixed_alphas = [0.0, 0.04, 0.07, 0.10, 0.15, 1.0]
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(curves_arr), start=1):
        train_curves = list(curves_arr[tr_idx])
        val_curves = list(curves_arr[vl_idx])
        learned_alpha, train_rmse = best_global_alpha(train_curves, alphas)

        eval_alphas = list(dict.fromkeys(fixed_alphas + [learned_alpha]))
        for alpha in eval_alphas:
            row_rmse, per_well_rmse, detail = score_alpha(val_curves, float(alpha))
            label = "learned_alpha" if abs(alpha - learned_alpha) < 1e-12 else f"alpha_{alpha:.3f}"
            summary_rows.append(
                {
                    "fold": fold,
                    "method": label,
                    "alpha": float(alpha),
                    "train_selected_alpha": learned_alpha,
                    "train_selected_rmse": train_rmse,
                    "row_rmse": row_rmse,
                    "per_well_rmse": per_well_rmse,
                    "n_wells": len(val_curves),
                    "n_rows": int(detail["n_post"].sum()),
                }
            )
            detail = detail.copy()
            detail["fold"] = fold
            detail["method"] = label
            detail_frames.append(detail)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True)
    return summary, details


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("method", as_index=False)
        .agg(
            alpha_mean=("alpha", "mean"),
            row_rmse_mean=("row_rmse", "mean"),
            row_rmse_std=("row_rmse", "std"),
            per_well_rmse_mean=("per_well_rmse", "mean"),
            per_well_rmse_std=("per_well_rmse", "std"),
            folds=("fold", "count"),
        )
        .sort_values("row_rmse_mean")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV attenuation baselines on held-out train wells.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-wells", type=int)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.4)
    parser.add_argument("--alpha-step", type=float, default=0.005)
    parser.add_argument("--out-prefix", type=Path, default=REPORT_DIR / "cv_attenuated_physics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_DIR, TRAIN_DIR, REPORT_DIR
    DATA_DIR = args.data_dir
    TRAIN_DIR = DATA_DIR / "train"
    REPORT_DIR = DATA_DIR / "reports"

    ids = all_well_ids(args.limit_wells)
    curves = []
    for i, well_id in enumerate(ids, start=1):
        c = load_curve(well_id)
        if c is not None:
            curves.append(c)
        if i % 150 == 0:
            print(f"loaded {i}/{len(ids)} wells")

    print(f"usable wells: {len(curves)} / {len(ids)}")
    if len(curves) < args.folds:
        raise RuntimeError("Not enough usable wells for requested CV folds.")

    alphas = np.arange(args.alpha_min, args.alpha_max + 1e-12, args.alpha_step)
    summary, details = run_cv(curves, alphas, args.folds, args.seed)
    aggregate = aggregate_summary(summary)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_folds.csv")
    details_path = args.out_prefix.with_name(args.out_prefix.name + "_details.csv")
    aggregate_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    print("\nCV summary:")
    print(aggregate.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {details_path}")
    print(f"Saved: {aggregate_path}")


if __name__ == "__main__":
    main()
