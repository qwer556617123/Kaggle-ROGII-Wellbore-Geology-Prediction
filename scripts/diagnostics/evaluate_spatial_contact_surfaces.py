"""Evaluate test-safe spatial contact-surface TVT reconstruction.

Train horizontal wells include formation contact columns, but test wells do not.
This diagnostic withholds one train well, predicts its contact surface from
other wells in X/Y space, and then estimates the per-well TVT offset only from
known TVT_input rows.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional Kaggle/local dependency
    LGBMRegressor = None


CONTACT_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true.astype(float) - y_pred.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def well_ids(data_dir: Path, split: str = "train") -> list[str]:
    return [
        os.path.basename(path).split("__")[0]
        for path in sorted(glob.glob(str(data_dir / split / "*__horizontal_well.csv")))
    ]


def load_horizontal(data_dir: Path, wid: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")


def make_hidden_mask(hw: pd.DataFrame, known_frac: float, min_known: int, min_eval: int) -> tuple[pd.DataFrame, np.ndarray]:
    truth_mask = hw["TVT"].notna().to_numpy()
    truth_idx = np.where(truth_mask)[0]
    if len(truth_idx) < min_known + min_eval:
        raise ValueError("Not enough TVT truth rows for requested mask.")
    first = int(truth_idx[0])
    last = int(truth_idx[-1])
    span = last - first + 1
    cut = first + int(round(span * float(known_frac)))
    cut = int(np.clip(cut, first + min_known - 1, last - min_eval))

    out = hw.copy()
    out["TVT_input"] = out["TVT"].where(np.arange(len(out)) <= cut, np.nan)
    eval_mask = (np.arange(len(out)) > cut) & truth_mask
    return out, eval_mask


def load_surface_pool(data_dir: Path, exclude: set[str], stride: int, max_rows: int | None, seed: int) -> pd.DataFrame:
    rows = []
    cols = ["well", "X", "Y", *CONTACT_COLS]
    for wid in well_ids(data_dir, "train"):
        if wid in exclude:
            continue
        hw = load_horizontal(data_dir, wid)
        missing = [c for c in CONTACT_COLS if c not in hw.columns]
        if missing:
            continue
        slim = hw.loc[::stride, ["X", "Y", *CONTACT_COLS]].copy()
        slim.insert(0, "well", wid)
        rows.append(slim)
    if not rows:
        raise RuntimeError("No contact surface pool rows were loaded.")
    pool = pd.concat(rows, ignore_index=True)
    pool = pool.dropna(subset=["X", "Y", *CONTACT_COLS])
    if max_rows and len(pool) > max_rows:
        pool = pool.sample(n=max_rows, random_state=seed)
    return pool.reset_index(drop=True)


def predict_contact_surface(pool: pd.DataFrame, target: pd.DataFrame, contact_col: str, k: int, xy_scale: float) -> np.ndarray:
    train_xy = pool[["X", "Y"]].to_numpy(dtype=float) / xy_scale
    target_xy = target[["X", "Y"]].to_numpy(dtype=float) / xy_scale
    tree = cKDTree(train_xy)
    dist, idx = tree.query(target_xy, k=min(k, len(pool)), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    vals = pool[contact_col].to_numpy(dtype=float)[idx]
    weights = 1.0 / np.maximum(dist, 1e-3) ** 2
    return np.sum(vals * weights, axis=1) / np.sum(weights, axis=1)


def spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    return pd.DataFrame({
        "x": x,
        "y": y,
        "x2": x * x,
        "y2": y * y,
        "xy": x * y,
    })


def predict_contact_surface_lgbm(
    pool: pd.DataFrame,
    target: pd.DataFrame,
    contact_col: str,
    rounds: int,
    seed: int,
) -> np.ndarray:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed.")
    model = LGBMRegressor(
        objective="regression",
        n_estimators=rounds,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(spatial_features(pool), pool[contact_col].to_numpy(dtype=float))
    return model.predict(spatial_features(target)).astype(float)


def offset_predict(hw: pd.DataFrame, contact_pred: np.ndarray, mode: str) -> np.ndarray:
    base = contact_pred - hw["Z"].to_numpy(dtype=float)
    known = hw["TVT_input"].notna().to_numpy() & np.isfinite(base)
    if int(known.sum()) < 8:
        return hw["TVT_input"].ffill().bfill().to_numpy(dtype=float)

    residual = hw.loc[known, "TVT_input"].to_numpy(dtype=float) - base[known]
    if mode == "tail_linear":
        known_idx = np.where(known)[0]
        n_tail = min(220, max(30, len(residual) // 2))
        tail_idx = known_idx[-n_tail:]
        x = hw.loc[tail_idx, "MD"].to_numpy(dtype=float)
        x0 = float(x[-1])
        x_scale = max(float(np.nanstd(x)), 1.0)
        x_norm = (x - x0) / x_scale
        y = residual[-n_tail:]
        slope, intercept = np.polyfit(x_norm, y, deg=1)
        all_x = (hw["MD"].to_numpy(dtype=float) - x0) / x_scale
        correction = intercept + slope * all_x
        # Prevent wild extrapolation from a noisy known tail.
        med = float(np.median(y))
        correction = np.clip(correction, med - 80.0, med + 80.0)
        pred = base + correction
        anchor = hw["TVT_input"].ffill().bfill().to_numpy(dtype=float)
        return np.where(np.isfinite(pred), pred, anchor)

    if mode.startswith("tail"):
        n_tail = min(80, max(12, len(residual) // 3))
        residual = residual[-n_tail:]
    if mode.endswith("mean"):
        weights = np.linspace(0.35, 1.0, len(residual))
        offset = float(np.average(residual, weights=weights))
    else:
        offset = float(np.median(residual))
    pred = base + offset
    anchor = hw["TVT_input"].ffill().bfill().to_numpy(dtype=float)
    return np.where(np.isfinite(pred), pred, anchor)


def select_wells(data_dir: Path, limit: int | None) -> list[str]:
    test_ids = set(well_ids(data_dir, "test"))
    ids = [wid for wid in well_ids(data_dir, "train") if wid not in test_ids]
    return ids[:limit] if limit else ids


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in details.groupby("variant", sort=True):
        weighted = np.sqrt(np.average(g["rmse"] ** 2, weights=g["n_eval"]))
        rows.append({
            "variant": variant,
            "n_masks": int(len(g)),
            "n_wells": int(g["well"].nunique()),
            "n_rows": int(g["n_eval"].sum()),
            "row_rmse": float(weighted),
            "well_rmse_mean": float(g["rmse"].mean()),
            "well_rmse_median": float(g["rmse"].median()),
            "well_rmse_p75": float(g["rmse"].quantile(0.75)),
            "well_rmse_max": float(g["rmse"].max()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--well-limit", type=int, default=24)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--contacts", default="EGFDU,EGFDL,ASTNL,BUDA")
    parser.add_argument("--k-values", default="8,16,32,64")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-pool-rows", type=int, default=250000)
    parser.add_argument("--xy-scale", type=float, default=1000.0)
    parser.add_argument("--model", choices=["knn", "lgbm"], default="knn")
    parser.add_argument("--lgb-rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/spatial_contact_surface_summary.csv"))
    parser.add_argument("--details-output", type=Path, default=Path("docs/spatial_contact_surface_details.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contacts = [c.strip() for c in args.contacts.split(",") if c.strip()]
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = select_wells(args.data_dir, args.well_limit)
    rows = []

    for wi, wid in enumerate(wells, 1):
        print(f"[{wi:03d}/{len(wells):03d}] build pool excluding {wid}", flush=True)
        pool = load_surface_pool(args.data_dir, {wid}, args.stride, args.max_pool_rows, args.seed)
        hw_raw = load_horizontal(args.data_dir, wid)
        for contact in contacts:
            if contact not in pool.columns or contact not in hw_raw.columns:
                continue
            model_keys = k_values if args.model == "knn" else [0]
            for k in model_keys:
                if args.model == "knn":
                    contact_pred = predict_contact_surface(pool, hw_raw, contact, k, args.xy_scale)
                    model_name = f"k{k}"
                else:
                    contact_pred = predict_contact_surface_lgbm(pool, hw_raw, contact, args.lgb_rounds, args.seed)
                    model_name = f"lgb{args.lgb_rounds}"
                for mode in ["tail_mean", "tail_median", "tail_linear"]:
                    variant = f"{contact}_{model_name}_{mode}"
                    for known_frac in known_fracs:
                        try:
                            hw, eval_mask = make_hidden_mask(hw_raw, known_frac, args.min_known, args.min_eval)
                        except ValueError as exc:
                            print(f"[skip] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                            continue
                        pred = offset_predict(hw, contact_pred, mode)
                        y_true = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
                        y_pred = pred[eval_mask]
                        rows.append({
                            "well": wid,
                            "known_frac": float(known_frac),
                            "variant": variant,
                            "contact": contact,
                            "k": int(k),
                            "mode": mode,
                            "rmse": rmse(y_true, y_pred),
                            "n_eval": int(eval_mask.sum()),
                        })

    if not rows:
        raise RuntimeError("No evaluations were produced.")
    details = pd.DataFrame(rows)
    summary = summarize(details)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print("\nSummary")
    print(summary.head(24).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.details_output}")
    print(f"Saved {args.summary_output}")


if __name__ == "__main__":
    main()
