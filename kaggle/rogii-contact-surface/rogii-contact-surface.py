"""ROGII spatial contact-surface probe.

Test files do not include formation contact columns. This notebook predicts an
EGFDL contact surface from training wells in X/Y space, then estimates the
per-well TVT offset only from known TVT_input rows.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CONTACT_COL = os.getenv("ROGII_SURFACE_CONTACT", "EGFDL")
K_NEIGHBORS = int(os.getenv("ROGII_SURFACE_K", "64"))
STRIDE = int(os.getenv("ROGII_SURFACE_STRIDE", "20"))
XY_SCALE = float(os.getenv("ROGII_SURFACE_XY_SCALE", "1000.0"))
TAIL_MIN = int(os.getenv("ROGII_SURFACE_TAIL_MIN", "12"))
TAIL_MAX = int(os.getenv("ROGII_SURFACE_TAIL_MAX", "80"))
OUTPUT_PATH = Path(os.getenv("ROGII_OUTPUT_PATH", "/kaggle/working/submission.csv"))
if not OUTPUT_PATH.parent.exists():
    OUTPUT_PATH = Path("submission.csv")


def find_data_dir() -> Path:
    for candidate in [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("."),
    ]:
        if (candidate / "train").exists() and (candidate / "test").exists():
            return candidate
    hits = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError("Cannot locate competition data.")


def well_ids(data_dir: Path, split: str) -> list[str]:
    return [
        Path(path).name.split("__")[0]
        for path in sorted(glob.glob(str(data_dir / split / "*__horizontal_well.csv")))
    ]


def build_surface_pool(data_dir: Path) -> pd.DataFrame:
    test_ids = set(well_ids(data_dir, "test"))
    rows = []
    for path in sorted(glob.glob(str(data_dir / "train" / "*__horizontal_well.csv"))):
        wid = Path(path).name.split("__")[0]
        if wid in test_ids:
            continue
        try:
            hw = pd.read_csv(path, usecols=["X", "Y", CONTACT_COL])
        except ValueError:
            continue
        rows.append(hw.iloc[::STRIDE].dropna())
    if not rows:
        raise RuntimeError(f"No train rows with {CONTACT_COL} were found.")
    return pd.concat(rows, ignore_index=True)


def predict_surface(pool: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    train_xy = pool[["X", "Y"]].to_numpy(dtype=float) / XY_SCALE
    target_xy = target[["X", "Y"]].to_numpy(dtype=float) / XY_SCALE
    tree = cKDTree(train_xy)
    dist, idx = tree.query(target_xy, k=min(K_NEIGHBORS, len(pool)), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    vals = pool[CONTACT_COL].to_numpy(dtype=float)[idx]
    weights = 1.0 / np.maximum(dist, 1e-3) ** 2
    return np.sum(vals * weights, axis=1) / np.sum(weights, axis=1)


def anchor_path(hw: pd.DataFrame) -> np.ndarray:
    known = hw["TVT_input"].ffill().bfill()
    if known.notna().any():
        return known.to_numpy(dtype=float)
    return np.zeros(len(hw), dtype=float)


def tvt_from_surface(hw: pd.DataFrame, contact_pred: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    base = contact_pred - hw["Z"].to_numpy(dtype=float)
    known = hw["TVT_input"].notna().to_numpy() & np.isfinite(base)
    if int(known.sum()) < 8:
        return anchor_path(hw), {"n_known": int(known.sum()), "mode": "anchor_fallback"}

    residual = hw.loc[known, "TVT_input"].to_numpy(dtype=float) - base[known]
    n_tail = min(TAIL_MAX, max(TAIL_MIN, len(residual) // 3))
    tail = residual[-n_tail:]
    weights = np.linspace(0.35, 1.0, len(tail))
    offset = float(np.average(tail, weights=weights))
    pred = base + offset
    anchor = anchor_path(hw)
    pred = np.where(np.isfinite(pred), pred, anchor)
    return pred.astype(float), {
        "n_known": int(known.sum()),
        "n_tail": int(n_tail),
        "offset": offset,
        "contact_min": float(np.nanmin(contact_pred)),
        "contact_max": float(np.nanmax(contact_pred)),
        "pred_min": float(np.nanmin(pred)),
        "pred_max": float(np.nanmax(pred)),
    }


def build_submission(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    pool = build_surface_pool(data_dir)
    predictions: dict[str, float] = {}
    summary: dict[str, object] = {
        "contact": CONTACT_COL,
        "k": K_NEIGHBORS,
        "stride": STRIDE,
        "pool_rows": int(len(pool)),
        "wells": {},
    }

    for wid in well_ids(data_dir, "test"):
        hw = pd.read_csv(data_dir / "test" / f"{wid}__horizontal_well.csv")
        contact_pred = predict_surface(pool, hw)
        pred, meta = tvt_from_surface(hw, contact_pred)
        summary["wells"][wid] = meta
        for idx, value in enumerate(pred):
            predictions[f"{wid}_{idx}"] = float(value)

    out = sample.copy()
    out["tvt"] = out["id"].map(predictions)
    if out["tvt"].isna().any():
        missing = out.loc[out["tvt"].isna(), "id"].head(10).tolist()
        raise RuntimeError(f"Missing predictions for sample ids: {missing}")
    summary["rows"] = int(len(out))
    summary["submission_min"] = float(out["tvt"].min())
    summary["submission_max"] = float(out["tvt"].max())
    return out[["id", "tvt"]], summary


def main() -> None:
    data_dir = find_data_dir()
    submission, summary = build_submission(data_dir)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    (OUTPUT_PATH.parent / "contact_surface_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved {OUTPUT_PATH} rows={len(submission)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(submission.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
