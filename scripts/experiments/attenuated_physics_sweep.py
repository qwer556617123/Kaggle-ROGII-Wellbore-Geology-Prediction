"""
Attenuated anchored-physics baseline.

The current LGBM family often over-trusts post-PS trajectory. This script tests
a simpler hypothesis:

    prediction = anchor_tvt + alpha * (anchored_physics - anchor_tvt)

alpha=0 is the constant-anchor baseline.
alpha=1 is raw anchored physics.

The script also supports per-well alpha values, because the oracle diagnostic
shows the three public wells prefer different amounts of trend attenuation.

If train truth for the same well IDs exists locally, the script writes an oracle
sweep report. The default submission uses the user-provided alpha and does not
use oracle truth to choose it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
SUBS_DIR = DATA_DIR / "submissions"
DEFAULT_WELLS = ("000d7d20", "00bbac68", "00e12e8b")


def get_ps(hw: pd.DataFrame) -> int:
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def tvt_input(hw: pd.DataFrame) -> pd.Series:
    return hw["TVT_input"].astype(str).replace("", np.nan).astype(float)


def anchored_physics(hw: pd.DataFrame, ps: int) -> tuple[float, float, np.ndarray]:
    z = hw["Z"].astype(float).to_numpy()
    ti = tvt_input(hw)
    known = ti.iloc[:ps].dropna()
    anchor = float(ti.ffill().iloc[max(0, ps - 1)])
    z_anchor = float(z[max(0, ps - 1)])

    if len(known) > 5:
        pre_z = z[known.index.to_numpy()]
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), known.to_numpy())
        slope = float(reg.coef_[0])
    else:
        slope = -1.0

    physics = anchor + slope * (z[ps:] - z_anchor)
    return anchor, slope, physics


def prediction_for_alpha(hw: pd.DataFrame, alpha: float) -> tuple[int, float, float, np.ndarray, np.ndarray]:
    ps = get_ps(hw)
    anchor, slope, physics = anchored_physics(hw, ps)
    pred = anchor + alpha * (physics - anchor)
    return ps, anchor, slope, physics, pred


def truth_for_well(well_id: str, ps: int, n_post: int) -> np.ndarray | None:
    path = TRAIN_DIR / f"{well_id}__horizontal_well.csv"
    if not path.exists():
        return None
    hw = pd.read_csv(path)
    if "TVT" not in hw.columns or len(hw) < ps + n_post:
        return None
    return hw["TVT"].astype(float).to_numpy()[ps : ps + n_post]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_alpha_map(items: list[str] | None) -> dict[str, float]:
    alpha_map: dict[str, float] = {}
    if not items:
        return alpha_map
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected WELL=ALPHA, got: {item}")
        well_id, value = item.split("=", 1)
        alpha_map[well_id.strip()] = float(value)
    return alpha_map


def build_submission(
    wells: list[str],
    alpha: float,
    alpha_map: dict[str, float],
    output: Path,
) -> pd.DataFrame:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    for well_id in wells:
        hw = pd.read_csv(TEST_DIR / f"{well_id}__horizontal_well.csv")
        well_alpha = alpha_map.get(well_id, alpha)
        ps, _anchor, _slope, _physics, pred = prediction_for_alpha(hw, well_alpha)
        ids = [f"{well_id}_{i}" for i in range(ps, ps + len(pred))]
        mapping = dict(zip(ids, pred))
        mask = sub["id"].isin(mapping.keys())
        sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(mapping)

    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False)
    return sub


def oracle_sweep(wells: list[str], alphas: np.ndarray) -> pd.DataFrame:
    rows = []
    per_well_cache = {}
    for well_id in wells:
        hw = pd.read_csv(TEST_DIR / f"{well_id}__horizontal_well.csv")
        ps, anchor, slope, physics, _pred = prediction_for_alpha(hw, 0.0)
        true = truth_for_well(well_id, ps, len(physics))
        if true is None:
            continue
        per_well_cache[well_id] = (ps, anchor, slope, physics, true)

        best_alpha = None
        best_rmse = float("inf")
        for alpha in alphas:
            pred = anchor + alpha * (physics - anchor)
            score = rmse(pred, true)
            if score < best_rmse:
                best_rmse = score
                best_alpha = float(alpha)

        rows.append(
            {
                "scope": "per_well_best",
                "well_id": well_id,
                "alpha": best_alpha,
                "rmse": best_rmse,
                "anchor_rmse": rmse(np.full_like(true, anchor), true),
                "physics_rmse": rmse(physics, true),
                "true_net": float(true[-1] - true[0]),
                "physics_net": float(physics[-1] - physics[0]),
                "n_post": len(true),
            }
        )

    if per_well_cache:
        best_alpha = None
        best_rmse = float("inf")
        for alpha in alphas:
            sq_sum = 0.0
            n_sum = 0
            for _well_id, (_ps, anchor, _slope, physics, true) in per_well_cache.items():
                pred = anchor + alpha * (physics - anchor)
                sq_sum += float(np.sum((pred - true) ** 2))
                n_sum += len(true)
            score = float(np.sqrt(sq_sum / n_sum))
            if score < best_rmse:
                best_rmse = score
                best_alpha = float(alpha)
        rows.append({"scope": "global_best", "well_id": "ALL", "alpha": best_alpha, "rmse": best_rmse})

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attenuated anchored-physics submissions.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--wells", nargs="+", default=list(DEFAULT_WELLS))
    parser.add_argument("--alpha", type=float, default=0.06)
    parser.add_argument(
        "--per-well-alpha",
        nargs="+",
        help="Optional WELL=ALPHA overrides, e.g. 000d7d20=0.115 00bbac68=0.075.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.20)
    parser.add_argument("--alpha-step", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "submissions" / "attenuated_physics_alpha_0p06.csv")
    parser.add_argument("--report", type=Path, default=DATA_DIR / "submissions" / "attenuated_physics_sweep.csv")
    parser.add_argument("--no-oracle-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_DIR, TRAIN_DIR, TEST_DIR, SUBS_DIR
    DATA_DIR = args.data_dir
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    SUBS_DIR = DATA_DIR / "submissions"

    alpha_map = parse_alpha_map(args.per_well_alpha)
    sub = build_submission(args.wells, args.alpha, alpha_map, args.output)
    print(f"Saved submission: {args.output}")
    print(f"Submission rows={len(sub)} NaN={sub['tvt'].isna().sum()} default_alpha={args.alpha:.4f}")
    if alpha_map:
        for well_id in args.wells:
            print(f"  alpha[{well_id}]={alpha_map.get(well_id, args.alpha):.4f}")

    if not args.no_oracle_sweep:
        alphas = np.arange(args.alpha_min, args.alpha_max + 1e-12, args.alpha_step)
        report = oracle_sweep(args.wells, alphas)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report, index=False)
        print(f"Saved oracle sweep report: {args.report}")
        if not report.empty:
            print(report.to_string(index=False))


if __name__ == "__main__":
    main()
