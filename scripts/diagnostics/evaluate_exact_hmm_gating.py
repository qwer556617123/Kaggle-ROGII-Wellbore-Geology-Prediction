"""Calibrate exact-HMM blend weights and posterior-std gating on native masks."""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


WEIGHTS = (0.0, 0.10, 0.15, 0.20, 0.225, 0.25, 0.30, 0.35)
MEAN_WEIGHTS = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50)
SHAPE_WEIGHTS = (0.15, 0.25, 0.35, 0.50, 0.75)


def load_hmm_module(notebook_path: Path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][58]["source"])
    if "def run_hmm2" not in source or "class HMMParams" not in source:
        raise RuntimeError("exact HMM core not found in source cell 58")
    name = "exact_hmm_native_runtime"
    module = types.ModuleType(name)
    module.__file__ = str(notebook_path)
    sys.modules[name] = module
    exec(compile(source, str(notebook_path), "exec"), module.__dict__)
    return module


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def select_wells(anchor: pd.DataFrame, limit: int, seed: int) -> list[str]:
    wells = sorted(anchor["id"].str.rsplit("_", n=1).str[0].unique())
    if 0 < limit < len(wells):
        rng = np.random.default_rng(seed)
        wells = sorted(rng.choice(wells, size=limit, replace=False).tolist())
    return wells


def evaluate_one(
    well: str,
    data_dir: Path,
    anchor_map: dict[str, float],
    hmm: Any,
    emission_variants: bool,
    student_only: bool,
    student_grid: bool,
) -> dict[str, Any]:
    started = time.time()
    hw = pd.read_csv(data_dir / "train" / f"{well}__horizontal_well.csv")
    tw = pd.read_csv(data_dir / "train" / f"{well}__typewell.csv")[["TVT", "GR"]]
    result = hmm.run_hmm2(hw[hmm.TEST_COLS].copy(), tw)
    eval_mask = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    rows = np.flatnonzero(eval_mask)
    ids = [f"{well}_{row}" for row in rows]
    anchor = np.asarray([anchor_map.get(row_id, np.nan) for row_id in ids], dtype=float)
    if not np.isfinite(anchor).all():
        raise RuntimeError(f"anchor is incomplete for {well}")
    truth = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
    pred = np.asarray(result["pred"], dtype=float)[eval_mask]
    std = np.asarray(result["std_eval"], dtype=float)
    variant_preds = {"gauss_std": pred}
    variant_stds = {"gauss_std": std}
    if emission_variants or student_only or student_grid:
        variants = [
            ("student_std", {"emission": "t", "sigma_mode": "std"}),
        ]
        if emission_variants:
            variants.extend([
            ("gauss_affine_mad", {"emission": "gauss", "sigma_mode": "mad"}),
            ("student_affine_mad", {"emission": "t", "sigma_mode": "mad"}),
            ])
        if student_grid:
            variants.extend([
                ("student_lam050", {"emission": "t", "sigma_mode": "std", "lam": 0.50}),
                ("student_lam075", {"emission": "t", "sigma_mode": "std", "lam": 0.75}),
                ("student_lam125", {"emission": "t", "sigma_mode": "std", "lam": 1.25}),
                ("student_lam150", {"emission": "t", "sigma_mode": "std", "lam": 1.50}),
                ("student_df2", {"emission": "t", "sigma_mode": "std", "df": 2.0}),
                ("student_df8", {"emission": "t", "sigma_mode": "std", "df": 8.0}),
            ])
        for name, param_values in variants:
            params = hmm.HMMParams(**param_values)
            variant = hmm.run_hmm2(hw[hmm.TEST_COLS].copy(), tw, params=params)
            variant_preds[name] = np.asarray(variant["pred"], dtype=float)[eval_mask]
            variant_stds[name] = np.asarray(variant["std_eval"], dtype=float)
    return {
        "well": well,
        "truth": truth,
        "anchor": anchor,
        "hmm": pred,
        "std": std,
        "variant_preds": variant_preds,
        "variant_stds": variant_stds,
        "runtime_sec": float(time.time() - started),
    }


def summarize(records: list[dict[str, Any]], pred: np.ndarray) -> dict[str, float]:
    truth = np.concatenate([record["truth"] for record in records])
    offsets = np.cumsum([0] + [len(record["truth"]) for record in records])
    well_rmse = np.asarray(
        [rmse(record["truth"], pred[offsets[i] : offsets[i + 1]]) for i, record in enumerate(records)]
    )
    return {
        "row_rmse": rmse(truth, pred),
        "well_mean": float(well_rmse.mean()),
        "well_median": float(np.median(well_rmse)),
        "well_p90": float(np.quantile(well_rmse, 0.90)),
    }


def gated_weight(std: np.ndarray, max_weight: float = 0.25) -> np.ndarray:
    """Shrink smoothly from max weight at <=1.25 ft to 20% at >=3.5 ft."""
    confidence = np.clip((3.5 - std) / (3.5 - 1.25), 0.0, 1.0)
    return max_weight * (0.20 + 0.80 * confidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--anchor", type=Path, default=Path("reports/native100_pf_anchor.csv"))
    parser.add_argument(
        "--hmm-notebook",
        type=Path,
        default=Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb"),
    )
    parser.add_argument("--well-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--emission-variants", action="store_true")
    parser.add_argument("--student-only", action="store_true")
    parser.add_argument("--student-grid", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/exact_hmm_native_gate.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    anchor_frame = pd.read_csv(args.anchor, usecols=["id", "tvt"])
    anchor_frame["id"] = anchor_frame["id"].astype(str)
    anchor_map = dict(zip(anchor_frame["id"], anchor_frame["tvt"].astype(float)))
    wells = select_wells(anchor_frame, args.well_limit, args.seed)
    hmm = load_hmm_module(args.hmm_notebook)
    records = Parallel(n_jobs=min(args.workers, len(wells)), prefer="threads")(
        delayed(evaluate_one)(
            well,
            args.data_dir,
            anchor_map,
            hmm,
            args.emission_variants,
            args.student_only,
            args.student_grid,
        )
        for well in wells
    )
    truth = np.concatenate([record["truth"] for record in records])
    anchor = np.concatenate([record["anchor"] for record in records])
    hmm_pred = np.concatenate([record["hmm"] for record in records])
    std = np.concatenate([record["std"] for record in records])
    direction = hmm_pred - anchor

    candidates: dict[str, dict[str, float]] = {}
    for weight in WEIGHTS:
        candidates[f"constant_{weight:.3f}"] = summarize(records, anchor + weight * direction)
    row_weight = gated_weight(std)
    candidates["std_gated_025"] = summarize(records, anchor + row_weight * direction)

    emission_candidates: dict[str, dict[str, dict[str, float]]] = {}
    decomposition_candidates: dict[str, dict[str, dict[str, float]]] = {}
    uncertainty_candidates: dict[str, dict[str, dict[str, float]]] = {}
    variant_names = sorted(records[0]["variant_preds"])
    for variant_name in variant_names:
        variant_pred = np.concatenate(
            [record["variant_preds"][variant_name] for record in records]
        )
        variant_std = np.concatenate(
            [record["variant_stds"][variant_name] for record in records]
        )
        variant_direction = variant_pred - anchor
        emission_candidates[variant_name] = {
            f"{weight:.3f}": summarize(records, anchor + weight * variant_direction)
            for weight in WEIGHTS
        }
        mean_parts = []
        shape_parts = []
        for record in records:
            well_direction = record["variant_preds"][variant_name] - record["anchor"]
            well_mean = float(np.mean(well_direction))
            mean_parts.append(np.full(len(well_direction), well_mean, dtype=float))
            shape_parts.append(well_direction - well_mean)
        mean_direction = np.concatenate(mean_parts)
        shape_direction = np.concatenate(shape_parts)
        variant_decomposition = {}
        for mean_weight in MEAN_WEIGHTS:
            for shape_weight in SHAPE_WEIGHTS:
                key = f"mean_{mean_weight:.3f}_shape_{shape_weight:.3f}"
                variant_decomposition[key] = summarize(
                    records,
                    anchor + mean_weight * mean_direction + shape_weight * shape_direction,
                )
        decomposition_candidates[variant_name] = variant_decomposition

        variant_uncertainty = {}
        for cap in (0.25, 0.30, 0.35):
            for floor in (0.0, 0.05, 0.10):
                for low, high in ((0.75, 2.0), (1.0, 2.5), (1.25, 3.5), (1.5, 4.5), (2.0, 6.0)):
                    confidence = np.clip((high - variant_std) / (high - low), 0.0, 1.0)
                    row_weights = floor + (cap - floor) * confidence
                    key = f"row_cap_{cap:.2f}_floor_{floor:.2f}_lo_{low:.2f}_hi_{high:.2f}"
                    metrics = summarize(records, anchor + row_weights * variant_direction)
                    metrics.update(
                        {
                            "weight_mean": float(np.mean(row_weights)),
                            "weight_p10": float(np.quantile(row_weights, 0.10)),
                            "weight_p90": float(np.quantile(row_weights, 0.90)),
                        }
                    )
                    variant_uncertainty[key] = metrics

                    well_weights = []
                    for record in records:
                        well_std = float(np.mean(record["variant_stds"][variant_name]))
                        well_confidence = float(np.clip((high - well_std) / (high - low), 0.0, 1.0))
                        well_weights.append(
                            np.full(
                                len(record["truth"]),
                                floor + (cap - floor) * well_confidence,
                                dtype=float,
                            )
                        )
                    well_weights_array = np.concatenate(well_weights)
                    well_key = f"well_cap_{cap:.2f}_floor_{floor:.2f}_lo_{low:.2f}_hi_{high:.2f}"
                    well_metrics = summarize(
                        records, anchor + well_weights_array * variant_direction
                    )
                    well_metrics.update(
                        {
                            "weight_mean": float(np.mean(well_weights_array)),
                            "weight_p10": float(np.quantile(well_weights_array, 0.10)),
                            "weight_p90": float(np.quantile(well_weights_array, 0.90)),
                        }
                    )
                    variant_uncertainty[well_key] = well_metrics

        # Isolate redistribution from scalar dose: every candidate below keeps
        # the mean HMM weight exactly at the scored 0.25 reference.
        for slope in (0.02, 0.04, 0.06, 0.08, 0.10):
            centered = 0.25 + slope * (float(np.median(variant_std)) - variant_std)
            for _ in range(8):
                centered = np.clip(centered + (0.25 - float(np.mean(centered))), 0.10, 0.35)
            key = f"row_centered_mean025_slope_{slope:.2f}"
            metrics = summarize(records, anchor + centered * variant_direction)
            metrics.update(
                {
                    "weight_mean": float(np.mean(centered)),
                    "weight_p10": float(np.quantile(centered, 0.10)),
                    "weight_p90": float(np.quantile(centered, 0.90)),
                }
            )
            variant_uncertainty[key] = metrics

            centered_by_well = []
            for record in records:
                record_std = record["variant_stds"][variant_name]
                record_weight = 0.25 + slope * (
                    float(np.median(record_std)) - record_std
                )
                for _ in range(8):
                    record_weight = np.clip(
                        record_weight + (0.25 - float(np.mean(record_weight))),
                        0.10,
                        0.35,
                    )
                centered_by_well.append(record_weight)
            centered_well = np.concatenate(centered_by_well)
            well_key = f"within_well_centered_mean025_slope_{slope:.2f}"
            well_metrics = summarize(
                records, anchor + centered_well * variant_direction
            )
            well_metrics.update(
                {
                    "weight_mean": float(np.mean(centered_well)),
                    "weight_p10": float(np.quantile(centered_well, 0.10)),
                    "weight_p90": float(np.quantile(centered_well, 0.90)),
                }
            )
            variant_uncertainty[well_key] = well_metrics
        uncertainty_candidates[variant_name] = variant_uncertainty

    bins = []
    edges = np.quantile(std, [0.0, 0.25, 0.50, 0.75, 1.0])
    for index in range(4):
        mask = (std >= edges[index]) & (std <= edges[index + 1] if index == 3 else std < edges[index + 1])
        base_sse = float(np.sum(np.square(truth[mask] - anchor[mask])))
        hmm_sse = float(np.sum(np.square(truth[mask] - (anchor[mask] + 0.25 * direction[mask]))))
        weight_rmse = {
            f"{weight:.3f}": rmse(truth[mask], anchor[mask] + weight * direction[mask])
            for weight in WEIGHTS
        }
        best_bin_weight = min(weight_rmse, key=weight_rmse.get)
        bins.append(
            {
                "bin": index,
                "rows": int(mask.sum()),
                "std_min": float(edges[index]),
                "std_max": float(edges[index + 1]),
                "sse_gain_at_025": base_sse - hmm_sse,
                "rmse_anchor": rmse(truth[mask], anchor[mask]),
                "rmse_hmm025": rmse(truth[mask], anchor[mask] + 0.25 * direction[mask]),
                "best_weight": float(best_bin_weight),
                "best_rmse": float(weight_rmse[best_bin_weight]),
                "weight_rmse": weight_rmse,
            }
        )

    well_calibration = []
    for record in records:
        weight_rmse = {
            f"{weight:.3f}": rmse(
                record["truth"], record["anchor"] + weight * (record["hmm"] - record["anchor"])
            )
            for weight in WEIGHTS
        }
        best_well_weight = min(weight_rmse, key=weight_rmse.get)
        well_calibration.append(
            {
                "well": record["well"],
                "rows": int(len(record["truth"])),
                "std_mean": float(np.mean(record["std"])),
                "std_p90": float(np.quantile(record["std"], 0.90)),
                "best_weight": float(best_well_weight),
                "anchor_rmse": float(weight_rmse["0.000"]),
                "best_rmse": float(weight_rmse[best_well_weight]),
            }
        )

    best_constant = min(
        (name for name in candidates if name.startswith("constant_")),
        key=lambda name: candidates[name]["row_rmse"],
    )
    base_rmse = candidates["constant_0.000"]["row_rmse"]
    gated_rmse = candidates["std_gated_025"]["row_rmse"]
    result = {
        "wells": len(records),
        "rows": int(len(truth)),
        "runtime_sec": float(time.time() - started),
        "best_constant": best_constant,
        "best_constant_gain": base_rmse - candidates[best_constant]["row_rmse"],
        "std_gate_gain_vs_base": base_rmse - gated_rmse,
        "std_gate_gain_vs_constant025": candidates["constant_0.250"]["row_rmse"] - gated_rmse,
        "std_weight_mean": float(row_weight.mean()),
        "std_weight_p10": float(np.quantile(row_weight, 0.10)),
        "std_weight_p90": float(np.quantile(row_weight, 0.90)),
        "candidates": candidates,
        "emission_candidates": emission_candidates,
        "decomposition_candidates": decomposition_candidates,
        "uncertainty_candidates": uncertainty_candidates,
        "std_bins": bins,
        "well_calibration": well_calibration,
        "well_runtime_max": float(max(record["runtime_sec"] for record in records)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
