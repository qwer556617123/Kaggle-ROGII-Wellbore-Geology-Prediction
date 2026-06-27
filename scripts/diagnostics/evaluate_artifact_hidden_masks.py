"""Evaluate PF/artifact blends on pseudo-hidden train-well tails.

This creates a temporary competition-like dataset whose test split contains
masked train wells. It then runs the current PF component and v10 artifact
component in inference mode, scores blend weights against the hidden truth, and
records whether any selector/blend is worth a Kaggle submission.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_pf_eval():
    path = Path("scripts/diagnostics/evaluate_pf_variants.py")
    spec = importlib.util.spec_from_file_location("pf_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true.astype(float) - y_pred.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def make_hidden_mask(
    hw: pd.DataFrame,
    known_frac: float,
    min_known: int,
    min_eval: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    truth_mask = hw["TVT"].notna().to_numpy()
    truth_idx = np.where(truth_mask)[0]
    if len(truth_idx) < min_known + min_eval:
        raise ValueError("not enough TVT truth rows")
    first = int(truth_idx[0])
    last = int(truth_idx[-1])
    cut = first + int(round((last - first + 1) * known_frac))
    cut = int(np.clip(cut, first + min_known - 1, last - min_eval))
    out = hw.copy()
    out["TVT_input"] = out["TVT"].where(np.arange(len(out)) <= cut, np.nan)
    eval_mask = (np.arange(len(out)) > cut) & truth_mask
    if int(eval_mask.sum()) < min_eval:
        raise ValueError("too few eval rows")
    return out, eval_mask


def prepare_dataset(
    data_dir: Path,
    out_dir: Path,
    wells: list[str],
    known_frac: float,
    min_known: int,
    min_eval: int,
    include_train: bool,
) -> pd.DataFrame:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "train").mkdir(parents=True)
    (out_dir / "test").mkdir(parents=True)

    heldout = set(wells)
    if include_train:
        for path in sorted((data_dir / "train").glob("*.csv")):
            wid = path.name.split("__")[0]
            if wid in heldout:
                continue
            shutil.copy2(path, out_dir / "train" / path.name)

    truth_rows = []
    sample_rows = []
    for wid in wells:
        hw = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
        tw_path = data_dir / "train" / f"{wid}__typewell.csv"
        hw_masked, eval_mask = make_hidden_mask(hw, known_frac, min_known, min_eval)
        hw_masked.to_csv(out_dir / "test" / f"{wid}__horizontal_well.csv", index=False)
        shutil.copy2(tw_path, out_dir / "test" / f"{wid}__typewell.csv")
        eval_idx = np.where(eval_mask)[0]
        for idx in eval_idx:
            row_id = f"{wid}_{int(idx)}"
            sample_rows.append({"id": row_id, "tvt": 0.0})
            truth_rows.append({"id": row_id, "well": wid, "row_idx": int(idx), "tvt": float(hw.loc[idx, "TVT"])})

    pd.DataFrame(sample_rows).to_csv(out_dir / "sample_submission.csv", index=False)
    return pd.DataFrame(truth_rows)


def run_component(script: Path, data_dir: Path, output_dir: Path, env: dict[str, str]) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cmd_env = dict(**env)
    cmd_env["ROGII_DATA_DIR"] = str(data_dir.resolve())
    cmd_env["ROGII_OUTPUT_DIR"] = str(output_dir.resolve())
    cmd_env["PYTHONIOENCODING"] = "utf-8"
    import os

    full_env = os.environ.copy()
    full_env.update(cmd_env)
    subprocess.run([sys.executable, str(script)], check=True, env=full_env, cwd=data_dir)
    out = output_dir / "submission.csv"
    if not out.exists():
        raise FileNotFoundError(out)
    return out


def summarize(details: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for name in candidates:
        rows.append(
            {
                "candidate": name,
                "n_rows": int(len(details)),
                "n_wells": int(details["well"].nunique()),
                "row_rmse": rmse(details["tvt"].to_numpy(float), details[name].to_numpy(float)),
                "well_rmse_mean": float(
                    details.groupby("well").apply(lambda g: rmse(g["tvt"].to_numpy(float), g[name].to_numpy(float))).mean()
                ),
                "well_rmse_max": float(
                    details.groupby("well").apply(lambda g: rmse(g["tvt"].to_numpy(float), g[name].to_numpy(float))).max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["well_rmse_mean", "row_rmse"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("tmp/rogii-v10-fresh-artifacts"))
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/artifact_hidden_masks"))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=3)
    parser.add_argument("--known-frac", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--pf-seeds", type=int, default=24)
    parser.add_argument("--pf-particles", type=int, default=120)
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("docs/artifact_hidden_mask_smoke_summary.csv"))
    parser.add_argument("--details-output", type=Path, default=Path("docs/artifact_hidden_mask_smoke_details.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pf_eval = load_pf_eval()
    wells = pf_eval.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    tmp_data = args.work_dir / f"data_frac{str(args.known_frac).replace('.', 'p')}_n{len(wells)}"
    truth = prepare_dataset(
        args.data_dir,
        tmp_data,
        wells,
        args.known_frac,
        args.min_known,
        args.min_eval,
        args.include_train,
    )
    print(f"Prepared {tmp_data} rows={len(truth)} wells={wells}", flush=True)

    root = Path.cwd()
    pf_csv = run_component(
        root / "kaggle/rogii-pf-artifact-blend/pf_component.py",
        tmp_data,
        args.work_dir / "pf",
        {
            "ROGII_VARIANT": "grid_s3_b0_h0p17",
            "ROGII_N_SEEDS": str(args.pf_seeds),
            "ROGII_N_PARTICLES": str(args.pf_particles),
            "ROGII_USE_VISIBLE_PHYSICAL": "0",
        },
    )
    artifact_csv = run_component(
        root / "kaggle/rogii-pf-artifact-blend/v10_artifact_component.py",
        tmp_data,
        args.work_dir / "artifact",
        {
            "ROGII_ARTIFACT_DIR": str(args.artifact_dir.resolve()),
            "ROGII_INFERENCE_ONLY": "1",
            "ROGII_SAVE_ARTIFACTS": "0",
            "ROGII_RUN_TABICL": "0",
            "ROGII_FORCE_CPU": "1",
            "ROGII_EXACT_OVERLAP": "0",
            "ROGII_NCPU": "4",
        },
    )

    pf = pd.read_csv(pf_csv).rename(columns={"tvt": "pf"})
    art = pd.read_csv(artifact_csv).rename(columns={"tvt": "artifact"})
    details = truth.merge(pf, on="id", how="left").merge(art, on="id", how="left")
    if details[["pf", "artifact"]].isna().any().any():
        missing = details[["pf", "artifact"]].isna().sum().to_dict()
        raise RuntimeError(f"missing predictions: {missing}")

    for weight in [0.70, 0.75, 0.775, 0.80, 0.825, 0.85, 0.90, 1.00]:
        name = f"blend_pf{int(round(weight * 100)):02d}"
        details[name] = weight * details["pf"] + (1.0 - weight) * details["artifact"]
    gap = details["pf"] - details["artifact"]
    details["selector_gap_hi_75"] = details["blend_pf80"]
    gap_by_well = details.assign(abs_gap=np.abs(gap)).groupby("well")["abs_gap"].mean()
    if len(gap_by_well):
        selected = str(gap_by_well.idxmax())
        mask = details["well"] == selected
        details.loc[mask, "selector_gap_hi_75"] = details.loc[mask, "blend_pf75"]
        selector_meta = {"gap_hi_well": selected, "gap_hi_mean_abs_gap": float(gap_by_well.loc[selected])}
    else:
        selector_meta = {}

    candidates = [c for c in details.columns if c.startswith("blend_") or c in {"pf", "artifact"} or c.startswith("selector_")]
    summary = summarize(details, candidates)
    summary["known_frac"] = float(args.known_frac)
    summary["well_limit"] = int(args.well_limit)
    summary["selector_meta"] = json.dumps(selector_meta)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    details.to_csv(args.details_output, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved {args.summary_output}")
    print(f"Saved {args.details_output}")


if __name__ == "__main__":
    main()
