"""Pseudo-hidden audit for the contact-shape residual basis probe."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "kaggle" / "rogii-pf-artifact-blend" / "rogii-pf-artifact-blend.py"
PF_EVAL = ROOT / "scripts" / "diagnostics" / "evaluate_pf_variants.py"
CONTACTS = ("EGFDL", "EGFDU")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred.astype(float) - y_true.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def make_hidden_mask(hw: pd.DataFrame, known_frac: float, min_known: int, min_eval: int) -> tuple[pd.DataFrame, np.ndarray]:
    truth_mask = hw["TVT"].notna().to_numpy()
    truth_idx = np.where(truth_mask)[0]
    if len(truth_idx) < min_known + min_eval:
        raise ValueError("Not enough TVT truth rows for requested mask")
    first = int(truth_idx[0])
    last = int(truth_idx[-1])
    cut = first + int(round((last - first + 1) * float(known_frac)))
    cut = int(np.clip(cut, first + min_known - 1, last - min_eval))

    out = hw.copy()
    out["TVT_input"] = out["TVT"].where(np.arange(len(out)) <= cut, np.nan)
    eval_mask = (np.arange(len(out)) > cut) & truth_mask
    return out, eval_mask


def load_contact_pool(data_dir: Path, exclude: str, stride: int) -> pd.DataFrame:
    rows = []
    usecols = ["X", "Y", *CONTACTS]
    for path in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
        wid = path.name.split("__")[0]
        if wid == exclude:
            continue
        try:
            hw = pd.read_csv(path, usecols=usecols)
        except ValueError:
            continue
        slim = hw.iloc[::stride].dropna(subset=usecols).copy()
        if not slim.empty:
            rows.append(slim)
    if not rows:
        raise RuntimeError("No contact pool rows loaded")
    return pd.concat(rows, ignore_index=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in details.groupby("variant", sort=True):
        row_rmse = float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n_eval"])))
        rows.append({
            "variant": variant,
            "n_masks": int(len(g)),
            "n_wells": int(g["well"].nunique()),
            "n_rows": int(g["n_eval"].sum()),
            "row_rmse": row_rmse,
            "well_rmse_mean": float(g["rmse"].mean()),
            "well_rmse_median": float(g["rmse"].median()),
            "well_rmse_p75": float(g["rmse"].quantile(0.75)),
            "well_rmse_max": float(g["rmse"].max()),
            "mean_selected_score": float(g["selected_score"].mean()),
            "mean_basis_abs": float(g["basis_mean_abs"].mean()),
            "mean_confidence": float(g["confidence"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=24)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--particles", type=int, default=160)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--basis-value", type=float, default=0.25)
    parser.add_argument("--max-abs", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/contact_shape_basis_summary.csv"))
    parser.add_argument("--detail-output", type=Path, default=Path("docs/contact_shape_basis_details.csv"))
    parser.add_argument("--fail-on-deterioration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper = load_module("rogii_pf_artifact_blend", WRAPPER)
    pf_eval = load_module("evaluate_pf_variants", PF_EVAL)
    cfg = pf_eval.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf_eval.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    rows: list[dict[str, object]] = []

    print(
        f"Contact-shape audit wells={len(wells)} selection={args.selection} "
        f"known_fracs={known_fracs} seeds={args.seeds} particles={args.particles}",
        flush=True,
    )
    for wi, wid in enumerate(wells, 1):
        hw_raw, tw = pf_eval.load_well(args.data_dir, wid, "train")
        try:
            pool = load_contact_pool(args.data_dir, wid, args.stride)
        except RuntimeError as exc:
            print(f"[skip] {wid}: {exc}", flush=True)
            continue
        for known_frac in known_fracs:
            try:
                hw, eval_mask = make_hidden_mask(hw_raw, known_frac, args.min_known, args.min_eval)
            except ValueError as exc:
                print(f"[skip] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                continue
            print(f"[{wi:03d}/{len(wells):03d}] {wid} frac={known_frac:.2f} n_eval={eval_mask.sum()}", flush=True)
            preds, _ = pf_eval.predict_variants(hw, tw, ["grid_s3_b0_h0p17"], cfg)
            base_full = preds["grid_s3_b0_h0p17"]
            row_idx = np.where(eval_mask)[0]
            base_eval = base_full[row_idx]
            y_true = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)

            candidates = []
            for contact in CONTACTS:
                if contact not in pool.columns or contact not in hw.columns:
                    continue
                try:
                    knn = wrapper.predict_contact_surface_knn(pool, hw, contact)
                    plane = wrapper.predict_contact_surface_plane(pool, hw, contact)
                    basis, _, meta = wrapper.contact_basis_for_rows(
                        hw,
                        row_idx,
                        base_eval,
                        knn,
                        plane,
                        args.max_abs,
                    )
                except RuntimeError as exc:
                    print(f"[skip-contact] {wid} frac={known_frac:.2f} {contact}: {exc}", flush=True)
                    continue
                score = float(meta["basis_mean_abs"] * meta["confidence"])
                candidates.append({
                    "well": wid,
                    "contact": contact,
                    "basis": basis,
                    "score": score,
                    "meta": meta,
                })
            if not candidates:
                continue
            selected = wrapper.select_contact_basis_candidate(candidates)
            basis = np.asarray(selected["basis"], dtype=float)
            meta = selected["meta"]
            variants = {
                "base_grid_s3_b0_h0p17": base_eval,
                "contact_shape_plus0p25": base_eval + args.basis_value * basis,
                "contact_shape_minus0p25": base_eval - args.basis_value * basis,
            }
            for variant, pred in variants.items():
                rows.append({
                    "well": wid,
                    "known_frac": float(known_frac),
                    "variant": variant,
                    "rmse": rmse(y_true, pred),
                    "bias": float(np.mean(pred - y_true)),
                    "n_eval": int(eval_mask.sum()),
                    "selected_contact": str(selected["contact"]),
                    "selected_score": float(selected["score"]),
                    **{str(k): float(v) for k, v in meta.items()},
                })

    if not rows:
        raise RuntimeError("No contact-shape evaluations were produced")

    details = pd.DataFrame(rows)
    summary = summarize(details)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    details.to_csv(args.detail_output, index=False)

    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    indexed = summary.set_index("variant")
    base = indexed.loc["base_grid_s3_b0_h0p17"]
    plus = indexed.loc["contact_shape_plus0p25"]
    minus = indexed.loc["contact_shape_minus0p25"]
    row_improves = min(float(plus["row_rmse"]), float(minus["row_rmse"])) < float(base["row_rmse"])
    well_improves = min(float(plus["well_rmse_mean"]), float(minus["well_rmse_mean"])) < float(base["well_rmse_mean"])
    verdict = "PASS" if (row_improves or well_improves) else "STOP"
    print(f"\nGate verdict: {verdict}")
    print(f"Saved {args.detail_output}")
    print(f"Saved {args.summary_output}")
    if args.fail_on_deterioration and verdict == "STOP":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
