"""Geo-datum audit for contact-surface TVT reconstruction.

This diagnostic tests the geological hypothesis directly: withheld train wells
should be predictable as `contact_surface - Z + per-well offset` when the
contact surface is reconstructed from other train wells. It never uses the
withheld well's own formation columns for prediction.
"""
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
DEFAULT_CONTACTS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


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


def load_contact_pool(data_dir: Path, exclude: str, stride: int, contacts: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    usecols = ["X", "Y", *contacts]
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


def reconstruct_tvt(
    wrapper,
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    row_idx: np.ndarray,
    contact_pred: np.ndarray,
    plane: np.ndarray,
    knn: np.ndarray,
    surface_by_contact: dict[str, np.ndarray],
    surface_meta: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    surface_base = contact_pred - hw["Z"].to_numpy(dtype=float)
    known = hw["TVT_input"].notna().to_numpy() & np.isfinite(surface_base)
    if int(known.sum()) < 8:
        raise RuntimeError("Not enough known TVT_input rows")
    residual = hw.loc[known, "TVT_input"].to_numpy(dtype=float) - surface_base[known]
    offset, offset_meta = wrapper.robust_tail_offset(residual, tail_max=80)
    pred = surface_base + offset
    row_idx = np.asarray(row_idx, dtype=int)

    known_fit = float(np.sqrt(np.mean((residual - offset) ** 2)))
    method_gap = float(np.mean(np.abs(knn[row_idx] - plane[row_idx])))
    order_fraction = wrapper.contact_order_fraction(surface_by_contact, row_idx)
    roughness = wrapper.surface_roughness(hw, contact_pred, row_idx)
    gr = wrapper.gr_typewell_path_score(hw, tw, pred)
    confidence = 1.0 / (
        1.0
        + offset_meta["tail_fit_rmse"] / 20.0
        + offset_meta["tail_residual_std"] / 25.0
        + method_gap / 25.0
        + known_fit / 30.0
        + roughness / 0.04
        + max(0.0, 1.0 - order_fraction) * 2.5
    )
    confidence *= 0.50 + 0.50 * gr["gr_confidence"]
    meta = {
        "offset": float(offset),
        "n_known": float(known.sum()),
        **offset_meta,
        "known_fit_rmse": known_fit,
        "method_disagreement_mean": method_gap,
        "contact_order_fraction": order_fraction,
        "surface_roughness": roughness,
        "geo_confidence": float(np.clip(confidence, 0.03, 1.0)),
        "gr_loss": float(gr["gr_loss"]),
        "gr_confidence": float(gr["gr_confidence"]),
        "gr_sigma": float(gr["gr_sigma"]),
        "gr_r2": float(gr["gr_r2"]),
        **{str(k): float(v) for k, v in surface_meta.items() if isinstance(v, (int, float, np.floating))},
    }
    return pred[row_idx], meta


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in details.groupby("variant", sort=True):
        rows.append({
            "variant": variant,
            "n_masks": int(len(g)),
            "n_wells": int(g["well"].nunique()),
            "n_rows": int(g["n_eval"].sum()),
            "row_rmse": float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n_eval"]))),
            "well_rmse_mean": float(g["rmse"].mean()),
            "well_rmse_median": float(g["rmse"].median()),
            "well_rmse_p75": float(g["rmse"].quantile(0.75)),
            "well_rmse_max": float(g["rmse"].max()),
            "mean_confidence": float(g["confidence"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=100)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--contacts", default=",".join(DEFAULT_CONTACTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/geo_datum_audit_summary.csv"))
    parser.add_argument("--detail-output", type=Path, default=Path("docs/geo_datum_audit_details.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper = load_module("rogii_pf_artifact_blend", WRAPPER)
    pf_eval = load_module("evaluate_pf_variants", PF_EVAL)
    contacts = tuple(x.strip() for x in args.contacts.split(",") if x.strip())
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf_eval.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    rows: list[dict[str, object]] = []

    print(
        f"Geo datum audit wells={len(wells)} selection={args.selection} "
        f"known_fracs={known_fracs} contacts={contacts}",
        flush=True,
    )
    for wi, wid in enumerate(wells, 1):
        hw_raw, tw = pf_eval.load_well(args.data_dir, wid, "train")
        try:
            pool = load_contact_pool(args.data_dir, wid, args.stride, contacts)
        except RuntimeError as exc:
            print(f"[skip] {wid}: {exc}", flush=True)
            continue
        for known_frac in known_fracs:
            try:
                hw, eval_mask = make_hidden_mask(hw_raw, known_frac, args.min_known, args.min_eval)
            except ValueError as exc:
                print(f"[skip] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                continue
            row_idx = np.where(eval_mask)[0]
            y_true = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
            print(f"[{wi:03d}/{len(wells):03d}] {wid} frac={known_frac:.2f} n_eval={len(row_idx)}", flush=True)

            surface_cache: dict[str, dict[str, object]] = {}
            surface_by_contact: dict[str, np.ndarray] = {}
            for contact in contacts:
                try:
                    geo, plane, knn, surf_meta = wrapper.predict_contact_surface_geo(pool, hw, contact)
                except RuntimeError as exc:
                    print(f"[skip-surface] {wid} {contact}: {exc}", flush=True)
                    continue
                surface_cache[contact] = {"geo": geo, "plane": plane, "knn": knn, "meta": surf_meta}
                surface_by_contact[contact] = geo

            scored = []
            for contact, surfaces in surface_cache.items():
                try:
                    pred, meta = reconstruct_tvt(
                        wrapper,
                        hw,
                        tw,
                        row_idx,
                        np.asarray(surfaces["geo"], dtype=float),
                        np.asarray(surfaces["plane"], dtype=float),
                        np.asarray(surfaces["knn"], dtype=float),
                        surface_by_contact,
                        surfaces["meta"],
                    )
                except RuntimeError as exc:
                    print(f"[skip-contact] {wid} {contact}: {exc}", flush=True)
                    continue
                score = float(meta["geo_confidence"])
                scored.append((score, contact, pred, meta))
                rows.append({
                    "well": wid,
                    "known_frac": float(known_frac),
                    "variant": f"contact_{contact}",
                    "contact": contact,
                    "rmse": rmse(y_true, pred),
                    "bias": float(np.mean(pred - y_true)),
                    "n_eval": int(len(row_idx)),
                    "confidence": score,
                    **meta,
                })

            if scored:
                score, contact, pred, meta = max(scored, key=lambda item: item[0])
                rows.append({
                    "well": wid,
                    "known_frac": float(known_frac),
                    "variant": "geo_selected",
                    "contact": contact,
                    "rmse": rmse(y_true, pred),
                    "bias": float(np.mean(pred - y_true)),
                    "n_eval": int(len(row_idx)),
                    "confidence": float(score),
                    **meta,
                })

    if not rows:
        raise RuntimeError("No geo-datum evaluations were produced")

    details = pd.DataFrame(rows)
    summary = summarize(details)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    details.to_csv(args.detail_output, index=False)
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved {args.detail_output}")
    print(f"Saved {args.summary_output}")


if __name__ == "__main__":
    main()
