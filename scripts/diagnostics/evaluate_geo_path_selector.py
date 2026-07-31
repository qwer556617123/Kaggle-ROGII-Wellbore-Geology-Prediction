"""Pseudo-hidden audit for geo-first datum/path selection.

This diagnostic treats TVT prediction as candidate path selection. It uses
train wells as pseudo-hidden test wells, with formation columns withheld from
the target well and contact surfaces reconstructed from other train wells.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
            "mean_selector_score": float(g["selector_score"].mean()),
            "mean_confidence": float(g["confidence"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def gate_summary(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    min_row_gain: float,
    max_well_deterioration: float,
    max_dominant_contact_fraction: float,
) -> dict[str, object]:
    indexed = summary.set_index("variant")
    base = indexed.loc["base_grid_s3_b0_h0p17"]
    selector = indexed.loc["geo_path_selector_v1"]
    conservative = indexed.loc["geo_path_hybrid_0p3"]
    best_name = (
        "geo_path_selector_v1"
        if float(selector["row_rmse"]) <= float(conservative["row_rmse"])
        else "geo_path_hybrid_0p3"
    )
    best = indexed.loc[best_name]
    row_gain = float(base["row_rmse"] - best["row_rmse"])
    well_deterioration = float(best["well_rmse_mean"] - base["well_rmse_mean"])
    selected = details[details["variant"] == best_name].copy()
    geo_contacts = selected[selected["selected_type"] == "geo_contact"]["selected_contact"].astype(str)
    contact_distribution = {
        str(k): float(v)
        for k, v in geo_contacts.value_counts(normalize=True).to_dict().items()
        if str(k)
    }
    dominant_contact_fraction = float(max(contact_distribution.values())) if contact_distribution else 0.0
    passed = (
        row_gain >= min_row_gain
        and well_deterioration <= max_well_deterioration
        and dominant_contact_fraction <= max_dominant_contact_fraction
    )
    return {
        "verdict": "PASS" if passed else "STOP",
        "best_variant": best_name,
        "row_gain": row_gain,
        "well_mean_deterioration": well_deterioration,
        "dominant_contact_fraction": dominant_contact_fraction,
        "contact_distribution": contact_distribution,
        "min_row_gain": float(min_row_gain),
        "max_well_deterioration": float(max_well_deterioration),
        "max_dominant_contact_fraction": float(max_dominant_contact_fraction),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=100)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--contacts", default=",".join(DEFAULT_CONTACTS))
    parser.add_argument("--max-abs-delta", type=float, default=80.0)
    parser.add_argument("--min-row-gain", type=float, default=0.30)
    parser.add_argument("--max-well-deterioration", type=float, default=0.0)
    parser.add_argument("--max-dominant-contact-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/geo_path_selector_summary.csv"))
    parser.add_argument("--detail-output", type=Path, default=Path("docs/geo_path_selector_details.csv"))
    parser.add_argument("--gate-output", type=Path, default=Path("docs/geo_path_selector_gate.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper = load_module("rogii_pf_artifact_blend", WRAPPER)
    pf_eval = load_module("evaluate_pf_variants", PF_EVAL)
    cfg = pf_eval.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    contacts = tuple(x.strip() for x in args.contacts.split(",") if x.strip())
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf_eval.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    rows: list[dict[str, object]] = []

    print(
        f"Geo path selector audit wells={len(wells)} selection={args.selection} "
        f"known_fracs={known_fracs} seeds={args.seeds} particles={args.particles}",
        flush=True,
    )
    for wi, wid in enumerate(wells, 1):
        hw_raw, tw = pf_eval.load_well(args.data_dir, wid, "train")
        try:
            pool = load_contact_pool(args.data_dir, wid, args.stride, contacts)
        except RuntimeError as exc:
            print(f"[skip] {wid}: {exc}", flush=True)
            continue
        surface_cache: dict[str, dict[str, object]] = {}
        for contact in contacts:
            if contact not in pool.columns:
                continue
            try:
                geo, plane, knn, surf_meta = wrapper.predict_contact_surface_geo(pool, hw_raw, contact)
            except (RuntimeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                print(f"[skip-surface] {wid} {contact}: {exc}", flush=True)
                continue
            surface_cache[contact] = {"geo": geo, "plane": plane, "knn": knn, "meta": surf_meta}
        if not surface_cache:
            print(f"[skip] {wid}: no contact surfaces", flush=True)
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
            try:
                candidates = wrapper.build_geo_path_candidates(
                    hw,
                    tw,
                    row_idx,
                    base_eval,
                    base_eval,
                    None,
                    pool,
                    contacts=contacts,
                    max_abs_delta=args.max_abs_delta,
                    precomputed_surfaces=surface_cache,
                )
                selected, _ = wrapper.select_geo_path_candidate(hw, tw, candidates, row_idx, base_eval)
            except (RuntimeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                print(f"[skip-selector] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                continue

            scored_geo = []
            scored_all = []
            for cand in candidates:
                score, meta = wrapper.geo_path_score(hw, tw, cand["full"], base_full, row_idx, cand.get("meta", {}))
                pred = np.asarray(cand["full"], dtype=float)[row_idx]
                scored_all.append((score, cand, pred, meta))
                if cand["candidate_type"] == "geo_contact":
                    scored_geo.append((score, cand, pred, meta))
            if not scored_geo:
                continue
            oracle_geo = min(scored_geo, key=lambda item: rmse(y_true, item[2]))
            gr_geo = min(scored_geo, key=lambda item: float(item[0]))
            selected_pred = np.asarray(selected["full"], dtype=float)[row_idx]
            conservative_pred = 0.7 * base_eval + 0.3 * selected_pred
            variants = [
                ("base_grid_s3_b0_h0p17", base_eval, {"candidate_type": "base_blend", "contact": "", "score": 0.0, "confidence": 1.0}),
                ("best_single_geo_oracle", oracle_geo[2], {"candidate_type": "geo_contact", "contact": oracle_geo[1].get("contact", ""), **oracle_geo[3]}),
                ("gr_selected_geo", gr_geo[2], {"candidate_type": "geo_contact", "contact": gr_geo[1].get("contact", ""), **gr_geo[3]}),
                ("geo_path_selector_v1", selected_pred, {"candidate_type": selected["candidate_type"], "contact": selected.get("contact", ""), **selected["score_meta"]}),
                ("geo_path_hybrid_0p3", conservative_pred, {"candidate_type": selected["candidate_type"], "contact": selected.get("contact", ""), **selected["score_meta"]}),
            ]
            for variant, pred, meta in variants:
                rows.append({
                    "well": wid,
                    "known_frac": float(known_frac),
                    "variant": variant,
                    "rmse": rmse(y_true, pred),
                    "bias": float(np.mean(pred - y_true)),
                    "n_eval": int(eval_mask.sum()),
                    "selected_type": str(meta.get("candidate_type", "")),
                    "selected_contact": str(meta.get("contact", "")),
                    "selector_score": float(meta.get("selector_score", meta.get("score", 0.0))),
                    "confidence": float(meta.get("confidence", 1.0)),
                })

    if not rows:
        raise RuntimeError("No geo path selector evaluations were produced")
    details = pd.DataFrame(rows)
    summary = summarize(details)
    gate = gate_summary(
        summary,
        details,
        args.min_row_gain,
        args.max_well_deterioration,
        args.max_dominant_contact_fraction,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    details.to_csv(args.detail_output, index=False)
    args.gate_output.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGate")
    print(json.dumps(gate, indent=2))
    print(f"Saved {args.detail_output}")
    print(f"Saved {args.summary_output}")
    print(f"Saved {args.gate_output}")


if __name__ == "__main__":
    main()
