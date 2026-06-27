"""Train and evaluate a prefix-only pseudo-hidden regime selector.

This is a local gate for the Kaggle rerun setting. It creates hidden tails
inside train wells, scores a small candidate pool, and tests whether a simple
well-level selector can beat fixed PF baselines out of fold.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier, export_text


def load_pf_module():
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
    span = last - first + 1
    cut = first + int(round(span * known_frac))
    cut = int(np.clip(cut, first + min_known - 1, last - min_eval))

    out = hw.copy()
    out["TVT_input"] = out["TVT"].where(np.arange(len(out)) <= cut, np.nan)
    eval_mask = (np.arange(len(out)) > cut) & truth_mask
    if int(eval_mask.sum()) < min_eval:
        raise ValueError("too few eval rows")
    return out, eval_mask


def finite_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    xv = x[mask].astype(float)
    yv = y[mask].astype(float)
    denom = float(np.var(xv))
    if denom <= 1e-9:
        return 0.0
    return float(np.cov(xv, yv, bias=True)[0, 1] / denom)


def prefix_features(hw: pd.DataFrame, eval_mask: np.ndarray, meta: dict[str, float]) -> dict[str, float]:
    known = hw["TVT_input"].notna().to_numpy()
    known_df = hw.loc[known].copy()
    eval_df = hw.loc[eval_mask].copy()
    tail = known_df.tail(min(80, max(12, len(known_df) // 3)))
    gr_tail = tail["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)
    tvt_tail = tail["TVT_input"].to_numpy(dtype=float)
    md_tail = tail["MD"].to_numpy(dtype=float)
    z_tail = tail["Z"].to_numpy(dtype=float)
    full_gr_known = known_df["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)

    if len(tvt_tail) >= 3:
        tvt_step = np.diff(tvt_tail)
        tvt_curv = np.diff(tvt_tail, n=2)
    else:
        tvt_step = np.array([0.0])
        tvt_curv = np.array([0.0])

    out = {
        "n": float(len(hw)),
        "n_known": float(known.sum()),
        "n_eval": float(eval_mask.sum()),
        "known_frac": float(known.mean()),
        "eval_z_span": float(eval_df["Z"].max() - eval_df["Z"].min()) if len(eval_df) else 0.0,
        "eval_md_span": float(eval_df["MD"].max() - eval_df["MD"].min()) if len(eval_df) else 0.0,
        "eval_gr_nan": float(eval_df["GR"].isna().mean()) if len(eval_df) else 0.0,
        "prefix_gr_mean": float(np.nanmean(full_gr_known)) if len(full_gr_known) else 0.0,
        "prefix_gr_std": float(np.nanstd(full_gr_known)) if len(full_gr_known) else 0.0,
        "tail_gr_mean": float(np.nanmean(gr_tail)) if len(gr_tail) else 0.0,
        "tail_gr_std": float(np.nanstd(gr_tail)) if len(gr_tail) else 0.0,
        "tail_gr_slope_md": finite_slope(md_tail, gr_tail),
        "tail_tvt_slope_md": finite_slope(md_tail, tvt_tail),
        "tail_z_slope_md": finite_slope(md_tail, z_tail),
        "tail_tvt_step_mean": float(np.nanmean(tvt_step)),
        "tail_tvt_step_std": float(np.nanstd(tvt_step)),
        "tail_tvt_curv_abs_mean": float(np.nanmean(np.abs(tvt_curv))),
    }
    for key, value in meta.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[f"meta_{key}"] = float(value)
    return out


def summarize_scores(details: pd.DataFrame, candidate_cols: list[str], selected: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name in candidate_cols:
        rows.append(
            {
                "variant": name,
                "n_masks": int(len(details)),
                "n_wells": int(details["well"].nunique()),
                "row_rmse": rmse(
                    np.concatenate(details["y_true"].to_list()),
                    np.concatenate(details[f"pred_{name}"].to_list()),
                ),
                "well_rmse_mean": float(details[f"rmse_{name}"].mean()),
                "well_rmse_median": float(details[f"rmse_{name}"].median()),
                "well_rmse_p75": float(details[f"rmse_{name}"].quantile(0.75)),
                "well_rmse_max": float(details[f"rmse_{name}"].max()),
            }
        )
    if selected is not None:
        y_pred = []
        selected_rmse = []
        for idx, name in selected.items():
            y_pred.append(details.loc[idx, f"pred_{name}"])
            selected_rmse.append(float(details.loc[idx, f"rmse_{name}"]))
        rows.append(
            {
                "variant": "oof_prefix_selector",
                "n_masks": int(len(details)),
                "n_wells": int(details["well"].nunique()),
                "row_rmse": rmse(np.concatenate(details["y_true"].to_list()), np.concatenate(y_pred)),
                "well_rmse_mean": float(np.mean(selected_rmse)),
                "well_rmse_median": float(np.median(selected_rmse)),
                "well_rmse_p75": float(np.quantile(selected_rmse, 0.75)),
                "well_rmse_max": float(np.max(selected_rmse)),
            }
        )
    return pd.DataFrame(rows).sort_values(["well_rmse_mean", "row_rmse"])


def train_oof_selector(
    details: pd.DataFrame,
    candidate_cols: list[str],
    feature_cols: list[str],
    baseline: str,
    max_depth: int,
    min_leaf: int,
    min_improve: float,
    min_confidence: float,
) -> tuple[pd.Series, pd.Series, list[str]]:
    rmse_cols = [f"rmse_{name}" for name in candidate_cols]
    rmse_values = details[rmse_cols].to_numpy(float)
    best_idx = np.argmin(rmse_values, axis=1)
    best_names = np.asarray(candidate_cols, dtype=object)[best_idx]
    baseline_rmse = details[f"rmse_{baseline}"].to_numpy(float)
    best_rmse = rmse_values[np.arange(len(details)), best_idx]
    labels = np.where((baseline_rmse - best_rmse) >= min_improve, best_names, baseline)
    labels = pd.Series(labels, index=details.index, name="target_regime")

    x = details[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    groups = details["well"].astype(str).to_numpy()
    selected = pd.Series(baseline, index=details.index, name="selected_regime", dtype=object)
    fold_rules: list[str] = []

    splitter = LeaveOneGroupOut()
    for fold, (tr, va) in enumerate(splitter.split(x, labels, groups), 1):
        y_train = labels.iloc[tr]
        if y_train.nunique() <= 1:
            selected.iloc[va] = baseline
            fold_rules.append(f"fold={fold}: single-class fallback {baseline}")
            continue
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(x.iloc[tr], y_train)
        proba = clf.predict_proba(x.iloc[va])
        pred = clf.classes_[np.argmax(proba, axis=1)]
        conf = np.max(proba, axis=1)
        pred = np.where(conf >= min_confidence, pred, baseline)
        selected.iloc[va] = pred
        fold_rules.append(f"fold={fold} heldout={groups[va][0]}\n{export_text(clf, feature_names=feature_cols)}")
    return selected, labels, fold_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--particles", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument(
        "--variants",
        default="grid_s3_b0_h0p17,grid_s3_b0_h0p2,grid_s3_b0_h0p24,grid_s8_b0_h0p2,bin_lb_safe,anchor",
    )
    parser.add_argument("--baseline", default="grid_s3_b0_h0p17")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-leaf", type=int, default=4)
    parser.add_argument("--min-improve", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/hidden_regime_selector_smoke_summary.csv"))
    parser.add_argument("--details-output", type=Path, default=Path("docs/hidden_regime_selector_smoke_details.csv"))
    parser.add_argument("--rules-output", type=Path, default=Path("docs/hidden_regime_selector_rules.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pf = load_pf_module()
    cfg = pf.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if args.baseline not in variants:
        raise ValueError(f"--baseline must be one of --variants: {args.baseline}")
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    pf_variants = [v for v in variants if v != "anchor"]

    rows = []
    for wi, wid in enumerate(wells, 1):
        hw_raw, tw = pf.load_well(args.data_dir, wid, "train")
        for known_frac in known_fracs:
            try:
                hw, eval_mask = make_hidden_mask(hw_raw, known_frac, args.min_known, args.min_eval)
            except ValueError as exc:
                print(f"[skip] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                continue
            print(f"[{wi:03d}/{len(wells):03d}] {wid} frac={known_frac:.2f} eval={int(eval_mask.sum())}", flush=True)
            preds = {}
            meta = {}
            if pf_variants:
                preds, meta = pf.predict_variants(hw, tw, pf_variants, cfg)
            if "anchor" in variants:
                preds["anchor"] = hw["TVT_input"].ffill().bfill().to_numpy(float)

            y_true = hw.loc[eval_mask, "TVT"].to_numpy(float)
            row = {
                "well": wid,
                "known_frac_request": float(known_frac),
                "y_true": y_true,
            }
            row.update(prefix_features(hw, eval_mask, meta))
            for variant in variants:
                y_pred = preds[variant][eval_mask].astype(float)
                row[f"pred_{variant}"] = y_pred
                row[f"rmse_{variant}"] = rmse(y_true, y_pred)
                row[f"bias_{variant}"] = float(np.mean(y_pred - y_true))
                row[f"trend_delta_{variant}"] = float((y_pred[-1] - y_pred[0]) - (y_true[-1] - y_true[0]))
            rows.append(row)

    if not rows:
        raise RuntimeError("No pseudo-hidden masks were evaluated.")

    details = pd.DataFrame(rows)
    feature_cols = [
        c
        for c in details.columns
        if c
        not in {"well", "y_true"}
        and not c.startswith("pred_")
        and not c.startswith("rmse_")
        and not c.startswith("bias_")
        and not c.startswith("trend_delta_")
    ]
    selected, target, rules = train_oof_selector(
        details,
        variants,
        feature_cols,
        args.baseline,
        args.max_depth,
        args.min_leaf,
        args.min_improve,
        args.min_confidence,
    )
    details["selector_target"] = target
    details["selector_oof"] = selected
    summary = summarize_scores(details, variants, selected)

    serializable = details.drop(columns=["y_true", *[f"pred_{v}" for v in variants]]).copy()
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    serializable.to_csv(args.details_output, index=False)
    args.rules_output.write_text("\n\n".join(rules), encoding="utf-8")

    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    base_row = summary.loc[summary["variant"] == args.baseline].iloc[0]
    sel_row = summary.loc[summary["variant"] == "oof_prefix_selector"].iloc[0]
    print(
        "\nGate delta vs baseline: "
        f"well_mean={base_row['well_rmse_mean'] - sel_row['well_rmse_mean']:.4f} "
        f"row={base_row['row_rmse'] - sel_row['row_rmse']:.4f}",
        flush=True,
    )
    print(f"Saved {args.summary_output}")
    print(f"Saved {args.details_output}")
    print(f"Saved {args.rules_output}")


if __name__ == "__main__":
    main()
