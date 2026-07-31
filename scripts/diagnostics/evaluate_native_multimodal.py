"""Evaluate the multimodal tracker on the competition's native train masks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multimodal_geosteering import PosteriorConfig, predict_multimodal


ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(error * error)))


def load_pf_module():
    path = Path("scripts/diagnostics/evaluate_pf_variants.py")
    spec = importlib.util.spec_from_file_location("native_pf_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_wells(data_dir: Path, limit: int, seed: int) -> list[str]:
    wells = sorted(p.name.split("__")[0] for p in (data_dir / "train").glob("*__horizontal_well.csv"))
    rng = np.random.default_rng(seed)
    if 0 < limit < len(wells):
        wells = sorted(rng.choice(wells, size=limit, replace=False).tolist())
    return wells


def load_public_profiles(data_dir: Path) -> list[dict[str, float]]:
    rows = []
    for path in sorted((data_dir / "test").glob("*__horizontal_well.csv")):
        hw = pd.read_csv(path, usecols=["Z", "GR", "TVT_input"])
        known = hw["TVT_input"].notna().to_numpy()
        eval_mask = ~known
        if not eval_mask.any():
            continue
        rows.append(
            {
                "well": path.name[:8],
                "known_frac": float(known.mean()),
                "eval_rows": float(eval_mask.sum()),
                "gr_missing": float(hw.loc[eval_mask, "GR"].isna().mean()),
                "z_span": float(hw.loc[eval_mask, "Z"].max() - hw.loc[eval_mask, "Z"].min()),
            }
        )
    return rows


def nearest_profile(profile: dict[str, float], public: list[dict[str, float]]) -> str:
    if not public:
        return "none"
    scales = {"known_frac": 0.05, "eval_rows": 2000.0, "gr_missing": 0.15, "z_span": 60.0}
    best = min(
        public,
        key=lambda ref: sum(abs(profile[key] - ref[key]) / scales[key] for key in scales),
    )
    return str(best["well"])


def summary(name: str, records: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    y_true = np.concatenate([record["truth"] for record in records])
    y_pred = np.concatenate([record[prediction_key] for record in records])
    well_rmse = np.asarray([rmse(record["truth"], record[prediction_key]) for record in records])
    well_sse = np.asarray(
        [float(np.sum((record["truth"] - record[prediction_key]) ** 2)) for record in records]
    )
    count = max(1, int(np.ceil(0.10 * len(well_sse))))
    worst_share = float(np.sort(well_sse)[-count:].sum() / max(well_sse.sum(), 1e-12))
    return {
        "variant": name,
        "wells": int(len(records)),
        "rows": int(len(y_true)),
        "row_rmse": rmse(y_true, y_pred),
        "well_mean": float(well_rmse.mean()),
        "well_median": float(np.median(well_rmse)),
        "well_p90": float(np.quantile(well_rmse, 0.90)),
        "well_max": float(well_rmse.max()),
        "worst10_sse_share": worst_share,
    }


def confidence_tier(record: dict[str, Any]) -> str:
    meta = record["metadata"]
    coverage = float(meta.get("gr_coverage", 0.0))
    calibration = float(meta.get("calibration_rmse", float("inf")))
    replay = float(meta.get("prefix_replay_rmse", float("inf")))
    confidence = float(meta.get("confidence", 0.0))
    score_gain = float(meta.get("map_score_gain", 0.0))
    if coverage < 0.20 or calibration > 35.0 or replay > 10.0 or score_gain < 0.03:
        return "low"
    if confidence >= 0.35:
        return "high"
    if confidence >= 0.16:
        return "medium"
    return "low"


def profile_bin(record: dict[str, Any]) -> str:
    known = record["known_frac"]
    missing = record["gr_missing"]
    known_bin = "k0" if known < 0.23 else "k1" if known < 0.30 else "k2"
    gr_bin = "g0" if missing < 0.20 else "g1" if missing < 0.45 else "g2"
    return f"{known_bin}_{gr_bin}_{confidence_tier(record)}"


def best_alpha(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    truth = np.concatenate([record["truth"] for record in records])
    anchor = np.concatenate([record["anchor"] for record in records])
    posterior = np.concatenate([record["posterior"] for record in records])
    return float(min(ALPHA_GRID, key=lambda value: rmse(truth, anchor + value * (posterior - anchor))))


def fit_policy(records: list[dict[str, Any]], min_bin: int) -> dict[str, Any]:
    global_alpha = best_alpha(records)
    by_tier = {}
    for tier in ("low", "medium", "high"):
        subset = [record for record in records if confidence_tier(record) == tier]
        by_tier[tier] = best_alpha(subset) if len(subset) >= min_bin else global_alpha
    by_bin = {}
    keys = sorted(set(profile_bin(record) for record in records))
    for key in keys:
        subset = [record for record in records if profile_bin(record) == key]
        if len(subset) >= min_bin:
            by_bin[key] = best_alpha(subset)
    return {"global": global_alpha, "by_tier": by_tier, "by_bin": by_bin, "min_bin": int(min_bin)}


def policy_alpha(record: dict[str, Any], policy: dict[str, Any]) -> float:
    if confidence_tier(record) == "low":
        return 0.0
    key = profile_bin(record)
    if key in policy["by_bin"]:
        return float(policy["by_bin"][key])
    return float(policy["by_tier"].get(confidence_tier(record), policy["global"]))


def apply_oof_policy(records: list[dict[str, Any]], min_bin: int) -> tuple[dict[str, Any], list[float]]:
    alphas = [0.0] * len(records)
    for fold in range(5):
        train_records = [record for record in records if record["fold"] != fold]
        policy = fit_policy(train_records, min_bin)
        for index, record in enumerate(records):
            if record["fold"] == fold:
                alphas[index] = policy_alpha(record, policy)
    for record, alpha in zip(records, alphas):
        record["hybrid_oof"] = record["anchor"] + alpha * (record["posterior"] - record["anchor"])
        record["alpha_oof"] = alpha
    return fit_policy(records, min_bin), alphas


def load_anchor_csv(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(frame.columns):
        raise ValueError("Anchor CSV must contain id,tvt")
    return dict(zip(frame["id"].astype(str), frame["tvt"].astype(float)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--well-limit", type=int, default=773)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--anchor", choices=["prior", "last", "pf", "csv"], default="prior")
    parser.add_argument("--anchor-csv", type=Path)
    parser.add_argument("--pf-seeds", type=int, default=16)
    parser.add_argument("--pf-particles", type=int, default=100)
    parser.add_argument("--min-policy-bin", type=int, default=30)
    parser.add_argument("--details-output", type=Path, default=Path("docs/native_multimodal_details.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("docs/native_multimodal_summary.csv"))
    parser.add_argument("--gate-output", type=Path, default=Path("docs/native_multimodal_gate.json"))
    parser.add_argument("--anchor-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PosteriorConfig(temperature=args.temperature)
    wells = select_wells(args.data_dir, args.well_limit, args.seed)
    public_profiles = load_public_profiles(args.data_dir)
    anchor_map = load_anchor_csv(args.anchor_csv)
    pf = load_pf_module() if args.anchor == "pf" else None
    pf_config = pf.PfConfig(n_particles=args.pf_particles, n_seeds=args.pf_seeds) if pf else None
    records: list[dict[str, Any]] = []

    for index, well in enumerate(wells, 1):
        hw = pd.read_csv(args.data_dir / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(args.data_dir / "train" / f"{well}__typewell.csv")
        eval_mask = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
        if int(eval_mask.sum()) < 40 or int(hw["TVT_input"].notna().sum()) < 40:
            continue
        if args.anchor == "last":
            value = float(hw["TVT_input"].dropna().iloc[-1])
            anchor_full = np.full(len(hw), value, dtype=float)
        elif args.anchor == "pf":
            predictions, _ = pf.predict_variants(hw, tw, ["grid_s3_b0_h0p17"], pf_config)
            anchor_full = predictions["grid_s3_b0_h0p17"]
        elif args.anchor == "csv":
            anchor_full = np.full(len(hw), np.nan, dtype=float)
            known_mask = hw["TVT_input"].notna().to_numpy()
            anchor_full[known_mask] = hw.loc[known_mask, "TVT_input"].to_numpy(dtype=float)
            for row_index in np.flatnonzero(eval_mask):
                row_id = f"{well}_{row_index}"
                if row_id in anchor_map:
                    anchor_full[row_index] = anchor_map[row_id]
            if not np.isfinite(anchor_full[eval_mask]).all():
                print(f"[skip] {well}: anchor CSV is incomplete", flush=True)
                continue
        else:
            anchor_full = None

        try:
            result = predict_multimodal(
                hw,
                tw,
                anchor=anchor_full,
                config=config,
                alpha=1.0,
                search_from_anchor=args.anchor in {"pf", "csv"},
            )
        except Exception as exc:
            print(f"[skip] {well}: {exc}", flush=True)
            continue
        if anchor_full is None:
            anchor_full = result.prior

        truth = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        anchor = np.asarray(anchor_full, dtype=float)[eval_mask]
        posterior = result.posterior[eval_mask]
        map_path = result.map_path[eval_mask]
        known_frac = float(hw["TVT_input"].notna().mean())
        gr_missing = float(hw.loc[eval_mask, "GR"].isna().mean())
        profile = {
            "known_frac": known_frac,
            "eval_rows": float(eval_mask.sum()),
            "gr_missing": gr_missing,
            "z_span": float(hw.loc[eval_mask, "Z"].max() - hw.loc[eval_mask, "Z"].min()),
        }
        record = {
            "well": well,
            "row_ids": [f"{well}_{row_index}" for row_index in np.flatnonzero(eval_mask)],
            "fold": int(hashlib.sha256(well.encode()).digest()[0] % 5),
            "truth": truth,
            "anchor": anchor,
            "prior": result.prior[eval_mask],
            "map_path": map_path,
            "posterior": posterior,
            "metadata": result.metadata,
            **profile,
            "nearest_public_profile": nearest_profile(profile, public_profiles),
        }
        records.append(record)
        print(
            f"[{index:03d}/{len(wells):03d}] {well} rows={int(eval_mask.sum())} "
            f"base={rmse(truth, anchor):.3f} post={rmse(truth, posterior):.3f} "
            f"map={result.metadata['map_name']} conf={result.metadata['confidence']:.3f}",
            flush=True,
        )

    if not records:
        raise RuntimeError("No native masks evaluated")
    policy, _ = apply_oof_policy(records, args.min_policy_bin)
    variants = [
        summary("anchor", records, "anchor"),
        summary("heel_prior", records, "prior"),
        summary("map_path", records, "map_path"),
        summary("posterior", records, "posterior"),
        summary("hybrid_oof", records, "hybrid_oof"),
    ]
    summary_frame = pd.DataFrame(variants).sort_values("row_rmse")

    profile_rows = []
    for profile_name in sorted(set(record["nearest_public_profile"] for record in records)):
        subset = [record for record in records if record["nearest_public_profile"] == profile_name]
        for variant, key in (("anchor", "anchor"), ("posterior", "posterior"), ("hybrid_oof", "hybrid_oof")):
            row = summary(f"{profile_name}:{variant}", subset, key)
            row["profile"] = profile_name
            row["profile_variant"] = variant
            profile_rows.append(row)
    profile_frame = pd.DataFrame(profile_rows)

    base = next(row for row in variants if row["variant"] == "anchor")
    posterior_row = next(row for row in variants if row["variant"] == "posterior")
    hybrid_row = next(row for row in variants if row["variant"] == "hybrid_oof")
    profile_improvements = []
    for profile_name in profile_frame["profile"].unique():
        group = profile_frame[profile_frame["profile"] == profile_name].set_index("profile_variant")
        profile_improvements.append(float(group.loc["anchor", "row_rmse"] - group.loc["posterior", "row_rmse"]))
    gate = {
        "anchor_kind": args.anchor,
        "wells": int(len(records)),
        "config": config.__dict__,
        "policy": policy,
        "posterior_row_gain": float(base["row_rmse"] - posterior_row["row_rmse"]),
        "posterior_median_gain": float(base["well_median"] - posterior_row["well_median"]),
        "posterior_p90_delta": float(posterior_row["well_p90"] - base["well_p90"]),
        "profiles_improved": int(sum(value > 0 for value in profile_improvements)),
        "profile_count": int(len(profile_improvements)),
        "hybrid_row_gain": float(base["row_rmse"] - hybrid_row["row_rmse"]),
        "hybrid_well_mean_delta": float(hybrid_row["well_mean"] - base["well_mean"]),
    }
    gate["posterior_gate"] = bool(
        gate["posterior_row_gain"] >= 0.60
        and gate["posterior_median_gain"] >= 0.50
        and gate["posterior_p90_delta"] <= 0.50
        and gate["profiles_improved"] >= min(2, gate["profile_count"])
    )
    gate["hybrid_gate"] = bool(
        gate["hybrid_row_gain"] >= 0.35 and gate["hybrid_well_mean_delta"] <= 0.0
    )
    gate["submission_gate_valid"] = bool(args.anchor in {"pf", "csv"})

    detail_rows = []
    for record in records:
        detail_rows.append(
            {
                "well": record["well"],
                "fold": record["fold"],
                "known_frac": record["known_frac"],
                "eval_rows": int(record["eval_rows"]),
                "gr_missing": record["gr_missing"],
                "z_span": record["z_span"],
                "nearest_public_profile": record["nearest_public_profile"],
                "confidence_tier": confidence_tier(record),
                "profile_bin": profile_bin(record),
                "alpha_oof": record["alpha_oof"],
                "anchor_rmse": rmse(record["truth"], record["anchor"]),
                "prior_rmse": rmse(record["truth"], record["prior"]),
                "map_rmse": rmse(record["truth"], record["map_path"]),
                "posterior_rmse": rmse(record["truth"], record["posterior"]),
                "hybrid_oof_rmse": rmse(record["truth"], record["hybrid_oof"]),
                **{
                    key: record["metadata"].get(key)
                    for key in (
                        "map_name",
                        "map_datum",
                        "second_mode_gap",
                        "second_score_margin",
                        "ambiguous",
                        "entropy",
                        "confidence",
                        "calibration_rmse",
                        "prefix_replay_rmse",
                        "gr_coverage",
                        "base_score",
                        "map_score",
                        "map_score_gain",
                    )
                },
            }
        )
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    if args.anchor_output is not None:
        anchor_rows = []
        for record in records:
            anchor_rows.extend(
                {"id": row_id, "tvt": float(value)}
                for row_id, value in zip(record["row_ids"], record["anchor"])
            )
        args.anchor_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(anchor_rows).to_csv(args.anchor_output, index=False)
    pd.DataFrame(detail_rows).to_csv(args.details_output, index=False)
    summary_frame.to_csv(args.summary_output, index=False)
    profile_frame.to_csv(args.summary_output.with_name(args.summary_output.stem + "_profiles.csv"), index=False)
    args.gate_output.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    print("\nSummary")
    print(summary_frame.to_string(index=False))
    print("\nGate")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
