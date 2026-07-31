"""Evaluate MPSC profiles and analog references on native train masks."""
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

from multiscale_stratigraphic_alignment import (
    MPSCConfig,
    mix_reference_results,
    reference_prefix_loss,
    run_mpsc,
)


BLEND_GRID = (0.0, 0.10, 0.20, 0.30, 0.50, 1.0)


def load_hmm_module(notebook_path: Path) -> types.ModuleType:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][58]["source"])
    source = source.replace("@njit(cache=True, nogil=True)", "@njit(cache=False, nogil=True)")
    if "def run_hmm2" not in source or "def _hmm2_fb" not in source:
        raise RuntimeError("Exact HMM core was not found in the source notebook")
    module = types.ModuleType("mpsc_native_hmm")
    module.__file__ = str(notebook_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(notebook_path), "exec"), module.__dict__)
    return module


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(pred)))))


def well_catalog(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
        first = pd.read_csv(path, usecols=["X", "Y"], nrows=1)
        if first.empty:
            continue
        rows.append(
            {
                "well": path.name.split("__")[0],
                "x": float(first.iloc[0]["X"]),
                "y": float(first.iloc[0]["Y"]),
            }
        )
    return pd.DataFrame(rows).sort_values("well").reset_index(drop=True)


def select_wells(catalog: pd.DataFrame, limit: int, seed: int) -> list[str]:
    wells = catalog["well"].astype(str).tolist()
    if 0 < limit < len(wells):
        wells = sorted(
            np.random.default_rng(seed).choice(wells, size=limit, replace=False).tolist()
        )
    return wells


def public_profiles(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data_dir / "test").glob("*__horizontal_well.csv")):
        hw = pd.read_csv(path, usecols=["Z", "GR", "TVT_input"])
        eval_mask = hw["TVT_input"].isna()
        rows.append(
            {
                "profile": path.name.split("__")[0],
                "known_frac": float((~eval_mask).mean()),
                "eval_rows": float(eval_mask.sum()),
                "gr_missing": float(hw.loc[eval_mask, "GR"].isna().mean()),
                "z_span": float(hw.loc[eval_mask, "Z"].max() - hw.loc[eval_mask, "Z"].min()),
            }
        )
    return pd.DataFrame(rows)


def profile_for_well(hw: pd.DataFrame, profiles: pd.DataFrame) -> str:
    eval_mask = hw["TVT_input"].isna()
    values = np.asarray(
        [
            float((~eval_mask).mean()),
            float(eval_mask.sum()),
            float(hw.loc[eval_mask, "GR"].isna().mean()),
            float(hw.loc[eval_mask, "Z"].max() - hw.loc[eval_mask, "Z"].min()),
        ]
    )
    matrix = profiles[["known_frac", "eval_rows", "gr_missing", "z_span"]].to_numpy(float)
    scale = np.maximum(np.std(matrix, axis=0), np.asarray([0.05, 1000.0, 0.05, 10.0]))
    distance = np.sum(np.square((matrix - values) / scale), axis=1)
    return str(profiles.iloc[int(np.argmin(distance))]["profile"])


def choose_analog(
    well: str,
    hw: pd.DataFrame,
    data_dir: Path,
    catalog: pd.DataFrame,
    config: MPSCConfig,
    nearest: int,
) -> tuple[str | None, pd.DataFrame | None, dict[str, float]]:
    target = catalog[catalog["well"] == well].iloc[0]
    candidates = catalog[catalog["well"] != well].copy()
    candidates["distance"] = np.square(candidates["x"] - float(target["x"]))
    candidates["distance"] += np.square(candidates["y"] - float(target["y"]))
    best: tuple[str | None, pd.DataFrame | None, dict[str, float]] = (
        None,
        None,
        {"loss": float("inf"), "rows": 0, "rmse": float("inf")},
    )
    for candidate in candidates.nsmallest(nearest, "distance")["well"].astype(str):
        tw_path = data_dir / "train" / f"{candidate}__typewell.csv"
        tw = pd.read_csv(tw_path, usecols=["TVT", "GR"])
        score = reference_prefix_loss(hw, tw, config)
        if score["rows"] >= 80 and score["loss"] < best[2]["loss"]:
            best = (candidate, tw, score)
    return best


def evaluate_one(
    well: str,
    data_dir: Path,
    catalog: pd.DataFrame,
    profiles: pd.DataFrame,
    profile_names: tuple[str, ...],
    analog: bool,
    nearest: int,
    scale_mode: str,
    hmm: Any,
) -> dict[str, Any]:
    started = time.time()
    hw = pd.read_csv(data_dir / "train" / f"{well}__horizontal_well.csv")
    tw = pd.read_csv(data_dir / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    eval_mask = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    if not eval_mask.any():
        raise RuntimeError(f"No native eval rows for {well}")
    truth = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
    raw_result = hmm.run_hmm2(
        hw[hmm.TEST_COLS].copy(),
        tw,
        params=hmm.HMMParams(emission="t", sigma_mode="std"),
    )
    raw_full = np.asarray(raw_result["pred"], dtype=float)
    raw = raw_full[eval_mask]
    candidates: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name in profile_names:
        config = MPSCConfig(
            profile=name,
            horizontal_scale_mode=scale_mode,
            use_coarse_corridor=scale_mode != "anchor_path",
        )
        own = run_mpsc(
            hw,
            tw,
            hmm._hmm2_fb,
            config,
            scale_path=raw_full if scale_mode == "anchor_path" else None,
        )
        chosen = own
        chosen_meta = dict(own.metadata)
        if analog:
            analog_well, analog_tw, analog_score = choose_analog(
                well, hw, data_dir, catalog, config, nearest
            )
            if analog_tw is not None:
                analog_result = run_mpsc(
                    hw,
                    analog_tw,
                    hmm._hmm2_fb,
                    config,
                    scale_path=raw_full if scale_mode == "anchor_path" else None,
                )
                chosen = mix_reference_results(own, analog_result)
                chosen_meta = dict(chosen.metadata)
                chosen_meta.update(
                    {
                        "analog_well": analog_well,
                        "analog_selection_loss": analog_score["loss"],
                        "analog_selection_rows": analog_score["rows"],
                    }
                )
            else:
                chosen_meta.update({"analog_well": None, "analog_weight": 0.0})
        candidates[name] = chosen.pred[eval_mask]
        metadata[name] = chosen_meta
    return {
        "well": well,
        "public_profile": profile_for_well(hw, profiles),
        "truth": truth,
        "raw": raw,
        "candidates": candidates,
        "metadata": metadata,
        "runtime_sec": float(time.time() - started),
    }


def summarize(records: list[dict[str, Any]], predictions: list[np.ndarray]) -> dict[str, float]:
    truth = np.concatenate([record["truth"] for record in records])
    pred = np.concatenate(predictions)
    well_rmse = np.asarray(
        [rmse(record["truth"], candidate) for record, candidate in zip(records, predictions)]
    )
    well_sse = np.asarray(
        [float(np.sum(np.square(record["truth"] - candidate))) for record, candidate in zip(records, predictions)]
    )
    worst_count = max(1, int(np.ceil(0.10 * len(well_sse))))
    return {
        "row_rmse": rmse(truth, pred),
        "well_mean": float(well_rmse.mean()),
        "well_median": float(np.median(well_rmse)),
        "well_p90": float(np.quantile(well_rmse, 0.90)),
        "worst10_sse_share": float(np.sort(well_sse)[-worst_count:].sum() / well_sse.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--hmm-notebook",
        type=Path,
        default=Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb"),
    )
    parser.add_argument("--profiles", nargs="+", default=["raw", "balanced", "coarse"])
    parser.add_argument("--well-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--analog", action="store_true")
    parser.add_argument("--nearest", type=int, default=16)
    parser.add_argument(
        "--scale-mode", choices=["prefix", "anchor_path"], default="prefix"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mpsc_native_gate.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    unknown = sorted(set(args.profiles) - {"raw", "balanced", "coarse"})
    if unknown:
        raise ValueError(f"Unsupported profiles: {unknown}")
    catalog = well_catalog(args.data_dir)
    wells = select_wells(catalog, args.well_limit, args.seed)
    test_profiles = public_profiles(args.data_dir)
    hmm = load_hmm_module(args.hmm_notebook)
    records = Parallel(n_jobs=min(args.workers, len(wells)), prefer="threads")(
        delayed(evaluate_one)(
            well,
            args.data_dir,
            catalog,
            test_profiles,
            tuple(args.profiles),
            args.analog,
            args.nearest,
            args.scale_mode,
            hmm,
        )
        for well in wells
    )
    raw_metrics = summarize(records, [record["raw"] for record in records])
    candidates: dict[str, dict[str, dict[str, float]]] = {}
    for profile in args.profiles:
        direction = [record["candidates"][profile] - record["raw"] for record in records]
        candidates[profile] = {}
        for weight in BLEND_GRID:
            predictions = [
                record["raw"] + weight * delta for record, delta in zip(records, direction)
            ]
            candidates[profile][f"{weight:.2f}"] = summarize(records, predictions)

    profile_rank = sorted(
        args.profiles,
        key=lambda name: candidates[name]["0.20"]["row_rmse"],
    )
    selected = profile_rank[0]
    if "balanced" in profile_rank:
        best_rmse = candidates[selected]["0.20"]["row_rmse"]
        balanced_rmse = candidates["balanced"]["0.20"]["row_rmse"]
        if balanced_rmse - best_rmse < 0.05:
            selected = "balanced"
    selected_metrics = candidates[selected]["0.20"]
    profile_improvements = {}
    for public_profile in test_profiles["profile"].astype(str):
        subset = [record for record in records if record["public_profile"] == public_profile]
        if not subset:
            profile_improvements[public_profile] = None
            continue
        base = summarize(subset, [record["raw"] for record in subset])
        candidate = summarize(
            subset,
            [
                record["raw"]
                + 0.20 * (record["candidates"][selected] - record["raw"])
                for record in subset
            ],
        )
        profile_improvements[public_profile] = base["row_rmse"] - candidate["row_rmse"]
    profiles_improved = sum(
        value is not None and value > 0 for value in profile_improvements.values()
    )
    row_gain = raw_metrics["row_rmse"] - selected_metrics["row_rmse"]
    median_gain = raw_metrics["well_median"] - selected_metrics["well_median"]
    p90_delta = selected_metrics["well_p90"] - raw_metrics["well_p90"]
    verdict = "PASS" if (
        row_gain >= 0.30
        and median_gain >= 0.20
        and p90_delta <= 0.25
        and profiles_improved >= 2
    ) else "STOP"
    details = []
    for record in records:
        selected_meta = record["metadata"][selected]
        details.append(
            {
                "well": record["well"],
                "public_profile": record["public_profile"],
                "rows": int(len(record["truth"])),
                "raw_rmse": rmse(record["truth"], record["raw"]),
                "candidate_rmse": rmse(record["truth"], record["candidates"][selected]),
                "blend020_rmse": rmse(
                    record["truth"],
                    record["raw"]
                    + 0.20 * (record["candidates"][selected] - record["raw"]),
                ),
                "prefix_loss": selected_meta.get("prefix_loss"),
                "active_features": selected_meta.get("active_features", []),
                "feature_quality": {
                    name: {
                        "correlation": values.get("correlation"),
                        "loss": values.get("loss"),
                        "sigma": values.get("sigma"),
                        "active": values.get("active", False),
                    }
                    for name, values in selected_meta.get("calibrations", {}).items()
                },
                "gr_coverage": selected_meta.get("gr_coverage"),
                "entropy": selected_meta.get("entropy"),
                "analog_well": selected_meta.get("analog_well"),
                "analog_weight": selected_meta.get("analog_weight", 0.0),
                "runtime_sec": record["runtime_sec"],
            }
        )
    result = {
        "verdict": verdict,
        "selected_profile": selected,
        "analog_enabled": bool(args.analog),
        "scale_mode": args.scale_mode,
        "wells": len(records),
        "rows": int(sum(len(record["truth"]) for record in records)),
        "runtime_sec": float(time.time() - started),
        "raw": raw_metrics,
        "candidates": candidates,
        "gate": {
            "row_gain": row_gain,
            "well_median_gain": median_gain,
            "well_p90_delta": p90_delta,
            "profiles_improved": profiles_improved,
            "profile_improvements": profile_improvements,
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    details_path = args.output.with_name(args.output.stem + "_details.csv")
    pd.DataFrame(details).to_csv(details_path, index=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
