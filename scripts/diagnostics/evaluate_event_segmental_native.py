"""Native-mask gate for event-driven semi-Markov datum corrections."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from event_segmental_alignment import SegmentalConfig, run_segmental
from evaluate_multiscale_stratigraphic_alignment import (
    load_hmm_module,
    profile_for_well,
    public_profiles,
    rmse,
    select_wells,
    summarize,
    well_catalog,
)


CONFIGS = {
    "moderate": SegmentalConfig(change_penalty=10.0, magnitude_penalty=10.0),
    "conservative": SegmentalConfig(change_penalty=40.0, magnitude_penalty=10.0),
}
BLENDS = (0.0, 0.05, 0.10, 0.20, 0.30)


def evaluate_one(
    well: str,
    data_dir: Path,
    test_profiles: pd.DataFrame,
    config_names: tuple[str, ...],
    hmm: Any,
) -> dict[str, Any]:
    started = time.time()
    hw = pd.read_csv(data_dir / "train" / f"{well}__horizontal_well.csv")
    tw = pd.read_csv(data_dir / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    eval_mask = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    truth = hw.loc[eval_mask, "TVT"].to_numpy(float)
    raw_result = hmm.run_hmm2(
        hw[hmm.TEST_COLS].copy(),
        tw,
        params=hmm.HMMParams(emission="t", sigma_mode="std"),
    )
    raw_full = np.asarray(raw_result["pred"], dtype=float)
    candidates = {}
    metadata = {}
    for name in config_names:
        result = run_segmental(hw, tw, raw_full, CONFIGS[name])
        candidates[name] = result.pred[eval_mask]
        metadata[name] = result.metadata
    return {
        "well": well,
        "public_profile": profile_for_well(hw, test_profiles),
        "truth": truth,
        "raw": raw_full[eval_mask],
        "candidates": candidates,
        "metadata": metadata,
        "runtime_sec": float(time.time() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--hmm-notebook",
        type=Path,
        default=Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb"),
    )
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--well-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/event_segmental_native_gate.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.configs) - set(CONFIGS))
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}")
    started = time.time()
    catalog = well_catalog(args.data_dir)
    wells = select_wells(catalog, args.well_limit, args.seed)
    test_profiles = public_profiles(args.data_dir)
    hmm = load_hmm_module(args.hmm_notebook)
    records = Parallel(n_jobs=min(args.workers, len(wells)), prefer="threads")(
        delayed(evaluate_one)(well, args.data_dir, test_profiles, tuple(args.configs), hmm)
        for well in wells
    )
    raw_metrics = summarize(records, [record["raw"] for record in records])
    candidates: dict[str, dict[str, dict[str, float]]] = {}
    for name in args.configs:
        candidates[name] = {}
        for weight in BLENDS:
            predictions = [
                record["raw"]
                + weight * (record["candidates"][name] - record["raw"])
                for record in records
            ]
            candidates[name][f"{weight:.2f}"] = summarize(records, predictions)
    selected = min(
        args.configs,
        key=lambda name: candidates[name]["0.20"]["row_rmse"],
    )
    selected_metrics = candidates[selected]["0.20"]
    profile_gains = {}
    for profile in test_profiles["profile"].astype(str):
        subset = [record for record in records if record["public_profile"] == profile]
        if not subset:
            profile_gains[profile] = None
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
        profile_gains[profile] = base["row_rmse"] - candidate["row_rmse"]
    profiles_improved = sum(value is not None and value > 0 for value in profile_gains.values())
    row_gain = raw_metrics["row_rmse"] - selected_metrics["row_rmse"]
    median_gain = raw_metrics["well_median"] - selected_metrics["well_median"]
    p90_delta = selected_metrics["well_p90"] - raw_metrics["well_p90"]
    verdict = "PASS" if (
        row_gain >= 0.30
        and median_gain >= 0.20
        and p90_delta <= 0.25
        and profiles_improved >= 2
    ) else "STOP"
    oracle_predictions = []
    details = []
    for record in records:
        candidate = record["candidates"][selected]
        raw_score = rmse(record["truth"], record["raw"])
        candidate_score = rmse(record["truth"], candidate)
        oracle_predictions.append(candidate if candidate_score < raw_score else record["raw"])
        meta = record["metadata"][selected]
        details.append(
            {
                "well": record["well"],
                "public_profile": record["public_profile"],
                "rows": int(len(record["truth"])),
                "raw_rmse": raw_score,
                "candidate_rmse": candidate_score,
                "blend020_rmse": rmse(
                    record["truth"], record["raw"] + 0.20 * (candidate - record["raw"])
                ),
                "map_boundaries": json.dumps(meta.get("map_boundaries", [])),
                "map_residuals": json.dumps(meta.get("map_residuals", [])),
                "correction_rms": meta.get("correction_rms"),
                "posterior_std_mean": meta.get("posterior_std_mean"),
                "runtime_sec": record["runtime_sec"],
            }
        )
    oracle = summarize(records, oracle_predictions)
    result = {
        "verdict": verdict,
        "selected_config": selected,
        "wells": len(records),
        "rows": int(sum(len(record["truth"]) for record in records)),
        "runtime_sec": float(time.time() - started),
        "raw": raw_metrics,
        "candidates": candidates,
        "oracle": oracle,
        "oracle_row_gain": raw_metrics["row_rmse"] - oracle["row_rmse"],
        "gate": {
            "row_gain": row_gain,
            "well_median_gain": median_gain,
            "well_p90_delta": p90_delta,
            "profiles_improved": profiles_improved,
            "profile_improvements": profile_gains,
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(details).to_csv(
        args.output.with_name(args.output.stem + "_details.csv"), index=False
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
