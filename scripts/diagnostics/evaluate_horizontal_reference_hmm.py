"""Evaluate formation-warped horizontal references with the exact Student-t HMM."""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from horizontal_stratigraphic_reference import HorizontalReferenceBank, choice_to_dict


WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def load_hmm_module(notebook_path: Path) -> types.ModuleType:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][58]["source"])
    source = source.replace("@njit(cache=True, nogil=True)", "@njit(cache=False, nogil=True)")
    module = types.ModuleType("horizontal_reference_hmm_runtime")
    module.__file__ = str(notebook_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(notebook_path), "exec"), module.__dict__)
    return module


def metrics(records: list[dict], prediction_key: str) -> dict[str, float]:
    truth = np.concatenate([item["truth"] for item in records])
    pred = np.concatenate([item[prediction_key] for item in records])
    per_well = np.asarray(
        [np.sqrt(np.mean(np.square(item["truth"] - item[prediction_key]))) for item in records]
    )
    return {
        "row_rmse": float(np.sqrt(np.mean(np.square(truth - pred)))),
        "well_mean": float(per_well.mean()),
        "well_median": float(np.median(per_well)),
        "well_p90": float(np.quantile(per_well, 0.90)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--hmm-notebook",
        type=Path,
        default=Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb"),
    )
    parser.add_argument("--anchor", type=Path, default=Path("reports/native100_pf_anchor.csv"))
    parser.add_argument("--well-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, default=Path("reports/horizontal_reference_hmm.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    hmm = load_hmm_module(args.hmm_notebook)
    bank = HorizontalReferenceBank(args.data_dir)
    anchor = pd.read_csv(args.anchor, usecols=["id", "tvt"])
    anchor["id"] = anchor["id"].astype(str)
    anchor_map = dict(zip(anchor["id"], anchor["tvt"].astype(float)))
    wells = sorted(anchor["id"].str.rsplit("_", n=1).str[0].unique())
    if 0 < args.well_limit < len(wells):
        wells = sorted(
            np.random.default_rng(args.seed).choice(wells, size=args.well_limit, replace=False)
        )

    records = []
    params = hmm.HMMParams(emission="t", sigma_mode="std")
    for position, well in enumerate(wells, start=1):
        well_started = time.time()
        horizontal = pd.read_csv(args.data_dir / "train" / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(args.data_dir / "train" / f"{well}__typewell.csv")
        choice = bank.choose_reference(str(well), horizontal, typewell)
        own = hmm.run_hmm2(horizontal[hmm.TEST_COLS].copy(), typewell[["TVT", "GR"]], params=params)
        selected = hmm.run_hmm2(
            horizontal[hmm.TEST_COLS].copy(), choice.reference[["TVT", "GR"]], params=params
        )
        eval_mask = horizontal["TVT_input"].isna().to_numpy()
        rows = np.flatnonzero(eval_mask)
        ids = [f"{well}_{int(row)}" for row in rows]
        anchor_values = np.asarray([anchor_map.get(row_id, np.nan) for row_id in ids], dtype=float)
        if not np.isfinite(anchor_values).all():
            raise RuntimeError(f"Anchor is incomplete for {well}")
        record = {
            "well": str(well),
            "truth": horizontal.loc[eval_mask, "TVT"].to_numpy(dtype=float),
            "anchor": anchor_values,
            "own": np.asarray(own["pred"], dtype=float)[eval_mask],
            "selected": np.asarray(selected["pred"], dtype=float)[eval_mask],
            "choice": choice_to_dict(choice),
            "runtime_sec": float(time.time() - well_started),
        }
        records.append(record)
        print(
            f"[{position}/{len(wells)}] {well} {choice.name} "
            f"loss={choice.prefix_loss:.3f} runtime={record['runtime_sec']:.1f}s",
            flush=True,
        )

    truth = np.concatenate([item["truth"] for item in records])
    anchor_values = np.concatenate([item["anchor"] for item in records])
    own_values = np.concatenate([item["own"] for item in records])
    selected_values = np.concatenate([item["selected"] for item in records])
    blend_metrics = {}
    for weight in WEIGHTS:
        for label, values in (("own", own_values), ("selected", selected_values)):
            pred = anchor_values + weight * (values - anchor_values)
            blend_metrics[f"{label}_{weight:.2f}"] = float(
                np.sqrt(np.mean(np.square(truth - pred)))
            )

    report = {
        "wells": len(records),
        "rows": int(sum(len(item["truth"]) for item in records)),
        "runtime_sec": float(time.time() - started),
        "metrics": {
            "anchor": metrics(records, "anchor"),
            "own_hmm": metrics(records, "own"),
            "selected_hmm": metrics(records, "selected"),
        },
        "anchor_blends_row_rmse": blend_metrics,
        "selected_non_own_wells": int(sum(item["choice"]["name"] != "own" for item in records)),
        "well_audits": [
            {
                "well": item["well"],
                "choice": item["choice"],
                "runtime_sec": item["runtime_sec"],
            }
            for item in records
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
