"""Synthetic smoke tests for multiscale probabilistic stratigraphic alignment."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from multiscale_stratigraphic_alignment import (
    MPSCConfig,
    mix_reference_results,
    reference_prefix_loss,
    run_mpsc,
)


HMM_NOTEBOOK = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")


def load_hmm_module() -> types.ModuleType:
    notebook = json.loads(HMM_NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][58]["source"])
    source = source.replace("@njit(cache=True, nogil=True)", "@njit(cache=False, nogil=True)")
    module = types.ModuleType("mpsc_smoke_hmm")
    module.__file__ = str(HMM_NOTEBOOK)
    sys.modules[module.__name__] = module
    exec(compile(source, str(HMM_NOTEBOOK), "exec"), module.__dict__)
    return module


def typewell(phase: float = 0.0, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    tvt = np.arange(980.0, 1320.0, 0.5)
    gr = 75.0
    gr += 18.0 * np.sin(2.0 * np.pi * (tvt + phase) / 18.0)
    gr += 10.0 * np.sin(2.0 * np.pi * (tvt + phase) / 55.0)
    gr += 6.0 * np.cos(2.0 * np.pi * (tvt + phase) / 120.0)
    gr += 32.0 * np.exp(-0.5 * np.square((tvt - 1148.0) / 9.0))
    gr -= 18.0 * np.exp(-0.5 * np.square((tvt - 1215.0) / 13.0))
    if noise:
        gr += np.random.default_rng(seed).normal(0.0, noise, len(gr))
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def horizontal(
    tw: pd.DataFrame,
    *,
    missing: bool = False,
    fault: float = 0.0,
    fault_row: int = 430,
    variable_rate: bool = False,
    seed: int = 7,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = 620
    md = np.arange(rows, dtype=float)
    z = np.full(rows, -9000.0)
    if variable_rate:
        true_tvt = 1080.0 + 0.18 * md + 18.0 * np.sin(md / 38.0)
    else:
        true_tvt = 1040.0 + 0.25 * md + 1.5 * np.sin(md / 170.0)
    if fault:
        true_tvt[fault_row:] += fault
    expected = np.interp(true_tvt, tw["TVT"], tw["GR"])
    rng = np.random.default_rng(seed)
    gr = 1.45 * expected - 21.0 + rng.normal(0.0, 13.0, rows)
    if missing:
        gr[310:430] = np.nan
    known_rows = 170
    tvt_input = np.full(rows, np.nan)
    tvt_input[:known_rows] = true_tvt[:known_rows]
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": 2_980_000.0 + md,
            "Y": np.full(rows, 1_060_000.0),
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
        }
    )
    return frame, np.arange(rows) >= known_rows


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def vector_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def main() -> None:
    hmm = load_hmm_module()
    tw = typewell()
    hw, eval_mask = horizontal(tw)
    balanced_config = MPSCConfig(profile="balanced", top_modes=5)
    first = run_mpsc(hw, tw, hmm._hmm2_fb, balanced_config)
    second = run_mpsc(hw, tw, hmm._hmm2_fb, balanced_config)
    assert np.isfinite(first.pred).all()
    assert vector_hash(first.pred) == vector_hash(second.pred)
    assert first.metadata["active_features"]
    assert 0.0 < first.metadata["corridor_fraction"] <= 1.0
    assert len(first.metadata["mode_datums"]) >= 2

    raw = hmm.run_hmm2(
        hw[hmm.TEST_COLS].copy(),
        tw,
        params=hmm.HMMParams(emission="t", sigma_mode="std"),
    )
    truth = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
    raw_rmse = rmse(truth, np.asarray(raw["pred"])[eval_mask])
    balanced_rmse = rmse(truth, first.pred[eval_mask])
    path_aware = run_mpsc(
        hw,
        tw,
        hmm._hmm2_fb,
        MPSCConfig(
            profile="balanced",
            horizontal_scale_mode="anchor_path",
            use_coarse_corridor=False,
        ),
        scale_path=np.asarray(raw["pred"], dtype=float),
    )
    path_aware_rmse = rmse(truth, path_aware.pred[eval_mask])
    print(f"synthetic raw={raw_rmse:.6f} balanced={balanced_rmse:.6f}", flush=True)
    assert balanced_rmse < raw_rmse
    assert path_aware_rmse <= balanced_rmse
    assert path_aware.metadata["path_distance_span_ft"] > 0

    missing_hw, missing_mask = horizontal(tw, missing=True)
    missing = run_mpsc(missing_hw, tw, hmm._hmm2_fb, balanced_config)
    assert np.isfinite(missing.pred).all()
    assert missing.metadata["gr_coverage"] < 1.0
    assert np.max(np.abs(missing.pred[missing_mask] - missing_hw.loc[missing_mask, "TVT"])) < 100.0

    reversal_hw, reversal_mask = horizontal(tw, missing=True, variable_rate=True)
    reversal_raw = hmm.run_hmm2(
        reversal_hw[hmm.TEST_COLS].copy(),
        tw,
        params=hmm.HMMParams(emission="t", sigma_mode="std"),
    )
    reversal = run_mpsc(
        reversal_hw,
        tw,
        hmm._hmm2_fb,
        MPSCConfig(
            profile="balanced",
            horizontal_scale_mode="anchor_path",
            use_coarse_corridor=False,
        ),
        scale_path=np.asarray(reversal_raw["pred"], dtype=float),
    )
    assert np.isfinite(reversal.pred).all()
    assert reversal.metadata["path_distance_span_ft"] > 0
    reversal_truth = reversal_hw["TVT"].to_numpy(dtype=float)
    assert np.any(np.diff(reversal_truth) < 0) and np.any(np.diff(reversal_truth) > 0)
    heel_datum = float(reversal_hw.loc[~reversal_mask, "TVT_input"].iloc[-1])
    assert np.max(np.abs(reversal.pred[reversal_mask] - heel_datum)) <= 101.0

    fault_hw, fault_mask = horizontal(tw, fault=15.0)
    fault = run_mpsc(fault_hw, tw, hmm._hmm2_fb, balanced_config)
    assert np.isfinite(fault.pred).all()
    assert np.max(np.abs(fault.pred[fault_mask] - fault_hw.loc[fault_mask, "TVT"])) <= 100.0

    compatible = typewell(noise=1.0, seed=22)
    incompatible = typewell()
    wrong_rng = np.random.default_rng(23)
    incompatible["GR"] = 70.0 + np.cumsum(wrong_rng.normal(0.0, 1.2, len(incompatible)))
    good_loss = reference_prefix_loss(hw, compatible, balanced_config)
    bad_loss = reference_prefix_loss(hw, incompatible, balanced_config)
    print(f"reference good={good_loss} bad={bad_loss}", flush=True)
    assert good_loss["rows"] >= 80
    assert good_loss["loss"] < bad_loss["loss"]

    analog = run_mpsc(hw, compatible, hmm._hmm2_fb, balanced_config)
    mixed = mix_reference_results(first, analog)
    assert np.isfinite(mixed.pred).all()
    assert 0.0 <= mixed.metadata["analog_weight"] <= 1.0
    rejected = mix_reference_results(
        first,
        run_mpsc(hw, incompatible, hmm._hmm2_fb, balanced_config),
    )
    assert rejected.metadata["own_weight"] > rejected.metadata["analog_weight"]

    summary = {
        "raw_rmse": raw_rmse,
        "balanced_rmse": balanced_rmse,
        "path_aware_rmse": path_aware_rmse,
        "gain": raw_rmse - balanced_rmse,
        "missing_coverage": missing.metadata["gr_coverage"],
        "reversal_coverage": reversal.metadata["gr_coverage"],
        "mode_datums": first.metadata["mode_datums"],
        "good_reference_loss": good_loss["loss"],
        "bad_reference_loss": bad_loss["loss"],
        "mixed_analog_weight": mixed.metadata["analog_weight"],
        "sha256": vector_hash(first.pred),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
