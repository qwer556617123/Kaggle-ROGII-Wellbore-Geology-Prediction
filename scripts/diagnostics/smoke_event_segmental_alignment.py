"""Synthetic checks for event-driven segmental datum states."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from event_segmental_alignment import SegmentalConfig, run_segmental
from smoke_multiscale_stratigraphic_alignment import horizontal, load_hmm_module, typewell


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def main() -> None:
    hmm = load_hmm_module()
    tw = typewell()
    hw, eval_mask = horizontal(tw, fault=15.0, fault_row=370, seed=18)
    # Isolate the segmental layer with a smooth base that misses one +15 ft
    # block displacement.  Exact-HMM behavior is covered by the MPSC smoke.
    base = hw["TVT"].to_numpy(dtype=float).copy()
    base[370:] -= 15.0
    config = SegmentalConfig(
        min_segment_rows=80,
        boundary_stride=40,
        change_penalty=10.0,
        magnitude_penalty=10.0,
    )
    first = run_segmental(hw, tw, base, config)
    second = run_segmental(hw, tw, base, config)
    truth = hw.loc[eval_mask, "TVT"].to_numpy(float)
    base_score = rmse(truth, base[eval_mask])
    candidate_score = rmse(truth, first.pred[eval_mask])
    print(f"segmental raw={base_score:.6f} candidate={candidate_score:.6f} meta={first.metadata}")
    assert np.isfinite(first.pred).all()
    assert np.array_equal(first.pred, second.pred)
    assert candidate_score < base_score
    assert first.metadata["correction_max_abs"] <= 35.0 + 1e-9
    assert len(first.metadata["map_boundaries"]) <= 2
    print(
        json.dumps(
            {
                "base_rmse": base_score,
                "candidate_rmse": candidate_score,
                "gain": base_score - candidate_score,
                "map_boundaries": first.metadata["map_boundaries"],
                "map_residuals": first.metadata["map_residuals"],
                "correction_rms": first.metadata["correction_rms"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
