"""Decode four Hadamard rank-well probes into calibrated datum offsets."""
from __future__ import annotations

import argparse
import json

import numpy as np


CODES = ("ppp", "pmm", "mpm", "mmp")
MATRIX = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
)


def decode(
    baseline: float,
    scores: np.ndarray,
    amplitude: float,
    row_fractions: np.ndarray,
) -> dict:
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    if scores.shape != (4,):
        raise ValueError("exactly four coded scores are required")
    if row_fractions.shape != (3,) or np.any(row_fractions <= 0):
        raise ValueError("three positive row fractions are required")
    row_fractions = row_fractions / row_fractions.sum()

    coded_projection = (
        scores * scores - baseline * baseline - amplitude * amplitude
    ) / (2.0 * amplitude)
    weighted_mean_residual, _, _, _ = np.linalg.lstsq(
        MATRIX, coded_projection, rcond=None
    )
    reconstructed = MATRIX @ weighted_mean_residual
    equation_residual = coded_projection - reconstructed
    raw_offsets = -weighted_mean_residual / row_fractions
    rounded_offsets = np.round(raw_offsets / 0.025) * 0.025
    clamped_offsets = np.clip(rounded_offsets, -10.0, 10.0)
    gain_sq_unclamped = float(
        np.sum(weighted_mean_residual * weighted_mean_residual / row_fractions)
    )
    predicted_unclamped = float(
        np.sqrt(max(0.0, baseline * baseline - gain_sq_unclamped))
    )
    applied_gain_sq = float(
        -2.0 * np.dot(weighted_mean_residual, clamped_offsets)
        - np.dot(row_fractions, clamped_offsets * clamped_offsets)
    )
    predicted_clamped = float(
        np.sqrt(max(0.0, baseline * baseline - applied_gain_sq))
    )
    return {
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "codes": list(CODES),
        "scores": scores.tolist(),
        "row_fractions": row_fractions.tolist(),
        "coded_projection": coded_projection.tolist(),
        "weighted_mean_residual_by_rank": weighted_mean_residual.tolist(),
        "equation_residual": equation_residual.tolist(),
        "equation_residual_max_abs": float(np.max(np.abs(equation_residual))),
        "raw_offset_by_rank": raw_offsets.tolist(),
        "rounded_clamped_offset_by_rank": clamped_offsets.tolist(),
        "predicted_score_unclamped": predicted_unclamped,
        "predicted_score_rounded_clamped": predicted_clamped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=float, default=6.720)
    parser.add_argument("--amplitude", type=float, default=2.0)
    for code in CODES:
        parser.add_argument(f"--{code}", type=float, required=True)
    parser.add_argument(
        "--row-fractions",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("RANK0", "RANK1", "RANK2"),
    )
    args = parser.parse_args()
    scores = np.asarray([getattr(args, code) for code in CODES], dtype=float)
    result = decode(
        baseline=float(args.baseline),
        scores=scores,
        amplitude=float(args.amplitude),
        row_fractions=np.asarray(args.row_fractions, dtype=float),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
