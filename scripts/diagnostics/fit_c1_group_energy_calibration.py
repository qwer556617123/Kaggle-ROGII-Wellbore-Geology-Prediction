"""Recover all four hidden C1 bin energies and calibrated coefficients."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from fit_c1_four_bin_tomography import fit_response


def rounded_clip(values: np.ndarray, step: float, lower: float, upper: float) -> np.ndarray:
    clipped = np.clip(values, lower, upper)
    return np.round(clipped / step) * step


def fit_group_calibration(
    base_score: float,
    plus_score: float,
    minus_score: float,
    coded_scores: tuple[float, float, float],
    bin0_score: float,
    bin3_score: float,
    selective_score: float,
    selective_alpha: float = 0.35,
    round_step: float = 0.025,
    lower: float = -0.5,
    upper: float = 1.25,
) -> dict[str, object]:
    response = fit_response(
        base_score, plus_score, minus_score, coded_scores
    )
    projection = np.asarray(response["bin_projection"], dtype=float)
    total_energy = float(response["total_direction_energy"])
    base2 = float(base_score) ** 2

    energy = np.empty(4, dtype=float)
    energy[0] = float(bin0_score) ** 2 - base2 - 2.0 * projection[0]
    energy[3] = float(bin3_score) ** 2 - base2 - 2.0 * projection[3]

    active = np.asarray([0, 2, 3], dtype=int)
    active_energy = (
        float(selective_score) ** 2
        - base2
        - 2.0 * selective_alpha * float(projection[active].sum())
    ) / (selective_alpha**2)
    energy[2] = active_energy - energy[0] - energy[3]
    energy[1] = total_energy - float(energy[[0, 2, 3]].sum())

    if not np.isfinite(energy).all() or (energy <= 0.0).any():
        raise ValueError(f"recovered non-positive bin energy: {energy.tolist()}")
    energy_error = float(energy.sum() - total_energy)
    if abs(energy_error) > 1.0e-8:
        raise ValueError(f"bin energies do not sum to total: {energy_error}")

    raw_alpha = -projection / energy
    deployed_alpha = rounded_clip(raw_alpha, round_step, lower, upper)
    raw_score2 = base2 - float(np.sum(projection * projection / energy))
    deployed_score2 = base2 + float(
        np.sum(2.0 * deployed_alpha * projection + deployed_alpha**2 * energy)
    )
    return {
        "base_score": float(base_score),
        "bin_projection": projection.tolist(),
        "bin_energy": energy.tolist(),
        "total_direction_energy": total_energy,
        "energy_sum_error": energy_error,
        "raw_bin_alpha": raw_alpha.tolist(),
        "deployed_bin_alpha": deployed_alpha.tolist(),
        "round_step": float(round_step),
        "clamp": [float(lower), float(upper)],
        "raw_predicted_score": math.sqrt(max(raw_score2, 0.0)),
        "deployed_predicted_score": math.sqrt(max(deployed_score2, 0.0)),
        "calibration_scores": {
            "bin0": float(bin0_score),
            "bin3": float(bin3_score),
            "selective": float(selective_score),
            "selective_alpha": float(selective_alpha),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=float, default=6.696)
    parser.add_argument("--plus", type=float, default=6.905)
    parser.add_argument("--minus", type=float, default=7.802)
    parser.add_argument("--ppmm", type=float, default=7.623)
    parser.add_argument("--pmpm", type=float, default=7.440)
    parser.add_argument("--pmmp", type=float, default=6.824)
    parser.add_argument("--bin0", type=float, required=True)
    parser.add_argument("--bin3", type=float, required=True)
    parser.add_argument("--selective", type=float, required=True)
    parser.add_argument("--selective-alpha", type=float, default=0.35)
    parser.add_argument("--round-step", type=float, default=0.025)
    parser.add_argument("--lower", type=float, default=-0.5)
    parser.add_argument("--upper", type=float, default=1.25)
    args = parser.parse_args()
    result = fit_group_calibration(
        args.base,
        args.plus,
        args.minus,
        (args.ppmm, args.pmpm, args.pmmp),
        args.bin0,
        args.bin3,
        args.selective,
        args.selective_alpha,
        args.round_step,
        args.lower,
        args.upper,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
