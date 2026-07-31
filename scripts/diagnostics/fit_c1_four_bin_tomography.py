from __future__ import annotations

import argparse
import json
import math

import numpy as np


HADAMARD_4 = np.asarray(
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ],
    dtype=float,
)


def fit_response(
    base_score: float,
    plus_score: float,
    minus_score: float,
    coded_scores: tuple[float, float, float],
) -> dict[str, object]:
    base2 = float(base_score) ** 2
    plus2 = float(plus_score) ** 2
    minus2 = float(minus_score) ** 2
    total_projection = (plus2 - minus2) / 4.0
    total_energy = (plus2 + minus2) / 2.0 - base2
    if total_energy <= 0.0:
        raise ValueError(f"non-positive C1 direction energy: {total_energy}")

    coded_projection = [
        (float(score) ** 2 - base2 - total_energy) / 2.0
        for score in coded_scores
    ]
    code_projection = np.asarray(
        [total_projection, *coded_projection], dtype=float
    )
    bin_projection = HADAMARD_4.T @ code_projection / 4.0

    global_alpha = -total_projection / total_energy
    global_pred2 = base2 - total_projection**2 / total_energy

    # The hidden rerun does not expose per-bin direction energies. The runtime
    # partition is approximately balanced, so this is a planning estimate only.
    equal_bin_energy = total_energy / 4.0
    bin_alpha_equal_energy = -bin_projection / equal_bin_energy
    bin_alpha_clamped = np.clip(bin_alpha_equal_energy, -0.5, 1.0)
    estimated_pred2 = base2 + float(
        np.sum(
            2.0 * bin_alpha_clamped * bin_projection
            + bin_alpha_clamped**2 * equal_bin_energy
        )
    )

    return {
        "base_score": float(base_score),
        "plus_score": float(plus_score),
        "minus_score": float(minus_score),
        "coded_scores": [float(value) for value in coded_scores],
        "total_projection": float(total_projection),
        "total_direction_energy": float(total_energy),
        "global_alpha": float(global_alpha),
        "global_predicted_score": math.sqrt(max(global_pred2, 0.0)),
        "code_projection": code_projection.tolist(),
        "bin_projection": bin_projection.tolist(),
        "equal_energy_assumption_per_bin": float(equal_bin_energy),
        "bin_alpha_equal_energy": bin_alpha_equal_energy.tolist(),
        "bin_alpha_clamped": bin_alpha_clamped.tolist(),
        "estimated_clamped_bin_score": math.sqrt(max(estimated_pred2, 0.0)),
        "warning": (
            "Per-bin alpha and score assume equal hidden direction energy; "
            "use projection signs first and keep calibration conservative."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=float, required=True)
    parser.add_argument("--plus", type=float, required=True)
    parser.add_argument("--minus", type=float, required=True)
    parser.add_argument("--ppmm", type=float, required=True)
    parser.add_argument("--pmpm", type=float, required=True)
    parser.add_argument("--pmmp", type=float, required=True)
    args = parser.parse_args()
    result = fit_response(
        args.base,
        args.plus,
        args.minus,
        (args.ppmm, args.pmpm, args.pmmp),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
