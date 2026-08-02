"""Decode four row-balanced datum projections from Hadamard code scores."""
from __future__ import annotations

import argparse
import json

import numpy as np


HADAMARD_4 = np.asarray(
    [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
    dtype=float,
)


def decode(anchor: float, scores: tuple[float, ...], amplitude: float) -> dict[str, object]:
    if len(scores) != 4 or amplitude <= 0.0:
        raise ValueError((scores, amplitude))
    anchor2 = float(anchor) ** 2
    code_projection = np.asarray(
        [
            (float(score) ** 2 - anchor2 - float(amplitude) ** 2)
            / (2.0 * float(amplitude))
            for score in scores
        ],
        dtype=float,
    )
    bin_projection = HADAMARD_4.T @ code_projection / 4.0
    equal_fraction = 0.25
    equal_fraction_alpha = -bin_projection / equal_fraction
    equal_fraction_pred2 = anchor2 - float(
        np.sum(bin_projection * bin_projection / equal_fraction)
    )
    return {
        "anchor_score": float(anchor),
        "code_scores": [float(value) for value in scores],
        "amplitude": float(amplitude),
        "code_projection": code_projection.tolist(),
        "bin_projection": bin_projection.tolist(),
        "equal_fraction_alpha": equal_fraction_alpha.tolist(),
        "equal_fraction_predicted_score": float(np.sqrt(max(equal_fraction_pred2, 0.0))),
        "note": "Final notebook should divide each projection by its run-local bin row fraction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=float, required=True)
    parser.add_argument("--pppp", type=float, required=True)
    parser.add_argument("--ppmm", type=float, required=True)
    parser.add_argument("--pmpm", type=float, required=True)
    parser.add_argument("--pmmp", type=float, required=True)
    parser.add_argument("--amplitude", type=float, default=2.0)
    args = parser.parse_args()
    result = decode(
        args.anchor,
        (args.pppp, args.ppmm, args.pmpm, args.pmmp),
        args.amplitude,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
