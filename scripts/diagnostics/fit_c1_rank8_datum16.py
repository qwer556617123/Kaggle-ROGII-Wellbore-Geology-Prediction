"""Decode the nested 16-bin datum Hadamard response."""
from __future__ import annotations

import argparse
import json

import numpy as np


H4 = np.asarray(
    [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
    dtype=float,
)
H16 = np.kron(H4, H4)


def decode(anchor: float, scores: tuple[float, ...], amplitude: float) -> dict[str, object]:
    if len(scores) != 16 or amplitude <= 0.0:
        raise ValueError((len(scores), amplitude))
    anchor2 = float(anchor) ** 2
    code_projection = np.asarray(
        [
            (float(score) ** 2 - anchor2 - float(amplitude) ** 2)
            / (2.0 * float(amplitude))
            for score in scores
        ],
        dtype=float,
    )
    leaf_projection = H16.T @ code_projection / 16.0
    equal_fraction = 1.0 / 16.0
    equal_fraction_offset = -leaf_projection / equal_fraction
    predicted2 = anchor2 - float(
        np.sum(leaf_projection * leaf_projection / equal_fraction)
    )
    return {
        "anchor_score": float(anchor),
        "amplitude": float(amplitude),
        "code_scores": [float(value) for value in scores],
        "code_projection": code_projection.tolist(),
        "leaf_projection": leaf_projection.tolist(),
        "equal_fraction_offset": equal_fraction_offset.tolist(),
        "equal_fraction_predicted_score": float(np.sqrt(max(predicted2, 0.0))),
        "note": "Final notebook should divide each leaf projection by its run-local row fraction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=float, required=True)
    parser.add_argument(
        "--scores",
        required=True,
        help="16 comma-separated scores in row-major (parent-code, child-code) order",
    )
    parser.add_argument("--amplitude", type=float, default=2.0)
    args = parser.parse_args()
    scores = tuple(float(value.strip()) for value in args.scores.split(","))
    print(json.dumps(decode(args.anchor, scores, args.amplitude), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
