"""Decode one four-code child-contrast stage of nested datum tomography."""
from __future__ import annotations

import argparse
import json

import numpy as np

from build_c1_rank8_datum_calibrated import BIN_PROJECTIONS


H4 = np.asarray(
    [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
    dtype=float,
)


def decode_stage(
    anchor: float,
    scores: tuple[float, float, float, float],
    amplitude: float,
    child_code_index: int,
) -> dict[str, object]:
    if amplitude <= 0.0 or child_code_index not in (1, 2, 3):
        raise ValueError((amplitude, child_code_index))
    if len(scores) != 4:
        raise ValueError(f"expected four scores, got {len(scores)}")
    anchor2 = float(anchor) ** 2
    code_projection = np.asarray(
        [
            (float(score) ** 2 - anchor2 - float(amplitude) ** 2)
            / (2.0 * float(amplitude))
            for score in scores
        ],
        dtype=float,
    )
    parent_contrast_projection = H4.T @ code_projection / 4.0
    parent_projection = np.asarray(BIN_PROJECTIONS, dtype=float)
    return {
        "anchor_score": float(anchor),
        "amplitude": float(amplitude),
        "child_code_index": int(child_code_index),
        "child_code": "".join("+" if value > 0 else "-" for value in H4[child_code_index]),
        "scores": [float(value) for value in scores],
        "code_projection": code_projection.tolist(),
        "parent_projection": parent_projection.tolist(),
        "parent_contrast_projection": parent_contrast_projection.tolist(),
        "deployment": (
            "For each parent, solve G*[datum,contrast] = -[parent_projection,"
            "parent_contrast_projection] with run-local row fractions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=float, required=True)
    parser.add_argument(
        "--scores",
        required=True,
        help="Four comma-separated scores ordered by parent-code index a=0,1,2,3.",
    )
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--child-code-index", type=int, default=1)
    args = parser.parse_args()
    scores = tuple(float(value.strip()) for value in args.scores.split(","))
    print(
        json.dumps(
            decode_stage(
                args.anchor,
                scores,
                args.amplitude,
                args.child_code_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
