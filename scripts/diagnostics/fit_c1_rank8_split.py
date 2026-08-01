"""Recover two rank-mod-8 C1 child responses from a parent-bin probe pair."""
from __future__ import annotations

import argparse
import json
import math


def fit_parent_split(
    base_score: float,
    parent_projection: float,
    parent_energy: float,
    split_score: float,
    child_score: float,
) -> dict[str, object]:
    base2 = float(base_score) ** 2
    split_delta = float(split_score) ** 2 - base2
    projection_contrast = (split_delta - float(parent_energy)) / 2.0
    p0 = (float(parent_projection) + projection_contrast) / 2.0
    p1 = float(parent_projection) - p0
    q0 = float(child_score) ** 2 - base2 - 2.0 * p0
    q1 = float(parent_energy) - q0
    if q0 <= 0.0 or q1 <= 0.0:
        raise ValueError(f"recovered non-positive child energy: {(q0, q1)}")
    a0 = -p0 / q0
    a1 = -p1 / q1
    parent_gain = float(parent_projection) ** 2 / float(parent_energy)
    child_gain = p0**2 / q0 + p1**2 / q1
    return {
        "base_score": float(base_score),
        "parent_projection": float(parent_projection),
        "parent_energy": float(parent_energy),
        "split_score": float(split_score),
        "child_score": float(child_score),
        "child_projection": [p0, p1],
        "child_energy": [q0, q1],
        "child_alpha": [a0, a1],
        "parent_gain_score2": parent_gain,
        "child_gain_score2": child_gain,
        "incremental_gain_score2": child_gain - parent_gain,
        "split_parent_only_predicted_score": math.sqrt(
            max(base2 - child_gain, 0.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=float, default=6.696)
    parser.add_argument("--parent-projection", type=float, required=True)
    parser.add_argument("--parent-energy", type=float, required=True)
    parser.add_argument("--split", type=float, required=True)
    parser.add_argument("--child", type=float, required=True)
    args = parser.parse_args()
    result = fit_parent_split(
        args.base,
        args.parent_projection,
        args.parent_energy,
        args.split,
        args.child,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
