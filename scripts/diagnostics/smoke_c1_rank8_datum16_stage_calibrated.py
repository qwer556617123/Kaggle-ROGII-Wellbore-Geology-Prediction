"""Synthetic Gram-solve smoke test for staged datum16 deployment."""
from __future__ import annotations

import numpy as np

from build_c1_rank8_datum16_stage_calibrated import solve_parent_coefficients


def main() -> None:
    fractions = np.asarray((0.07, 0.06, 0.05, 0.04), dtype=float)
    signs = np.asarray((1.0, 1.0, -1.0, -1.0), dtype=float)
    expected = np.asarray((0.35, -0.62), dtype=float)
    q = float(fractions.sum())
    cross = float(fractions @ signs)
    gram = np.asarray(((q, cross), (cross, q)), dtype=float)
    projections = -(gram @ expected)
    actual = solve_parent_coefficients(
        fractions,
        float(projections[0]),
        float(projections[1]),
        signs,
    )
    if not np.allclose(actual, expected, atol=1.0e-12):
        raise RuntimeError((actual, expected))
    print({"fractions": fractions.tolist(), "projections": projections.tolist(), "coefficients": actual.tolist()})


if __name__ == "__main__":
    main()
