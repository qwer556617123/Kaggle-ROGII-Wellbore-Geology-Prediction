"""Synthetic smoke for centered competitive-probe calibration."""
from __future__ import annotations

import numpy as np

from build_c1_datum_b12_centered_b3_final import solve_coefficients


directions = np.asarray(
    [
        [1.0, 1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0, -1.0],
    ]
)
gram = directions @ directions.T / directions.shape[1]
expected = np.asarray([0.35, -0.62, 0.27])
projections = -(gram @ expected)
actual = solve_coefficients(gram, projections)
np.testing.assert_allclose(actual, expected, atol=1.0e-12)
print({"gram": gram.tolist(), "projections": projections.tolist(), "coefficients": actual.tolist()})
