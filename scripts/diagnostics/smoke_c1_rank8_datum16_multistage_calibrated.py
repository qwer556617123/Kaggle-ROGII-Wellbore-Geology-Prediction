"""Synthetic smoke test for the datum16 12-dimensional joint solver."""
from __future__ import annotations

import numpy as np

from build_c1_rank8_datum16_multistage_calibrated import solve_parent_coefficients


fractions = np.asarray([0.07, 0.06, 0.05, 0.04], dtype=float)
basis = np.asarray(
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
    ]
)
expected = np.asarray([0.35, -0.62, 0.27])
offsets = expected @ basis
projections = -(basis * fractions[None, :]) @ offsets
actual = solve_parent_coefficients(fractions, projections, basis)
np.testing.assert_allclose(actual, expected, atol=1.0e-12)
print(
    {
        "fractions": fractions.tolist(),
        "projections": projections.tolist(),
        "coefficients": actual.tolist(),
        "offsets": offsets.tolist(),
    }
)
