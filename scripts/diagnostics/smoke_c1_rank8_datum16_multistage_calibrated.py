"""Synthetic smoke test for the datum16 12-dimensional joint solver."""
from __future__ import annotations

import numpy as np

from build_c1_rank8_datum16_codes import H4
from build_c1_rank8_datum16_multistage_calibrated import (
    decode_partial_contrast,
    solve_parent_coefficients,
)


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

anchor = 6.5
amplitude = 2.0
parent_contrast = np.asarray([0.31, -0.27, 0.44, -0.19])
hadamard = np.asarray(
    [[1.0 if char == "+" else -1.0 for char in row] for row in H4]
)
code_projection = hadamard @ parent_contrast
selected = (0, 1, 2)
scores = tuple(
    float(np.sqrt(anchor * anchor + amplitude * amplitude + 2.0 * amplitude * code_projection[index]))
    for index in selected
)
partial = np.asarray(
    decode_partial_contrast(anchor, scores, amplitude, selected)
)
expected_partial = hadamard[list(selected)].T @ code_projection[list(selected)] / 4.0
np.testing.assert_allclose(partial, expected_partial, atol=1.0e-12)
np.testing.assert_allclose(hadamard[3] @ partial, 0.0, atol=1.0e-12)
print(
    {
        "partial_parent_indices": selected,
        "partial_contrast": partial.tolist(),
        "omitted_code_projection": float(hadamard[3] @ partial),
    }
)
