"""Synthetic recovery test for four-bin C1 energy calibration."""
from __future__ import annotations

import math

import numpy as np

from fit_c1_group_energy_calibration import fit_group_calibration


def score(base2: float, projection: np.ndarray, energy: np.ndarray, alpha: np.ndarray) -> float:
    value = base2 + float(np.sum(2.0 * alpha * projection + alpha**2 * energy))
    return math.sqrt(value)


def main() -> None:
    base = 6.696
    base2 = base**2
    projection = np.asarray([-1.2, 0.5, -0.2, -2.4], dtype=float)
    energy = np.asarray([1.4, 2.1, 0.9, 5.0], dtype=float)
    codes = np.asarray(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=float,
    )
    code_scores = [score(base2, projection, energy, row) for row in codes]
    minus_score = score(base2, projection, energy, -np.ones(4))
    bin0_score = score(base2, projection, energy, np.asarray([1, 0, 0, 0]))
    bin3_score = score(base2, projection, energy, np.asarray([0, 0, 0, 1]))
    selective = np.asarray([0.35, 0.0, 0.35, 0.35])
    selective_score = score(base2, projection, energy, selective)

    result = fit_group_calibration(
        base,
        code_scores[0],
        minus_score,
        tuple(code_scores[1:]),
        bin0_score,
        bin3_score,
        selective_score,
    )
    assert np.allclose(result["bin_projection"], projection, atol=1.0e-10)
    assert np.allclose(result["bin_energy"], energy, atol=1.0e-10)
    assert np.allclose(result["raw_bin_alpha"], -projection / energy, atol=1.0e-10)
    print(result)


if __name__ == "__main__":
    main()
