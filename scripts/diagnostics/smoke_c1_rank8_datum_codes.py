"""Synthetic recovery check for row-balanced datum Hadamard codes."""
from __future__ import annotations

import math

import numpy as np

from fit_c1_rank8_datum_codes import HADAMARD_4, decode


def main() -> None:
    anchor = 6.5
    amplitude = 2.0
    projection = np.asarray([-0.4, 0.2, -0.1, 0.3])
    scores = []
    for code in HADAMARD_4:
        score2 = anchor**2 + 2.0 * amplitude * float(code @ projection) + amplitude**2
        scores.append(math.sqrt(score2))
    result = decode(anchor, tuple(scores), amplitude)
    assert np.allclose(result["bin_projection"], projection, atol=1.0e-12)
    assert np.allclose(result["equal_fraction_alpha"], -projection / 0.25)
    print(result)


if __name__ == "__main__":
    main()
