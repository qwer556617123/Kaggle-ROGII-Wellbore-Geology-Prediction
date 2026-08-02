"""Synthetic checks for nested 16-bin datum Hadamard decoding."""
from __future__ import annotations

import math

import numpy as np

from fit_c1_rank8_datum16 import H16, decode


def main() -> None:
    anchor = 6.5
    amplitude = 2.0
    projection = np.linspace(-0.16, 0.14, 16)
    scores = []
    for code in H16:
        score2 = anchor**2 + 2.0 * amplitude * float(code @ projection) + amplitude**2
        scores.append(math.sqrt(score2))
    result = decode(anchor, tuple(scores), amplitude)
    assert np.allclose(result["leaf_projection"], projection, atol=1.0e-12)
    assert np.allclose(result["equal_fraction_offset"], -projection / (1.0 / 16.0))
    print(result)


if __name__ == "__main__":
    main()
