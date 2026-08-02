"""Synthetic exact-recovery smoke test for a datum16 child stage."""
from __future__ import annotations

import numpy as np

from fit_c1_rank8_datum16_stage import H4, decode_stage


def main() -> None:
    anchor = 6.5
    amplitude = 2.0
    expected = np.asarray((-0.31, 0.12, -0.07, 0.23), dtype=float)
    code_projection = H4 @ expected
    score2 = anchor * anchor + amplitude * amplitude + 2.0 * amplitude * code_projection
    scores = tuple(np.sqrt(score2).tolist())
    result = decode_stage(anchor, scores, amplitude, child_code_index=1)
    recovered = np.asarray(result["parent_contrast_projection"], dtype=float)
    if not np.allclose(recovered, expected, atol=1.0e-12):
        raise RuntimeError((recovered, expected))
    print(result)


if __name__ == "__main__":
    main()
