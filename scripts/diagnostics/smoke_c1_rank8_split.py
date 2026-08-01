"""Synthetic checks for the rank-mod-8 C1 split inversion."""
from __future__ import annotations

import math

from fit_c1_rank8_split import fit_parent_split


def main() -> None:
    base = 6.696
    p = (-1.4, -1.0)
    q = (1.7, 2.5)
    parent_p = sum(p)
    parent_q = sum(q)
    split2 = base**2 + 2.0 * (p[0] - p[1]) + parent_q
    child2 = base**2 + 2.0 * p[0] + q[0]
    result = fit_parent_split(
        base,
        parent_p,
        parent_q,
        math.sqrt(split2),
        math.sqrt(child2),
    )
    for actual, expected in zip(result["child_projection"], p):
        assert abs(actual - expected) < 1.0e-10
    for actual, expected in zip(result["child_energy"], q):
        assert abs(actual - expected) < 1.0e-10
    assert result["incremental_gain_score2"] >= -1.0e-12
    print(result)


if __name__ == "__main__":
    main()
