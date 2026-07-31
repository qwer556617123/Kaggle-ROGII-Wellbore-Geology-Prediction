"""Build single-bin C1 probes for hidden direction-energy calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from build_hmm010_structural_tomography_batch import build_c1_bin_calibrated


@dataclass(frozen=True)
class Variant:
    slug: str
    bin_index: int
    bin_alphas: tuple[float, float, float, float]


VARIANTS = (
    Variant("rogii-hmm010-c1bin0-p1", 0, (1.0, 0.0, 0.0, 0.0)),
    Variant("rogii-hmm010-c1bin3-p1", 3, (0.0, 0.0, 0.0, 1.0)),
)


def main() -> None:
    for variant in VARIANTS:
        output = build_c1_bin_calibrated(variant.bin_alphas, variant.slug)
        print(f"built {output}")


if __name__ == "__main__":
    main()
