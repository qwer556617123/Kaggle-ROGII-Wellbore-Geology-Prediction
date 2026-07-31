"""Recover per-well HMM residual projections from three signed LB probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SIGN_MATRIX = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
    ],
    dtype=float,
)


def read_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "base_sha256",
        "hmm_sha256",
        "magnitude",
        "signs_by_sorted_well",
        "well_stats",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")
    return payload


def predicted_score(
    anchor_score: float,
    projections: np.ndarray,
    norms: np.ndarray,
    weights: np.ndarray,
) -> float:
    mse = (
        anchor_score**2
        + 2.0 * float(np.dot(weights, projections))
        + float(np.dot(norms, weights**2))
    )
    return float(np.sqrt(max(mse, 0.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppm-audit", type=Path, required=True)
    parser.add_argument("--pmp-audit", type=Path, required=True)
    parser.add_argument("--ppm-score", type=float, required=True)
    parser.add_argument("--pmp-score", type=float, required=True)
    parser.add_argument("--anchor-score", type=float, default=7.053)
    parser.add_argument("--all-plus-score", type=float, default=6.909)
    parser.add_argument("--min-weight", type=float, default=-0.50)
    parser.add_argument("--max-weight", type=float, default=0.75)
    parser.add_argument("--round-step", type=float, default=0.025)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/hmm_three_well_tomography.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ppm = read_audit(args.ppm_audit)
    pmp = read_audit(args.pmp_audit)
    if ppm["base_sha256"] != pmp["base_sha256"]:
        raise RuntimeError("probe base hashes differ")
    if ppm["hmm_sha256"] != pmp["hmm_sha256"]:
        raise RuntimeError("probe HMM hashes differ")
    expected_signs = ((1.0, 1.0, -1.0), (1.0, -1.0, 1.0))
    actual_signs = (
        tuple(float(value) for value in ppm["signs_by_sorted_well"]),
        tuple(float(value) for value in pmp["signs_by_sorted_well"]),
    )
    if actual_signs != expected_signs:
        raise RuntimeError(f"unexpected probe signs: {actual_signs}")
    magnitude = float(ppm["magnitude"])
    if not np.isclose(magnitude, float(pmp["magnitude"])):
        raise RuntimeError("probe magnitudes differ")

    ppm_stats = sorted(ppm["well_stats"], key=lambda item: item["well_rank"])
    pmp_stats = sorted(pmp["well_stats"], key=lambda item: item["well_rank"])
    wells = [str(item["well_id"]) for item in ppm_stats]
    if wells != [str(item["well_id"]) for item in pmp_stats]:
        raise RuntimeError("probe sorted wells differ")
    norms = np.asarray(
        [float(item["direction_sse_per_total_row"]) for item in ppm_stats],
        dtype=float,
    )
    pmp_norms = np.asarray(
        [float(item["direction_sse_per_total_row"]) for item in pmp_stats],
        dtype=float,
    )
    if not np.allclose(norms, pmp_norms, rtol=0.0, atol=1e-12):
        raise RuntimeError("probe direction norms differ")
    if len(wells) != 3 or np.any(norms <= 0.0):
        raise RuntimeError("tomography requires exactly three nonzero well directions")

    scores = np.asarray(
        [args.all_plus_score, args.ppm_score, args.pmp_score], dtype=float
    )
    common_quadratic = magnitude**2 * float(np.sum(norms))
    signed_projection_sums = (
        scores**2 - args.anchor_score**2 - common_quadratic
    ) / (2.0 * magnitude)
    projections = np.linalg.solve(SIGN_MATRIX, signed_projection_sums)
    raw_weights = -projections / norms
    clamped_weights = np.clip(raw_weights, args.min_weight, args.max_weight)
    rounded_weights = (
        np.round(clamped_weights / args.round_step) * args.round_step
    )
    rounded_weights = np.clip(rounded_weights, args.min_weight, args.max_weight)

    result = {
        "anchor_score": float(args.anchor_score),
        "all_plus_score": float(args.all_plus_score),
        "ppm_score": float(args.ppm_score),
        "pmp_score": float(args.pmp_score),
        "magnitude": magnitude,
        "base_sha256": ppm["base_sha256"],
        "hmm_sha256": ppm["hmm_sha256"],
        "common_quadratic": common_quadratic,
        "wells": [
            {
                "well_rank": index,
                "well_id": well,
                "direction_norm2_per_total_row": float(norms[index]),
                "residual_projection": float(projections[index]),
                "raw_optimal_weight": float(raw_weights[index]),
                "clamped_weight": float(clamped_weights[index]),
                "rounded_weight": float(rounded_weights[index]),
            }
            for index, well in enumerate(wells)
        ],
        "predicted_score_raw": predicted_score(
            args.anchor_score, projections, norms, raw_weights
        ),
        "predicted_score_clamped": predicted_score(
            args.anchor_score, projections, norms, clamped_weights
        ),
        "predicted_score_rounded": predicted_score(
            args.anchor_score, projections, norms, rounded_weights
        ),
        "clamp": [float(args.min_weight), float(args.max_weight)],
        "round_step": float(args.round_step),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
