"""Audit the LB-decoded four-group datum calibration notebook output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_c1_rank8_datum_calibrated import BIN_PROJECTIONS, OFFSET_CAP, SLUG


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path) -> dict[str, object]:
    datum = json.loads(
        (directory / "datum_calibration_audit.json").read_text(encoding="utf-8")
    )
    final = json.loads(
        (directory / "stopdose_submission_audit.json").read_text(encoding="utf-8")
    )
    submission = pd.read_csv(directory / "submission.csv", dtype={"id": "string"})
    base = pd.read_csv(
        directory / "submission_before_datum_calibration.csv", dtype={"id": "string"}
    )
    if len(submission) != len(base) or not submission["id"].equals(base["id"]):
        raise RuntimeError("base/final ID mismatch")
    values = pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(float)
    base_values = pd.to_numeric(base["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise RuntimeError("non-finite base or final values")
    fractions = np.asarray(datum["bin_row_fractions"], dtype=float)
    projections = np.asarray(datum["bin_projection"], dtype=float)
    expected_raw = np.divide(
        -np.asarray(BIN_PROJECTIONS, dtype=float),
        fractions,
        out=np.zeros(4, dtype=float),
        where=fractions > 0.0,
    )
    expected = np.clip(expected_raw, -OFFSET_CAP, OFFSET_CAP)
    if not np.allclose(projections, BIN_PROJECTIONS, atol=1.0e-12):
        raise RuntimeError("unexpected LB projections")
    if not np.allclose(datum["raw_bin_offsets"], expected_raw, atol=1.0e-12):
        raise RuntimeError("raw offset calculation mismatch")
    if not np.allclose(datum["deployed_bin_offsets"], expected, atol=1.0e-12):
        raise RuntimeError("deployed offset calculation mismatch")
    delta = values - base_values
    if not np.allclose(np.sort(np.unique(np.round(delta, 10))), np.sort(np.unique(np.round(expected, 10)))):
        raise RuntimeError("submission deltas do not match deployed offsets")
    base_hash = sha256(directory / "submission_before_datum_calibration.csv")
    final_hash = sha256(directory / "submission.csv")
    if datum["base_sha256"] != base_hash:
        raise RuntimeError("base hash mismatch")
    if datum["final_sha256"] != final_hash or final["submission_sha256"] != final_hash:
        raise RuntimeError("final hash mismatch")
    if datum["fixed_public_ids_used"] or datum["partition"] != "greedy_row_balanced_run_local_wells_4":
        raise RuntimeError("partition or fixed-ID contract failed")
    if not final["id_order_matches_sample"] or float(final["runtime_sec"]) >= 780.0:
        raise RuntimeError("sample alignment or runtime gate failed")
    return {
        "slug": SLUG,
        "rows": int(len(submission)),
        "runtime_sec": float(final["runtime_sec"]),
        "bin_rows": [int(value) for value in datum["bin_rows"]],
        "deployed_bin_offsets": expected.tolist(),
        "base_sha256": base_hash,
        "final_sha256": final_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("kaggle/outputs") / f"{SLUG}-v1",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
