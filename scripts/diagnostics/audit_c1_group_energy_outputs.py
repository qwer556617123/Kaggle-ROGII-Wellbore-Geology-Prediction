"""Audit completed single-bin C1 Kaggle notebook outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_output(directory: Path, expected_bin: int) -> dict[str, object]:
    audit_path = directory / "c1_heel_audit.json"
    base_path = directory / "submission_before_c1_heel.csv"
    final_path = directory / "submission.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    base = pd.read_csv(base_path, dtype={"id": "string"})[["id", "tvt"]]
    final = pd.read_csv(final_path, dtype={"id": "string"})[["id", "tvt"]]

    if len(base) != len(final) or not base["id"].equals(final["id"]):
        raise RuntimeError(f"base/final ID mismatch in {directory}")
    base_values = pd.to_numeric(base["tvt"], errors="coerce").to_numpy(float)
    final_values = pd.to_numeric(final["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(base_values).all() or not np.isfinite(final_values).all():
        raise RuntimeError(f"non-finite values in {directory}")
    if sha256(base_path) != audit["base_sha256"]:
        raise RuntimeError(f"base hash mismatch in {directory}")
    if sha256(final_path) != audit["final_sha256"]:
        raise RuntimeError(f"final hash mismatch in {directory}")

    alphas = tuple(float(value) for value in audit["bin_alphas"])
    expected = tuple(1.0 if index == expected_bin else 0.0 for index in range(4))
    if alphas != expected:
        raise RuntimeError(f"unexpected bin alphas in {directory}: {alphas}")
    if audit["probe_partition"] != "lexicographic_run_local_well_rank_mod_4":
        raise RuntimeError(f"unexpected partition in {directory}")
    if audit["fixed_public_ids_used"]:
        raise RuntimeError(f"fixed public ID flag set in {directory}")

    rows = audit["well_audits"]
    for row in rows:
        expected_alpha = 1.0 if int(row["probe_bin"]) == expected_bin else 0.0
        if float(row["effective_alpha"]) != expected_alpha:
            raise RuntimeError(f"effective alpha mismatch for {row['well']}")
    delta = final_values - base_values
    return {
        "directory": str(directory),
        "expected_bin": int(expected_bin),
        "rows": int(len(final)),
        "visible_wells": int(audit["wells"]),
        "visible_active_wells": int(
            sum(int(row["probe_bin"]) == expected_bin for row in rows)
        ),
        "base_sha256": audit["base_sha256"],
        "final_sha256": audit["final_sha256"],
        "mean_squared_move": float(np.mean(delta * delta)),
        "mean_abs_move": float(np.mean(np.abs(delta))),
        "max_abs_move": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin0", type=Path, required=True)
    parser.add_argument("--bin3", type=Path, required=True)
    args = parser.parse_args()

    reports = [audit_output(args.bin0, 0), audit_output(args.bin3, 3)]
    if reports[0]["base_sha256"] != reports[1]["base_sha256"]:
        raise RuntimeError("single-bin probes do not share the same HMM010 anchor")
    print(json.dumps({"outputs": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
