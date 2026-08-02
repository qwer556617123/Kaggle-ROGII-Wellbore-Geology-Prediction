"""Audit completed nested 16-bin datum Hadamard notebook outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_c1_rank8_datum16_codes import CODE_INDICES, H4, leaf_code


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, parent_code: int, child_code: int) -> dict[str, object]:
    slug = f"rogii-c1r8-d16-a{parent_code}b{child_code}"
    directory = root / f"{slug}-v1"
    datum = json.loads(
        (directory / "datum16_code_audit.json").read_text(encoding="utf-8")
    )
    final = json.loads(
        (directory / "stopdose_submission_audit.json").read_text(encoding="utf-8")
    )
    submission = pd.read_csv(directory / "submission.csv", dtype={"id": "string"})
    base = pd.read_csv(
        directory / "submission_before_datum16_code.csv", dtype={"id": "string"}
    )
    if len(submission) != len(base) or not submission["id"].equals(base["id"]):
        raise RuntimeError(f"base/final ID mismatch for {slug}")
    values = pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(float)
    base_values = pd.to_numeric(base["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise RuntimeError(f"non-finite values for {slug}")
    delta = values - base_values
    expected_leaf_code = leaf_code(parent_code, child_code)
    expected = {
        "parent_code_index": parent_code,
        "child_code_index": child_code,
        "parent_code": H4[parent_code],
        "child_code": H4[child_code],
        "leaf_code": expected_leaf_code,
    }
    if any(datum[key] != value for key, value in expected.items()):
        raise RuntimeError(f"unexpected code metadata for {slug}")
    if float(datum["amplitude"]) != 2.0:
        raise RuntimeError(f"unexpected amplitude for {slug}")
    if datum["partition"] != "nested_greedy_row_balanced_run_local_wells_4x4":
        raise RuntimeError(f"unexpected partition for {slug}")
    if datum["fixed_public_ids_used"] or not np.isclose(datum["mean_squared_move"], 4.0):
        raise RuntimeError(f"fixed-ID or energy audit failed for {slug}")
    if not np.allclose(np.abs(delta), 2.0, atol=2.0e-12):
        raise RuntimeError(f"datum16 move is not exactly +/-2 ft for {slug}")
    base_hash = sha256(directory / "submission_before_datum16_code.csv")
    final_hash = sha256(directory / "submission.csv")
    if datum["base_sha256"] != base_hash:
        raise RuntimeError(f"base hash mismatch for {slug}")
    if datum["final_sha256"] != final_hash or final["submission_sha256"] != final_hash:
        raise RuntimeError(f"final hash mismatch for {slug}")
    if not final["id_order_matches_sample"] or float(final["runtime_sec"]) >= 780.0:
        raise RuntimeError(f"contract/runtime gate failed for {slug}")
    return {
        "slug": slug,
        "parent_code_index": parent_code,
        "child_code_index": child_code,
        "leaf_code": expected_leaf_code,
        "rows": int(len(submission)),
        "runtime_sec": float(final["runtime_sec"]),
        "parent_bin_rows": [int(value) for value in datum["parent_bin_rows"]],
        "leaf_bin_rows": [int(value) for value in datum["leaf_bin_rows"]],
        "base_sha256": base_hash,
        "final_sha256": final_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("kaggle/outputs"))
    args = parser.parse_args()
    reports = [audit(args.root, parent, child) for parent, child in CODE_INDICES]
    if len({row["base_sha256"] for row in reports}) != 1:
        raise RuntimeError("datum16 probes do not share one rank-8 anchor")
    if len({tuple(row["parent_bin_rows"]) for row in reports}) != 1:
        raise RuntimeError("datum16 probes do not share one parent partition")
    if len({tuple(row["leaf_bin_rows"]) for row in reports}) != 1:
        raise RuntimeError("datum16 probes do not share one leaf partition")
    print(json.dumps({"outputs": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
