"""Audit the four completed rank-mod-8 C1 tomography outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "rogii-hmm010-c1r8-b1-split": (0, 1, 0, 0, 0, -1, 0, 0),
    "rogii-hmm010-c1r8-b1-child": (0, 1, 0, 0, 0, 0, 0, 0),
    "rogii-hmm010-c1r8-b3-split": (0, 0, 0, 1, 0, 0, 0, -1),
    "rogii-hmm010-c1r8-b3-child": (0, 0, 0, 1, 0, 0, 0, 0),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, slug: str, expected: tuple[int, ...]) -> dict[str, object]:
    directory = root / f"{slug}-v1"
    c1 = json.loads((directory / "c1_heel_audit.json").read_text(encoding="utf-8"))
    final = json.loads(
        (directory / "stopdose_submission_audit.json").read_text(encoding="utf-8")
    )
    submission = pd.read_csv(directory / "submission.csv", dtype={"id": "string"})
    base = pd.read_csv(
        directory / "submission_before_c1_heel.csv", dtype={"id": "string"}
    )
    if len(submission) != len(base) or not submission["id"].equals(base["id"]):
        raise RuntimeError(f"base/final ID mismatch for {slug}")
    values = pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite submission for {slug}")
    if tuple(c1["bin_alphas"]) != expected:
        raise RuntimeError(f"unexpected bin alphas for {slug}: {c1['bin_alphas']}")
    if c1["probe_partition"] != "lexicographic_run_local_well_rank_mod_8":
        raise RuntimeError(f"unexpected partition for {slug}")
    if c1["fixed_public_ids_used"]:
        raise RuntimeError(f"fixed public ID flag set for {slug}")
    base_hash = sha256(directory / "submission_before_c1_heel.csv")
    final_hash = sha256(directory / "submission.csv")
    if c1["base_sha256"] != base_hash:
        raise RuntimeError(f"base hash mismatch for {slug}")
    if c1["final_sha256"] != final_hash or final["submission_sha256"] != final_hash:
        raise RuntimeError(f"final hash mismatch for {slug}")
    if not final["id_order_matches_sample"] or float(final["runtime_sec"]) >= 780.0:
        raise RuntimeError(f"contract/runtime gate failed for {slug}")
    return {
        "slug": slug,
        "rows": int(len(submission)),
        "runtime_sec": float(final["runtime_sec"]),
        "base_sha256": base_hash,
        "final_sha256": final_hash,
        "visible_mean_squared_move": float(c1["mean_squared_move"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("kaggle/outputs"))
    args = parser.parse_args()
    reports = [audit(args.root, slug, alphas) for slug, alphas in EXPECTED.items()]
    if len({row["base_sha256"] for row in reports}) != 1:
        raise RuntimeError("rank-8 probes do not share one pre-C1 anchor")
    print(json.dumps({"outputs": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
