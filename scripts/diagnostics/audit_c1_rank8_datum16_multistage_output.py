"""Audit one downloaded datum16 multistage Kaggle output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-base-sha", required=True)
    parser.add_argument("--expected-child-codes", default="1,2,3")
    args = parser.parse_args()

    audit_path = args.root / "datum16_multistage_audit.json"
    submission_path = args.root / "submission.csv"
    base_path = args.root / "submission_before_datum16_multistage.csv"
    for path in (audit_path, submission_path, base_path):
        if not path.exists():
            raise FileNotFoundError(path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    submission = pd.read_csv(submission_path, dtype={"id": "string"})
    base = pd.read_csv(base_path, dtype={"id": "string"})
    expected_codes = [int(value) for value in args.expected_child_codes.split(",")]

    assert list(submission.columns) == ["id", "tvt"]
    assert submission["id"].equals(base["id"])
    assert submission["id"].is_unique
    assert np.isfinite(submission["tvt"].to_numpy(dtype=float)).all()
    assert len(submission) == int(audit["rows"])
    assert audit["child_code_indices"] == expected_codes
    assert audit["fixed_public_ids_used"] is False
    assert audit["base_sha256"] == args.expected_base_sha
    assert sha256(base_path) == audit["base_sha256"]
    assert sha256(submission_path) == audit["final_sha256"]
    assert float(audit["max_abs_move"]) <= float(audit["offset_cap"]) + 1.0e-12

    print(
        json.dumps(
            {
                "rows": len(submission),
                "base_sha256": audit["base_sha256"],
                "final_sha256": audit["final_sha256"],
                "child_code_indices": audit["child_code_indices"],
                "gram_ranks": audit["gram_ranks"],
                "max_abs_move": audit["max_abs_move"],
                "mean_squared_move": audit["mean_squared_move"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
