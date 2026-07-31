"""Audit a symmetric pair of Kaggle orientation-field notebook outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_output(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    final_path = directory / "submission.csv"
    anchor_path = directory / "submission_before_orientation_field.csv"
    audit_path = directory / "orientation_field_audit.json"
    final = pd.read_csv(final_path, dtype={"id": "string"})[["id", "tvt"]]
    anchor = pd.read_csv(anchor_path, dtype={"id": "string"})[["id", "tvt"]]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if len(final) != len(anchor) or not final["id"].equals(anchor["id"]):
        raise RuntimeError(f"anchor/final ID mismatch in {directory}")
    for name, frame in (("anchor", anchor), ("final", final)):
        values = pd.to_numeric(frame["tvt"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} contains non-finite values in {directory}")
    if sha256(final_path) != audit["final_sha256"]:
        raise RuntimeError(f"final hash mismatch in {directory}")
    if sha256(anchor_path) != audit["anchor_sha256"]:
        raise RuntimeError(f"anchor hash mismatch in {directory}")
    return final, anchor, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--minus", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()

    plus, plus_anchor, plus_audit = load_output(args.plus)
    minus, minus_anchor, minus_audit = load_output(args.minus)
    if not plus["id"].equals(minus["id"]):
        raise RuntimeError("plus/minus ID order differs")
    if plus_audit["orientation_component_sha256"] != minus_audit["orientation_component_sha256"]:
        raise RuntimeError("orientation component hashes differ")
    if plus_audit["anchor_sha256"] != minus_audit["anchor_sha256"]:
        raise RuntimeError("anchor hashes differ")
    if float(plus_audit["field_weight"]) != -float(minus_audit["field_weight"]):
        raise RuntimeError("field weights are not symmetric")

    plus_anchor_values = plus_anchor["tvt"].to_numpy(dtype=float)
    minus_anchor_values = minus_anchor["tvt"].to_numpy(dtype=float)
    anchor_max_diff = float(np.max(np.abs(plus_anchor_values - minus_anchor_values)))
    if anchor_max_diff > args.tolerance:
        raise RuntimeError(f"anchor vectors differ by {anchor_max_diff}")

    plus_delta = plus["tvt"].to_numpy(dtype=float) - plus_anchor_values
    minus_delta = minus["tvt"].to_numpy(dtype=float) - minus_anchor_values
    antisymmetry_error = float(np.max(np.abs(plus_delta + minus_delta)))
    if antisymmetry_error > args.tolerance:
        raise RuntimeError(f"pair is not antisymmetric: {antisymmetry_error}")

    report = {
        "rows": int(len(plus)),
        "field_weights": [
            float(plus_audit["field_weight"]),
            float(minus_audit["field_weight"]),
        ],
        "anchor_sha256": plus_audit["anchor_sha256"],
        "orientation_component_sha256": plus_audit["orientation_component_sha256"],
        "plus_final_sha256": plus_audit["final_sha256"],
        "minus_final_sha256": minus_audit["final_sha256"],
        "anchor_max_abs_diff": anchor_max_diff,
        "antisymmetry_max_abs_error": antisymmetry_error,
        "direction_mean": float(np.mean(plus_delta)),
        "direction_mean_abs": float(np.mean(np.abs(plus_delta))),
        "direction_rms": float(np.sqrt(np.mean(plus_delta * plus_delta))),
        "direction_max_abs": float(np.max(np.abs(plus_delta))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
