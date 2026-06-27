"""Smoke-test the rerun-safe final dynamic offset helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "kaggle" / "rogii-pf-artifact-blend" / "rogii-pf-artifact-blend.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("rogii_pf_artifact_blend", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load wrapper module from {WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_wrapper()
    submission = pd.DataFrame({
        "id": ["well_a_0", "well_a_1", "well_b_0", "well_b_1", "well_c_0"],
        "tvt": [100.0, 101.0, 200.0, 201.0, 300.0],
    })
    pf = pd.DataFrame({
        "id": submission["id"],
        "tvt": [110.0, 111.0, 206.0, 208.0, 301.0],
    })
    artifact = pd.DataFrame({
        "id": submission["id"],
        "tvt": [100.0, 101.0, 200.0, 200.0, 300.0],
    })
    well_ids = submission["id"].str.rsplit("_", n=1).str[0]

    stats, payload = module.component_gap_stats(well_ids, pf, artifact)
    assert payload["well_a"]["mean_abs_gap"] == 10.0
    assert payload["well_b"]["mean_abs_gap"] == 7.0

    unchanged, ops = module.apply_dynamic_final_offset(
        submission,
        well_ids,
        stats,
        "off",
        10.0,
    )
    assert ops == []
    assert unchanged["tvt"].tolist() == submission["tvt"].tolist()

    plus, plus_ops = module.apply_dynamic_final_offset(
        submission,
        well_ids,
        stats,
        "max_component_gap",
        10.0,
    )
    assert plus_ops[0]["well"] == "well_a"
    assert plus_ops[0]["rows"] == 2
    assert plus["tvt"].tolist() == [110.0, 111.0, 200.0, 201.0, 300.0]

    minus, minus_ops = module.apply_dynamic_final_offset(
        submission,
        well_ids,
        stats,
        "max_component_gap",
        -10.0,
    )
    assert minus_ops[0]["well"] == "well_a"
    assert minus_ops[0]["value"] == -10.0
    assert minus["tvt"].tolist() == [90.0, 91.0, 200.0, 201.0, 300.0]

    print("dynamic offset smoke passed")


if __name__ == "__main__":
    main()
