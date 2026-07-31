"""Smoke-test the contact-shape basis probe helper."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
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


def write_surface_train(path: Path, y: float, amp: float) -> None:
    x = np.arange(24, dtype=float) * 25.0
    shape = amp * np.sin(np.linspace(0.0, np.pi, len(x)))
    df = pd.DataFrame({
        "X": x,
        "Y": y,
        "ANCC": 1325.0 + 0.35 * shape,
        "ASTNU": 1230.0 + 0.45 * shape,
        "ASTNL": 1160.0 + 0.55 * shape,
        "EGFDU": 1120.0 + 0.65 * shape,
        "EGFDL": 1100.0 + shape,
        "BUDA": 965.0 + 0.25 * shape,
    })
    df.to_csv(path, index=False)


def write_test_well(path: Path, y: float, amp: float) -> None:
    n = 24
    x = np.arange(n, dtype=float) * 25.0
    shape = amp * np.sin(np.linspace(0.0, np.pi, n))
    tvt = 100.0 + shape
    df = pd.DataFrame({
        "X": x,
        "Y": y,
        "Z": np.full(n, 1000.0),
        "MD": x,
        "GR": 80.0 + 0.4 * tvt,
        "TVT_input": np.where(np.arange(n) < 12, tvt, np.nan),
    })
    df.to_csv(path, index=False)


def write_typewell(path: Path) -> None:
    tvt = np.linspace(80.0, 150.0, 96)
    gr = 80.0 + 0.4 * tvt
    pd.DataFrame({"TVT": tvt, "GR": gr}).to_csv(path, index=False)


def main() -> None:
    module = load_wrapper()

    centered = module.smooth_center_clip_basis(np.array([0.0, 15.0, 60.0, -30.0, -10.0]), 20.0)
    assert abs(float(centered.mean())) < 1e-9
    assert float(np.max(np.abs(centered))) <= 20.0 + 1e-9

    selected = module.select_contact_basis_candidate([
        {"well": "small", "score": 2.0},
        {"well": "big", "score": 5.0},
    ])
    assert selected["well"] == "big"

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "train").mkdir()
        (data_dir / "test").mkdir()
        write_surface_train(data_dir / "train" / "train_big__horizontal_well.csv", 0.0, 42.0)
        write_surface_train(data_dir / "train" / "train_small__horizontal_well.csv", 5000.0, 8.0)
        write_test_well(data_dir / "test" / "well_big__horizontal_well.csv", 0.0, 42.0)
        write_test_well(data_dir / "test" / "well_small__horizontal_well.csv", 5000.0, 8.0)
        write_typewell(data_dir / "test" / "well_big__typewell.csv")
        write_typewell(data_dir / "test" / "well_small__typewell.csv")

        ids = [f"well_big_{i}" for i in range(12, 24)] + [f"well_small_{i}" for i in range(12, 24)]
        submission = pd.DataFrame({"id": ids, "tvt": np.full(len(ids), 100.0)})

        unchanged, ops, summary, component = module.apply_contact_basis_probe(
            data_dir,
            submission,
            "off",
            0.25,
            30.0,
            data_dir / "off_component.csv",
        )
        assert ops == []
        assert component is None
        assert unchanged["tvt"].tolist() == submission["tvt"].tolist()
        assert summary["rule"] == "off"

        plus, plus_ops, plus_summary, plus_component = module.apply_contact_basis_probe(
            data_dir,
            submission,
            "max_contact_shape_gap",
            0.25,
            30.0,
            data_dir / "contact_basis_component.csv",
        )
        assert plus_ops[0]["well"] == "well_big"
        assert plus_ops[0]["rows"] == 12
        assert plus_ops[0]["basis_max_abs"] <= 30.0 + 1e-9
        assert abs(plus_ops[0]["basis_mean"]) < 1e-9
        assert plus_summary["selected"]["well"] == "well_big"
        assert plus_component is not None and plus_component.exists()

        changed = plus["tvt"].to_numpy(dtype=float) - submission["tvt"].to_numpy(dtype=float)
        assert np.any(np.abs(changed[:12]) > 0.0)
        assert np.allclose(changed[12:], 0.0)

        minus, minus_ops, _, _ = module.apply_contact_basis_probe(
            data_dir,
            submission,
            "max_contact_shape_gap",
            -0.25,
            30.0,
            data_dir / "contact_basis_component_minus.csv",
        )
        assert minus_ops[0]["well"] == plus_ops[0]["well"]
        assert np.allclose(minus["tvt"].to_numpy(dtype=float) - submission["tvt"].to_numpy(dtype=float), -changed)

        tvt_grid = np.linspace(50.0, 180.0, 256)
        tw = pd.DataFrame({
            "TVT": tvt_grid,
            "GR": 90.0 + 18.0 * np.sin(tvt_grid / 11.0) + 0.08 * tvt_grid,
        })
        path_true = np.linspace(70.0, 155.0, 64)
        gr_true = np.interp(path_true, tw["TVT"], tw["GR"])
        hw = pd.DataFrame({
            "MD": np.arange(64, dtype=float),
            "Z": np.zeros(64),
            "GR": 1.7 * gr_true + 13.0,
            "TVT_input": np.where(np.arange(64) < 32, path_true, np.nan),
        })
        path_shifted = path_true + 20.0
        true_score = module.gr_typewell_path_score(hw, tw, path_true)
        shifted_score = module.gr_typewell_path_score(hw, tw, path_shifted)
        assert true_score["gr_loss"] < shifted_score["gr_loss"]

    print("contact shape basis smoke passed")


if __name__ == "__main__":
    main()
