"""Synthetic checks for the orientation-field HMM."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from orientation_field_hmm import (
    CONTACT_COLUMNS,
    OrientationFieldConfig,
    build_orientation_field,
    run_orientation_hmm,
)


def _write_training_well(
    root: Path,
    well: str,
    angle: float,
    offsets: np.ndarray,
    fault: float = 0.0,
) -> None:
    rows = 512
    md = np.arange(rows, dtype=float)
    vx = np.cos(angle)
    vy = np.sin(angle)
    x = 1000.0 + 250.0 * int(well[-1]) + vx * md
    y = 2000.0 + 180.0 * int(well[-1]) + vy * md
    surface = 0.018 * x - 0.012 * y - 9200.0
    if fault:
        surface[300:] += fault
    frame = pd.DataFrame({"MD": md, "X": x, "Y": y})
    for column, offset in zip(CONTACT_COLUMNS, offsets):
        frame[column] = surface + offset
    frame.to_csv(root / f"{well}__horizontal_well.csv", index=False)


def _target_case(fault: float = 0.0, missing: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rows = 420
    known_rows = 140
    md = np.arange(rows, dtype=float)
    angle = 0.72
    x = 1450.0 + np.cos(angle) * md
    y = 2350.0 + np.sin(angle) * md
    z = -9000.0 - 0.025 * md
    u = 0.018 * x - 0.012 * y + 1600.0
    if fault:
        u[known_rows + 100 :] += fault
    tvt = u - z
    tvt_input = np.where(np.arange(rows) < known_rows, tvt, np.nan)
    tw_tvt = np.arange(float(tvt.min() - 80.0), float(tvt.max() + 80.0), 0.5)
    tw_gr = 75.0 + 17.0 * np.sin(tw_tvt / 6.0) + 8.0 * np.cos(tw_tvt / 15.0)
    gr = 1.2 * np.interp(tvt, tw_tvt, tw_gr) + 3.0
    if missing:
        gr[known_rows + 30 : known_rows + 100] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    anchor_u = u[known_rows - 1] + 0.006 * (md - md[known_rows - 1])
    anchor = anchor_u - z
    anchor[:known_rows] = tvt[:known_rows]
    return horizontal, typewell, anchor


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def main() -> None:
    config = OrientationFieldConfig(
        neighbor_count=8,
        min_neighbors=4,
        distance_fallback_ft=10000.0,
        fault_radius_ft=10000.0,
        min_gr_coverage=0.50,
    )
    with TemporaryDirectory() as first_name, TemporaryDirectory() as second_name:
        first = Path(first_name)
        second = Path(second_name)
        base_offsets = np.asarray([0.0, -175.0, -203.0, -276.0, -311.0, -451.0])
        shifted_offsets = base_offsets + np.asarray([50.0, -20.0, 130.0, 7.0, -80.0, 300.0])
        for index in range(8):
            angle = 0.2 + 0.16 * index
            fault = 15.0 if index in {2, 5} else 0.0
            _write_training_well(first, f"train{index}", angle, base_offsets, fault)
            _write_training_well(second, f"train{index}", angle, shifted_offsets, fault)

        first_model = build_orientation_field(first, config)
        second_model = build_orientation_field(second, config)
        assert len(first_model.observations) == len(second_model.observations)
        np.testing.assert_allclose(
            first_model.observations[:, 2:],
            second_model.observations[:, 2:],
            atol=1.0e-10,
        )

        horizontal, typewell, anchor = _target_case()
        path = first_model.predict_path(horizontal, anchor, target_well="target", use_faults=False)
        eval_mask = horizontal["TVT_input"].isna().to_numpy()
        truth = horizontal.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        assert np.isfinite(path.field_path).all()
        assert path.metadata["fallback_centers"] == 0
        assert _rmse(truth, path.field_path[eval_mask]) < _rmse(truth, anchor[eval_mask])

        result = run_orientation_hmm(
            horizontal,
            typewell,
            anchor,
            first_model,
            target_well="target",
            use_faults=False,
            use_tool_response=True,
        )
        assert np.isfinite(result.posterior).all()
        assert np.max(np.abs(result.posterior[eval_mask] - anchor[eval_mask])) <= config.max_residual
        assert result.metadata["response_widths"]

        fault_horizontal, fault_typewell, fault_anchor = _target_case(fault=15.0, missing=True)
        fault_result = run_orientation_hmm(
            fault_horizontal,
            fault_typewell,
            fault_anchor,
            first_model,
            target_well="target",
            use_faults=True,
            use_tool_response=True,
        )
        assert np.isfinite(fault_result.posterior).all()
        assert fault_result.metadata["orientation"]["fault_hazard_max"] > config.base_fault_hazard
        assert fault_result.metadata["fault_probability_max"] > 0.0

        excluded = first_model.predict_gradient(
            first_model.observations[0, :2],
            first_model.observations[0, 2:4],
            excluded_well=str(first_model.wells[0]),
        )
        assert excluded["neighbors"] >= config.min_neighbors
        print(
            {
                "observations": int(len(first_model.observations)),
                "fault_observations": int(len(first_model.fault_observations)),
                "anchor_rmse": _rmse(truth, anchor[eval_mask]),
                "field_rmse": _rmse(truth, path.field_path[eval_mask]),
                "posterior_rmse": _rmse(truth, result.posterior[eval_mask]),
                "contact_offset_invariant": True,
                "same_id_excluded": True,
            }
        )


if __name__ == "__main__":
    main()
