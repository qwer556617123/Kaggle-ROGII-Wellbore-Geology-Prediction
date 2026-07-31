"""Build horizontal-log reference curves in a target well's stratigraphic frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceChoice:
    name: str
    reference: pd.DataFrame
    prefix_loss: float
    prefix_rows: int
    neighbors: tuple[str, ...]
    coverage: float


def formation_anchors(typewell: pd.DataFrame) -> dict[str, float]:
    geology = typewell.dropna(subset=["Geology"]).copy()
    if geology.empty:
        return {}
    geology["Geology"] = geology["Geology"].astype(str)
    geology["TVT"] = pd.to_numeric(geology["TVT"], errors="coerce")
    geology = geology.dropna(subset=["TVT"])
    return geology.groupby("Geology", sort=False)["TVT"].min().astype(float).to_dict()


def _linear_extrapolated_interp(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    order = np.argsort(source, kind="stable")
    source = np.asarray(source, dtype=float)[order]
    target = np.asarray(target, dtype=float)[order]
    keep = np.concatenate([[True], np.diff(source) > 1e-6])
    source = source[keep]
    target = target[keep]
    if len(source) < 2:
        raise ValueError("At least two distinct formation anchors are required")
    out = np.interp(values, source, target)
    left_slope = (target[1] - target[0]) / (source[1] - source[0])
    right_slope = (target[-1] - target[-2]) / (source[-1] - source[-2])
    left = values < source[0]
    right = values > source[-1]
    out[left] = target[0] + left_slope * (values[left] - source[0])
    out[right] = target[-1] + right_slope * (values[right] - source[-1])
    return out


def warp_tvt_to_target(
    tvt: np.ndarray,
    source_typewell: pd.DataFrame,
    target_typewell: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    source_anchors = formation_anchors(source_typewell)
    target_anchors = formation_anchors(target_typewell)
    common = sorted(set(source_anchors).intersection(target_anchors))
    if len(common) < 2:
        source_mid = float(pd.to_numeric(source_typewell["TVT"], errors="coerce").median())
        target_mid = float(pd.to_numeric(target_typewell["TVT"], errors="coerce").median())
        return np.asarray(tvt, dtype=float) + target_mid - source_mid, tuple(common)
    source = np.asarray([source_anchors[name] for name in common], dtype=float)
    target = np.asarray([target_anchors[name] for name in common], dtype=float)
    return _linear_extrapolated_interp(np.asarray(tvt, dtype=float), source, target), tuple(common)


def robust_affine_prefix_loss(
    horizontal: pd.DataFrame,
    reference: pd.DataFrame,
    tail_rows: int = 1200,
) -> tuple[float, int, float, float]:
    known = horizontal.dropna(subset=["TVT_input", "GR"]).tail(tail_rows)
    ref = reference.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(known) < 80 or len(ref) < 20:
        return float("inf"), int(len(known)), 1.0, 0.0
    x = np.interp(
        known["TVT_input"].to_numpy(dtype=float),
        ref["TVT"].to_numpy(dtype=float),
        ref["GR"].to_numpy(dtype=float),
    )
    y = known["GR"].to_numpy(dtype=float)
    design = np.column_stack([x, np.ones(len(x), dtype=float)])
    weights = np.ones(len(x), dtype=float)
    coef = np.asarray([1.0, 0.0], dtype=float)
    for _ in range(8):
        lhs = design.T @ (weights[:, None] * design) + np.diag([1e-5, 1e-7])
        rhs = design.T @ (weights * y)
        coef = np.linalg.solve(lhs, rhs)
        coef[0] = np.clip(coef[0], 0.25, 4.0)
        residual = y - design @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 5.0)
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
    residual = y - design @ coef
    loss = float(np.sqrt(np.average(np.square(residual), weights=weights)))
    return loss, int(len(known)), float(coef[0]), float(coef[1])


class HorizontalReferenceBank:
    """Lazy cache of train horizontal GR curves and paired formation frameworks."""

    def __init__(self, data_dir: Path, grid_step: float = 0.5) -> None:
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.grid_step = float(grid_step)
        self.catalog = self._build_catalog()
        self._cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _build_catalog(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.train_dir.glob("*__horizontal_well.csv")):
            frame = pd.read_csv(path, usecols=["X", "Y"], nrows=1)
            if frame.empty:
                continue
            rows.append(
                {
                    "well": path.name.split("__", 1)[0],
                    "x": float(frame.iloc[0]["X"]),
                    "y": float(frame.iloc[0]["Y"]),
                }
            )
        if not rows:
            raise RuntimeError("No train wells were found for the horizontal reference bank")
        return pd.DataFrame(rows).sort_values("well").reset_index(drop=True)

    def _load_train_well(self, well: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if well not in self._cache:
            horizontal = pd.read_csv(
                self.train_dir / f"{well}__horizontal_well.csv",
                usecols=["TVT", "GR"],
            ).dropna(subset=["TVT", "GR"])
            typewell = pd.read_csv(
                self.train_dir / f"{well}__typewell.csv",
                usecols=["TVT", "GR", "Geology"],
            )
            self._cache[well] = (horizontal, typewell)
        return self._cache[well]

    def nearest_wells(
        self,
        target_well: str,
        x: float,
        y: float,
        count: int,
    ) -> tuple[str, ...]:
        candidates = self.catalog[self.catalog["well"].astype(str) != str(target_well)].copy()
        candidates["distance2"] = np.square(candidates["x"] - float(x))
        candidates["distance2"] += np.square(candidates["y"] - float(y))
        return tuple(candidates.nsmallest(count, "distance2")["well"].astype(str))

    def _neighbor_curve(
        self,
        well: str,
        target_typewell: pd.DataFrame,
        target_grid: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        horizontal, source_typewell = self._load_train_well(well)
        warped, common = warp_tvt_to_target(
            horizontal["TVT"].to_numpy(dtype=float), source_typewell, target_typewell
        )
        gr = horizontal["GR"].to_numpy(dtype=float)
        bins = np.rint((warped - target_grid[0]) / self.grid_step).astype(int)
        valid = (bins >= 0) & (bins < len(target_grid)) & np.isfinite(gr)
        if valid.sum() < 20:
            return np.full(len(target_grid), np.nan), len(common)
        grouped = pd.DataFrame({"bin": bins[valid], "gr": gr[valid]}).groupby("bin")["gr"].median()
        curve = np.full(len(target_grid), np.nan, dtype=float)
        curve[grouped.index.to_numpy(dtype=int)] = grouped.to_numpy(dtype=float)
        curve = pd.Series(curve).interpolate(limit=24, limit_direction="both").to_numpy(dtype=float)
        return curve, len(common)

    def candidate_references(
        self,
        target_well: str,
        horizontal: pd.DataFrame,
        target_typewell: pd.DataFrame,
        neighbor_counts: tuple[int, ...] = (4, 8, 16),
    ) -> list[ReferenceChoice]:
        target_tw = target_typewell[["TVT", "GR", "Geology"]].copy().sort_values("TVT")
        target_grid = target_tw["TVT"].to_numpy(dtype=float)
        own_gr = pd.to_numeric(target_tw["GR"], errors="coerce").interpolate(
            limit_direction="both"
        ).to_numpy(dtype=float)
        x = float(pd.to_numeric(horizontal["X"], errors="coerce").iloc[0])
        y = float(pd.to_numeric(horizontal["Y"], errors="coerce").iloc[0])
        nearest = self.nearest_wells(target_well, x, y, max(neighbor_counts))
        curves = []
        anchor_counts = []
        for well in nearest:
            curve, anchors = self._neighbor_curve(well, target_tw, target_grid)
            curves.append(curve)
            anchor_counts.append(anchors)
        matrix = np.stack(curves, axis=0)
        choices = []

        own = pd.DataFrame({"TVT": target_grid, "GR": own_gr})
        loss, rows, _, _ = robust_affine_prefix_loss(horizontal, own)
        choices.append(ReferenceChoice("own", own, loss, rows, tuple(), 1.0))

        for count in neighbor_counts:
            selected = matrix[:count]
            with np.errstate(all="ignore"):
                composite = np.nanmedian(selected, axis=0)
            coverage = float(np.isfinite(composite).mean())
            composite = pd.Series(composite).interpolate(limit_direction="both").to_numpy(dtype=float)
            if not np.isfinite(composite).all():
                composite = np.where(np.isfinite(composite), composite, own_gr)
            composite = pd.Series(composite).rolling(5, center=True, min_periods=1).median().to_numpy()
            for blend in (0.0, 0.5):
                values = composite if blend == 0.0 else blend * own_gr + (1.0 - blend) * composite
                label = f"horizontal_k{count}" if blend == 0.0 else f"own50_horizontal_k{count}"
                reference = pd.DataFrame({"TVT": target_grid, "GR": values})
                loss, rows, _, _ = robust_affine_prefix_loss(horizontal, reference)
                choices.append(
                    ReferenceChoice(
                        label,
                        reference,
                        loss,
                        rows,
                        tuple(nearest[:count]),
                        coverage,
                    )
                )
        return choices

    def choose_reference(
        self,
        target_well: str,
        horizontal: pd.DataFrame,
        target_typewell: pd.DataFrame,
        neighbor_counts: tuple[int, ...] = (4, 8, 16),
        own_loss_margin: float = 0.98,
    ) -> ReferenceChoice:
        choices = self.candidate_references(
            target_well, horizontal, target_typewell, neighbor_counts=neighbor_counts
        )
        own = choices[0]
        best = min(choices[1:], key=lambda item: item.prefix_loss)
        if best.prefix_rows < 80 or best.coverage < 0.60:
            return own
        return best if best.prefix_loss < own.prefix_loss * own_loss_margin else own


def choice_to_dict(choice: ReferenceChoice) -> dict[str, Any]:
    return {
        "name": choice.name,
        "prefix_loss": float(choice.prefix_loss),
        "prefix_rows": int(choice.prefix_rows),
        "neighbors": list(choice.neighbors),
        "coverage": float(choice.coverage),
    }
