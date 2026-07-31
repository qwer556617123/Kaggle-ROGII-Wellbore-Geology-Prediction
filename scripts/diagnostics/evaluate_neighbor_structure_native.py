"""Native-mask audit for cross-well structural-path transfer.

The candidate transfers the complete ``U = TVT + Z`` curve from nearby train
wells after XY nearest-path alignment.  The target well contributes only its
visible TVT prefix, which calibrates a robust affine datum correction.  Same-ID
train wells are always excluded.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


@dataclass
class StructuralProfile:
    well: str
    xy: np.ndarray
    md: np.ndarray
    u: np.ndarray
    direction: np.ndarray


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(pred)))))


def robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return intercept, slope, and robust residual RMSE."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0 = float(x[-1])
    scale = max(float(np.ptp(x)), 1.0)
    xn = (x - x0) / scale
    design = np.column_stack([np.ones(len(xn)), xn])
    weights = np.ones(len(xn), dtype=float)
    coef = np.array([float(np.median(y)), 0.0])
    for _ in range(8):
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        coef = np.linalg.solve(lhs + np.eye(2) * 1e-8, rhs)
        residual = y - design @ coef
        sigma = max(1.4826 * float(np.median(np.abs(residual))), 0.25)
        weights = np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-9))
    residual = y - design @ coef
    robust_residual = np.clip(residual, -3.0 * sigma, 3.0 * sigma)
    return float(coef[0]), float(coef[1] / scale), rmse(robust_residual, np.zeros_like(residual))


def unit_direction(xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(xy[-1] - xy[0], dtype=float)
    norm = float(np.linalg.norm(delta))
    return delta / max(norm, 1e-9)


def load_profiles(data_dir: Path, stride: int) -> tuple[dict[str, StructuralProfile], cKDTree, np.ndarray]:
    profiles: dict[str, StructuralProfile] = {}
    global_xy = []
    global_well = []
    paths = sorted((data_dir / "train").glob("*__horizontal_well.csv"))
    for index, path in enumerate(paths, 1):
        well = path.name.split("__", 1)[0]
        frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT"])
        valid = frame["TVT"].notna().to_numpy()
        rows = np.flatnonzero(valid)[::stride]
        if len(rows) < 32:
            continue
        xy = frame.loc[rows, ["X", "Y"]].to_numpy(dtype=float)
        md = frame.loc[rows, "MD"].to_numpy(dtype=float)
        u = (
            frame.loc[rows, "TVT"].to_numpy(dtype=float)
            + frame.loc[rows, "Z"].to_numpy(dtype=float)
        )
        u = gaussian_filter1d(u, sigma=max(1.0, 12.0 / max(stride, 1)), mode="nearest")
        profiles[well] = StructuralProfile(well, xy, md, u, unit_direction(xy))
        global_xy.append(xy)
        global_well.extend([well] * len(xy))
        if index % 100 == 0:
            print(f"[profiles] {index}/{len(paths)}", flush=True)
    xy_all = np.concatenate(global_xy, axis=0)
    return profiles, cKDTree(xy_all), np.asarray(global_well, dtype=object)


def nearby_wells(
    heel_xy: np.ndarray,
    target_well: str,
    tree: cKDTree,
    global_well: np.ndarray,
    count: int,
) -> list[str]:
    query_k = min(len(global_well), max(512, count * 64))
    _, indices = tree.query(np.asarray(heel_xy, dtype=float), k=query_k)
    selected = []
    for index in np.atleast_1d(indices):
        well = str(global_well[int(index)])
        if well == target_well or well in selected:
            continue
        selected.append(well)
        if len(selected) >= count:
            break
    return selected


def align_profile(
    target: pd.DataFrame,
    known: np.ndarray,
    profile: StructuralProfile,
    prefix_rows: int,
) -> tuple[np.ndarray, dict[str, float]] | None:
    target_xy = target[["X", "Y"]].to_numpy(dtype=float)
    target_md = target["MD"].to_numpy(dtype=float)
    target_u = target["TVT_input"].to_numpy(dtype=float) + target["Z"].to_numpy(dtype=float)
    target_direction = unit_direction(target_xy)
    direction_similarity = abs(float(np.dot(target_direction, profile.direction)))
    if direction_similarity < 0.65:
        return None

    distances, nearest = cKDTree(profile.xy).query(target_xy, k=1)
    mapped_u = profile.u[np.asarray(nearest, dtype=int)]
    mapped_u = gaussian_filter1d(mapped_u, sigma=10.0, mode="nearest")
    known_rows = np.flatnonzero(known)[-prefix_rows:]
    if len(known_rows) < 80:
        return None
    residual = target_u[known_rows] - mapped_u[known_rows]
    intercept, slope, prefix_error = robust_affine(target_md[known_rows], residual)
    correction = intercept + slope * (target_md - target_md[known_rows[-1]])
    correction = np.clip(correction, intercept - 25.0, intercept + 25.0)
    pred_u = mapped_u + correction
    metadata = {
        "prefix_rmse": prefix_error,
        "distance_prefix_median": float(np.median(distances[known_rows])),
        "distance_all_p90": float(np.quantile(distances, 0.90)),
        "direction_similarity": direction_similarity,
        "datum_intercept": intercept,
        "datum_slope_per_ft": slope,
    }
    metadata["score"] = (
        prefix_error
        + 0.0025 * metadata["distance_prefix_median"]
        + 12.0 * (1.0 - direction_similarity)
        + 0.05 * abs(correction[-1] - correction[known_rows[-1]])
    )
    return pred_u, metadata


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in details.groupby("variant", sort=False):
        scores = group["rmse"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "wells": int(group["well"].nunique()),
                "rows": int(group["rows"].sum()),
                "row_rmse": float(np.sqrt(group["sse"].sum() / group["rows"].sum())),
                "well_mean": float(np.mean(scores)),
                "well_median": float(np.median(scores)),
                "well_p90": float(np.quantile(scores, 0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("row_rmse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--anchor", type=Path, default=Path("reports/native100_pf_anchor.csv"))
    parser.add_argument("--anchor-details", type=Path, default=Path("reports/native100_pf_details.csv"))
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/neighbor_structure_native100"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--prefix-rows", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    anchor = pd.read_csv(args.anchor)[["id", "tvt"]]
    anchor["id"] = anchor["id"].astype(str)
    anchor["well"] = anchor["id"].str.rsplit("_", n=1).str[0]
    anchor["row"] = anchor["id"].str.rsplit("_", n=1).str[1].astype(int)
    metadata = pd.read_csv(args.anchor_details).drop_duplicates("well")
    profile_map = metadata.set_index("well").get("nearest_public_profile", pd.Series(dtype=str)).to_dict()
    wells = sorted(anchor["well"].unique())[: args.limit]
    profiles, global_tree, global_well = load_profiles(args.data_dir, args.stride)

    records = []
    audits = []
    for well_index, well in enumerate(wells, 1):
        target = pd.read_csv(args.data_dir / "train" / f"{well}__horizontal_well.csv")
        known = target["TVT_input"].notna().to_numpy()
        eval_mask = ~known & target["TVT"].notna().to_numpy()
        eval_rows = np.flatnonzero(eval_mask)
        group = anchor[anchor["well"].eq(well)].set_index("row")
        if not set(eval_rows).issubset(group.index):
            raise RuntimeError(f"anchor rows do not cover native mask for {well}")
        base = group.loc[eval_rows, "tvt"].to_numpy(dtype=float)
        truth = target.loc[eval_rows, "TVT"].to_numpy(dtype=float)
        z_eval = target.loc[eval_rows, "Z"].to_numpy(dtype=float)
        heel_xy = target.loc[np.flatnonzero(known)[-1], ["X", "Y"]].to_numpy(dtype=float)
        candidates = nearby_wells(
            heel_xy, well, global_tree, global_well, args.neighbor_count
        )
        aligned = []
        for neighbor in candidates:
            result = align_profile(target, known, profiles[neighbor], args.prefix_rows)
            if result is None:
                continue
            pred_u, info = result
            aligned.append((float(info["score"]), neighbor, pred_u, info))
        aligned.sort(key=lambda item: item[0])
        selected = aligned[: args.top_k]
        variants = {"anchor": base}
        for alpha in (0.10, 0.25, 0.50):
            variants[f"analog_full_{alpha:.2f}"] = base.copy()
            variants[f"analog_shape_{alpha:.2f}"] = base.copy()
        if selected:
            scores = np.asarray([item[0] for item in selected], dtype=float)
            weights = np.exp(-(scores - scores.min()) / max(float(np.std(scores)), 1.0))
            weights /= weights.sum()
            analog_u = np.sum(
                np.stack([item[2] for item in selected], axis=0) * weights[:, None],
                axis=0,
            )
            analog = analog_u[eval_rows] - z_eval
            move = analog - base
            shape_move = move - float(np.mean(move))
            for alpha in (0.10, 0.25, 0.50):
                variants[f"analog_full_{alpha:.2f}"] = base + alpha * move
                variants[f"analog_shape_{alpha:.2f}"] = base + alpha * shape_move
            audits.append(
                {
                    "well": well,
                    "neighbors": [item[1] for item in selected],
                    "weights": [float(value) for value in weights],
                    "scores": [float(item[0]) for item in selected],
                    "best_metadata": selected[0][3],
                    "move_mean": float(np.mean(move)),
                    "move_std": float(np.std(move)),
                }
            )
        for variant, prediction in variants.items():
            score = rmse(truth, prediction)
            records.append(
                {
                    "well": well,
                    "profile": profile_map.get(well, "unknown"),
                    "variant": variant,
                    "rows": int(len(truth)),
                    "rmse": score,
                    "sse": float(np.sum(np.square(truth - prediction))),
                }
            )
        print(
            f"[{well_index:03d}/{len(wells):03d}] {well} "
            f"anchor={rmse(truth, base):.3f} neighbors={len(selected)}",
            flush=True,
        )

    details = pd.DataFrame(records)
    summary = summarize(details)
    base_row = summary.set_index("variant").loc["anchor"]
    candidate = summary.iloc[0]
    gate = {
        "verdict": "PASS" if candidate["variant"] != "anchor" and base_row["row_rmse"] - candidate["row_rmse"] >= 0.15 else "STOP",
        "candidate": str(candidate["variant"]),
        "row_gain": float(base_row["row_rmse"] - candidate["row_rmse"]),
        "well_median_gain": float(base_row["well_median"] - candidate["well_median"]),
        "runtime_sec": float(time.time() - started),
        "same_id_excluded": True,
        "wells": int(len(wells)),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), index=False)
    summary.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), index=False)
    args.output_prefix.with_name(args.output_prefix.name + "_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_prefix.with_name(args.output_prefix.name + "_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
