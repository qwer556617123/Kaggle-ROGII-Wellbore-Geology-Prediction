"""Audit C0/C1 stratigraphic heel continuity on a native-mask anchor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_slope(md: np.ndarray, u: np.ndarray, from_end: bool) -> tuple[float, float]:
    slopes = []
    for width in (80, 160, 320, 640):
        if len(md) < max(40, width // 2):
            continue
        x = md[-width:] if from_end else md[:width]
        y = u[-width:] if from_end else u[:width]
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 40 or float(np.ptp(x[finite])) < 20.0:
            continue
        slope = float(np.polyfit(x[finite], y[finite], 1)[0])
        slopes.append(slope)
    if not slopes:
        return 0.0, float("inf")
    values = np.asarray(slopes, dtype=float)
    return float(np.median(values)), float(np.max(values) - np.min(values))


def summarize(records: list[dict], predictions: list[np.ndarray]) -> dict[str, float]:
    truth = np.concatenate([record["truth"] for record in records])
    pred = np.concatenate(predictions)
    rmses = np.asarray(
        [
            np.sqrt(np.mean(np.square(record["truth"] - well_pred)))
            for record, well_pred in zip(records, predictions)
        ]
    )
    sse = np.asarray(
        [np.sum(np.square(record["truth"] - well_pred)) for record, well_pred in zip(records, predictions)]
    )
    worst = np.argsort(sse)[-max(1, int(np.ceil(0.10 * len(sse)))) :]
    return {
        "row_rmse": float(np.sqrt(np.mean(np.square(truth - pred)))),
        "well_mean": float(rmses.mean()),
        "well_median": float(np.median(rmses)),
        "well_p90": float(np.quantile(rmses, 0.90)),
        "worst10_sse_share": float(sse[worst].sum() / sse.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--anchor", type=Path, default=Path("reports/native100_pf_anchor.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/c1_heel_continuity.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor = pd.read_csv(args.anchor, dtype={"id": "string"})[["id", "tvt"]]
    parts = anchor["id"].astype(str).str.rsplit("_", n=1, expand=True)
    anchor["well"] = parts[0].astype(str)
    anchor["row"] = pd.to_numeric(parts[1], errors="raise").astype(int)
    records = []
    for well, group in anchor.groupby("well", sort=True):
        horizontal = pd.read_csv(args.data_dir / "train" / f"{well}__horizontal_well.csv")
        group = group.sort_values("row")
        rows = group["row"].to_numpy(dtype=int)
        base = group["tvt"].to_numpy(dtype=float)
        md = horizontal["MD"].to_numpy(dtype=float)
        z = horizontal["Z"].to_numpy(dtype=float)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if not len(known) or int(rows[0]) != int(known[-1]) + 1:
            raise RuntimeError(f"Native suffix contract failed for {well}")
        known_u = tvt_input[known] + z[known]
        base_u = base + z[rows]
        known_slope, known_spread = robust_slope(md[known], known_u, from_end=True)
        pred_slope, pred_spread = robust_slope(md[rows], base_u, from_end=False)
        dm = md[rows] - md[known[-1]]
        records.append(
            {
                "well": str(well),
                "truth": horizontal.loc[rows, "TVT"].to_numpy(dtype=float),
                "base": base,
                "dm": dm,
                "gap": float(base_u[0] - known_u[-1]),
                "slope_mismatch": float(pred_slope - known_slope),
                "slope_spread": float(known_spread + pred_spread),
            }
        )

    candidates: dict[str, dict[str, float]] = {
        "base": summarize(records, [record["base"] for record in records])
    }
    c0_moves: dict[tuple[float, float], list[np.ndarray]] = {}
    for cap0 in (4.0, 8.0, 12.0):
        for tau0 in (120.0, 240.0, 480.0):
            moves = [
                -np.clip(record["gap"], -cap0, cap0) * np.exp(-record["dm"] / tau0)
                for record in records
            ]
            c0_moves[(cap0, tau0)] = moves
            candidates[f"c0_cap{cap0:g}_tau{tau0:g}"] = summarize(
                records,
                [record["base"] + move for record, move in zip(records, moves)],
            )

    base_c0 = c0_moves[(8.0, 240.0)]
    for tau1 in (240.0, 480.0, 960.0, 1920.0, 3840.0):
        for cap1 in (2.0, 4.0, 8.0, 12.0, 16.0):
            for shrink in (0.25, 0.50, 1.0):
                predictions = []
                for record, c0 in zip(records, base_c0):
                    reliability = np.clip(1.0 - record["slope_spread"] / 0.04, 0.0, 1.0)
                    raw = -record["slope_mismatch"] * record["dm"] * np.exp(
                        -record["dm"] / tau1
                    )
                    c1 = np.clip(raw, -cap1, cap1) * shrink * reliability
                    predictions.append(record["base"] + c0 + c1)
                key = f"c1_tau{tau1:g}_cap{cap1:g}_shrink{shrink:g}"
                candidates[key] = summarize(records, predictions)

    best = min(candidates, key=lambda key: candidates[key]["row_rmse"])
    report = {
        "wells": len(records),
        "rows": int(sum(len(record["truth"]) for record in records)),
        "best": best,
        "best_metrics": candidates[best],
        "base_metrics": candidates["base"],
        "current_c0_metrics": candidates["c0_cap8_tau240"],
        "candidates": candidates,
        "slope_mismatch_abs_median": float(
            np.median([abs(record["slope_mismatch"]) for record in records])
        ),
        "slope_mismatch_abs_p90": float(
            np.quantile([abs(record["slope_mismatch"]) for record in records], 0.90)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
