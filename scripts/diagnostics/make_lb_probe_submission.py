"""Create low-dimensional Public LB probe submissions.

This script applies simple, auditable perturbations to an existing submission:

- per-well constant offsets, in feet;
- per-well zero-mean linear trends, where the configured value is the
  end-minus-start delta in feet.

Use symmetric pairs, for example +10 and -10 ft, so the Public LB score
difference estimates the residual projection along that basis direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_json_map(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Expected JSON or comma-separated well=value pairs, got: {raw}") from None
            key, value = item.split("=", 1)
            data[key.strip()] = float(value)
    return {str(k): float(v) for k, v in data.items()}


def add_well_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    split = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    out["well"] = split[0]
    out["row_idx"] = split[1].astype(int)
    return out


def apply_probe(
    submission: pd.DataFrame,
    offsets: dict[str, float],
    trends: dict[str, float],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    work = add_well_columns(submission)
    work["delta"] = 0.0
    summary: list[dict[str, object]] = []

    for well, offset in offsets.items():
        mask = work["well"] == well
        if not mask.any():
            raise ValueError(f"Offset well not found in submission: {well}")
        work.loc[mask, "delta"] += offset
        summary.append({"well": well, "kind": "offset", "value": offset, "rows": int(mask.sum())})

    for well, trend in trends.items():
        mask = work["well"] == well
        if not mask.any():
            raise ValueError(f"Trend well not found in submission: {well}")
        idx = work.loc[mask].sort_values("row_idx").index
        if len(idx) == 1:
            basis = np.array([0.0])
        else:
            basis = np.linspace(-0.5, 0.5, len(idx), dtype=float)
        work.loc[idx, "delta"] += trend * basis
        summary.append({"well": well, "kind": "linear_trend", "value": trend, "rows": int(len(idx))})

    work["tvt"] = work["tvt"].astype(float) + work["delta"]
    return work[["id", "tvt"]], summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Base submission CSV with id,tvt columns.")
    parser.add_argument("--out", required=True, type=Path, help="Output probe submission CSV.")
    parser.add_argument("--offsets", default="{}", help='JSON map or well=value pairs, e.g. {"00e12e8b": 10} or 00e12e8b=10.')
    parser.add_argument("--trends", default="{}", help='JSON map or well=value pairs, e.g. {"00e12e8b": 20} or 00e12e8b=20.')
    parser.add_argument("--summary", type=Path, help="Optional JSON summary output.")
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    missing = {"id", "tvt"} - set(base.columns)
    if missing:
        raise ValueError(f"Base submission missing columns: {sorted(missing)}")

    offsets = parse_json_map(args.offsets)
    trends = parse_json_map(args.trends)
    out, summary = apply_probe(base[["id", "tvt"]], offsets, trends)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    payload = {
        "base": str(args.base),
        "out": str(args.out),
        "offsets": offsets,
        "trends": trends,
        "operations": summary,
        "rows": int(len(out)),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
