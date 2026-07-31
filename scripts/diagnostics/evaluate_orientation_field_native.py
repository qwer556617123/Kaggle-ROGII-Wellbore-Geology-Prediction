"""Native-mask audit for the cross-well orientation field and residual HMM."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from orientation_field_hmm import (
    OrientationFieldConfig,
    build_orientation_field,
    run_orientation_hmm,
)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(prediction)))))


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in details.groupby("variant", sort=False):
        scores = group["rmse"].to_numpy(dtype=float)
        sse = group["sse"].to_numpy(dtype=float)
        worst_count = max(1, int(np.ceil(0.10 * len(group))))
        rows.append(
            {
                "variant": variant,
                "wells": int(group["well"].nunique()),
                "rows": int(group["rows"].sum()),
                "row_rmse": float(np.sqrt(sse.sum() / group["rows"].sum())),
                "well_mean": float(scores.mean()),
                "well_median": float(np.median(scores)),
                "well_p90": float(np.quantile(scores, 0.90)),
                "worst10_sse_share": float(np.sort(sse)[-worst_count:].sum() / sse.sum()),
                "improved_wells": int(group["improved"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("row_rmse")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--anchor", type=Path, default=Path("reports/native100_pf_anchor.csv"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--hmm-limit", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/orientation_field_native100"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.time()
    anchor = pd.read_csv(args.anchor, dtype={"id": "string"})[["id", "tvt"]]
    parts = anchor["id"].astype(str).str.rsplit("_", n=1, expand=True)
    anchor["well"] = parts[0].astype(str)
    anchor["row"] = pd.to_numeric(parts[1], errors="raise").astype(int)
    wells = sorted(anchor["well"].unique())[: args.limit]
    config = OrientationFieldConfig()
    model = build_orientation_field(args.data_dir / "train", config)
    print(
        f"[field] observations={len(model.observations)} faults={len(model.fault_observations)}",
        flush=True,
    )

    records: list[dict] = []
    audits: list[dict] = []
    alphas = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    for well_index, well in enumerate(wells, 1):
        horizontal = pd.read_csv(args.data_dir / "train" / f"{well}__horizontal_well.csv")
        group = anchor[anchor["well"].eq(well)].sort_values("row")
        eval_rows = group["row"].to_numpy(dtype=int)
        base_eval = group["tvt"].to_numpy(dtype=float)
        truth = horizontal.loc[eval_rows, "TVT"].to_numpy(dtype=float)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=float)
        full_anchor = tvt_input.copy()
        full_anchor[eval_rows] = base_eval
        if not np.isfinite(full_anchor).all():
            raise RuntimeError(f"anchor does not cover native mask for {well}")
        path = model.predict_path(horizontal, full_anchor, target_well=well, use_faults=True)
        field_eval = path.field_path[eval_rows]
        move = field_eval - base_eval
        gated_move = path.confidence * move
        shape_move = move - float(np.mean(move))
        variants = {"anchor": base_eval, "field": field_eval}
        for alpha in alphas:
            variants[f"full_{alpha:.2f}"] = base_eval + alpha * move
            variants[f"gated_{alpha:.2f}"] = base_eval + alpha * gated_move
            variants[f"shape_{alpha:.2f}"] = base_eval + alpha * shape_move

        if well_index <= args.hmm_limit:
            typewell = pd.read_csv(args.data_dir / "train" / f"{well}__typewell.csv")
            for name, faults, response in (
                ("orientation_hmm", False, False),
                ("fault_hmm", True, False),
                ("tool_hmm", False, True),
                ("combined_hmm", True, True),
            ):
                result = run_orientation_hmm(
                    horizontal,
                    typewell,
                    full_anchor,
                    model,
                    target_well=well,
                    use_faults=faults,
                    use_tool_response=response,
                )
                posterior_eval = result.posterior[eval_rows]
                variants[name] = posterior_eval
                variants[f"{name}_010"] = 0.90 * base_eval + 0.10 * posterior_eval
                variants[f"{name}_025"] = 0.75 * base_eval + 0.25 * posterior_eval
                audits.append({"well": well, "variant": name, "metadata": result.metadata})

        anchor_score = _rmse(truth, base_eval)
        known_fraction = float(np.mean(np.isfinite(tvt_input)))
        stratum = "low" if known_fraction < 0.235 else ("mid" if known_fraction < 0.30 else "high")
        for variant, prediction in variants.items():
            score = _rmse(truth, prediction)
            records.append(
                {
                    "well": well,
                    "stratum": stratum,
                    "known_fraction": known_fraction,
                    "variant": variant,
                    "rows": int(len(truth)),
                    "rmse": score,
                    "sse": float(np.sum(np.square(truth - prediction))),
                    "improved": bool(score < anchor_score),
                    "confidence_mean": path.metadata["confidence_mean"],
                    "distance_median": path.metadata["distance_median"],
                    "condition_median": path.metadata["condition_median"],
                    "move_mean": float(np.mean(move)),
                    "move_std": float(np.std(move)),
                }
            )
        audits.append({"well": well, "variant": "orientation_path", "metadata": path.metadata})
        if well_index % 10 == 0 or well_index == len(wells):
            print(f"[{well_index:03d}/{len(wells):03d}] {well}", flush=True)

    details = pd.DataFrame(records)
    summary = _summarize(details)
    summary_index = summary.set_index("variant")
    candidate = "gated_0.15"
    base = summary_index.loc["anchor"]
    selected = summary_index.loc[candidate]
    strata = []
    for stratum, group in details[details["variant"].isin(["anchor", candidate])].groupby("stratum"):
        values = {row["variant"]: row for _, row in _summarize(group).iterrows()}
        strata.append(
            {
                "stratum": stratum,
                "wells": int(group["well"].nunique()),
                "anchor_row_rmse": float(values["anchor"]["row_rmse"]),
                "candidate_row_rmse": float(values[candidate]["row_rmse"]),
                "gain": float(values["anchor"]["row_rmse"] - values[candidate]["row_rmse"]),
            }
        )
    improved_strata = sum(row["gain"] > 0 for row in strata)
    gate = {
        "verdict": "PASS"
        if (
            float(base["row_rmse"] - selected["row_rmse"]) >= 0.50
            and float(base["well_median"] - selected["well_median"]) >= 0.20
            and float(selected["well_p90"]) <= float(base["well_p90"])
            and improved_strata >= 2
        )
        else "STOP",
        "candidate": candidate,
        "row_gain": float(base["row_rmse"] - selected["row_rmse"]),
        "well_median_gain": float(base["well_median"] - selected["well_median"]),
        "well_p90_change": float(selected["well_p90"] - base["well_p90"]),
        "improved_strata": int(improved_strata),
        "strata": strata,
        "wells": int(len(wells)),
        "rows": int(details[details["variant"].eq("anchor")]["rows"].sum()),
        "runtime_sec": float(time.time() - started),
        "same_id_contacts_excluded": True,
        "target_tail_tvt_used_for_prediction": False,
        "fixed_public_ids_used": False,
        "config": config.__dict__,
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
