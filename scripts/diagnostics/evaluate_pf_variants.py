"""Local CV for PF-family ROGII notebook variants.

This intentionally evaluates train wells as pseudo-hidden wells by using each
well's existing TVT_input mask and scoring against the withheld TVT tail.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")

PF_SCALES = (3.0, 5.0, 8.0, 12.0)
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
BIN_SELECTOR_VARIANTS = {
    "bin_best_v1": {
        0: "grid_s3_b0_h0p05",
        2: "grid_s8_b0_h0p05",
        3: "grid_s3_b0_h0p15",
        5: "grid_s12_b0_h0p2",
    },
    "bin_less_aggressive": {
        0: "grid_s3_b0_h0p1",
        2: "grid_s8_b0_h0p1",
        3: "grid_s3_b0_h0p15",
        5: "grid_s12_b0_h0p15",
    },
    "bin_lb_safe": {
        0: "grid_s3_b0_h0p2",
        2: "grid_s8_b0_h0p1",
        3: "grid_s3_b0_h0p15",
        5: "grid_s12_b0_h0p2",
    },
}
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]


@dataclass(frozen=True)
class PfConfig:
    n_particles: int
    n_seeds: int


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_well(data_dir: Path, wid: str, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = data_dir / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv")
    return hw, tw


def well_ids(data_dir: Path, split: str = "train") -> list[str]:
    return [
        os.path.basename(path).split("__")[0]
        for path in sorted(glob.glob(str(data_dir / split / "*__horizontal_well.csv")))
    ]


def interpolate_gr(hw: pd.DataFrame, fallback: float) -> np.ndarray:
    return hw["GR"].interpolate(limit_direction="both").fillna(fallback).to_numpy(dtype=float)


def typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(dtype=float)
    return tw_tvt, tw_gr


def well_profile(hw: pd.DataFrame) -> dict[str, float]:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    return {
        "n": float(len(hw)),
        "n_eval": float(eval_mask.sum()),
        "known_frac": float((~eval_mask).mean()),
        "z_span": float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0,
        "gr_nan": float(hw.loc[eval_mask, "GR"].isna().mean()) if eval_mask.any() else 0.0,
    }


def build_profiles(data_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    for wid in well_ids(data_dir, split):
        hw, _ = load_well(data_dir, wid, split)
        rows.append({"well": wid, **well_profile(hw)})
    return pd.DataFrame(rows)


def select_wells(data_dir: Path, selection: str, limit: int | None, seed: int) -> list[str]:
    train = build_profiles(data_dir, "train")
    test_ids = set(well_ids(data_dir, "test"))
    train = train[~train["well"].isin(test_ids)].copy()
    train = train[train["n_eval"] > 0].copy()

    if selection == "all":
        chosen = train.sort_values("well")
    elif selection == "random":
        chosen = train.sample(frac=1.0, random_state=seed)
    elif selection == "hard":
        chosen = train.assign(score=train["n_eval"].rank(pct=True) + train["z_span"].rank(pct=True))
        chosen = chosen.sort_values("score", ascending=False)
    elif selection == "lb_like":
        test = build_profiles(data_dir, "test")
        cols = ["n", "n_eval", "known_frac", "z_span", "gr_nan"]
        scale = train[cols].std().replace(0, 1.0)
        distances = []
        for _, tr in train.iterrows():
            d = []
            for _, te in test.iterrows():
                diff = ((tr[cols] - te[cols]) / scale).to_numpy(dtype=float)
                d.append(float(np.sqrt(np.mean(diff**2))))
            distances.append(min(d))
        chosen = train.assign(distance=distances).sort_values("distance")
    else:
        raise ValueError(f"Unknown selection={selection}")

    wells = chosen["well"].tolist()
    return wells[:limit] if limit else wells


def tvt_from_contacts(hw: pd.DataFrame, tw: pd.DataFrame, ref_col: str = "EGFDU") -> np.ndarray:
    tw_g = tw.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw["TVT"] - (ref_tvt - (hw["Z"] - hw[ref_col]))).mean()
    return (ref_tvt - (hw["Z"] - hw[ref_col]) + offset).to_numpy(dtype=float)


def run_particle_filter(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    cfg: PfConfig,
    seed: int,
    return_meta: bool = False,
):
    tw_tvt, tw_gr = typewell_arrays(tw)
    known = hw[hw["TVT_input"].notna()]
    eval_rows = hw[hw["TVT_input"].isna()]
    if len(eval_rows) == 0:
        out = hw["TVT_input"].to_numpy(dtype=float).copy()
        return (out, 0.0, {"entropy": 0.0, "n_resample": 0}) if return_meta else (out, 0.0)

    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])

    tw_at_known = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0).to_numpy(dtype=float) - tw_at_known), 10.0, 60.0))

    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(dtype=float))
    dz = np.diff(tail["Z"].to_numpy(dtype=float))
    dm = np.diff(tail["MD"].to_numpy(dtype=float))
    m = dm > 0
    init_rate = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    rng = np.random.default_rng(seed)
    n = cfg.n_particles
    pos = last_tvt + last_z + 2.0 * rng.standard_normal(n)
    rate = init_rate + 0.01 * rng.standard_normal(n)
    weights = np.ones(n) / n

    md_v = eval_rows["MD"].to_numpy(dtype=float)
    z_v = eval_rows["Z"].to_numpy(dtype=float)
    gr_v = interpolate_gr(hw, tw_gr.mean())[eval_rows.index]

    out = hw["TVT_input"].to_numpy(dtype=float).copy()
    pred = np.empty(len(eval_rows))
    prev_md = last_md
    log_lik = 0.0
    entropies = []
    n_resample = 0

    for i in range(len(eval_rows)):
        dm_step = max(md_v[i] - prev_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(n)
        pos = pos + rate * dm_step + 0.005 * rng.standard_normal(n)
        tvt_particles = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_particles + z_v[i]

        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        residual = (gr_v[i] - expected_gr) / gr_sigma
        likelihood = np.exp(-0.5 * np.minimum(residual**2, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        avg_likelihood = float((weights * likelihood).sum())
        log_lik += math.log(max(avg_likelihood, 1e-300))
        weights = weights * likelihood
        w_sum = weights.sum()
        weights = weights / w_sum if w_sum > 0 else np.ones(n) / n

        entropy = float(-np.sum(weights * np.log(weights + 1e-300)) / np.log(n))
        entropies.append(entropy)
        n_eff = 1.0 / np.sum(weights**2)
        if n_eff < 0.5 * n:
            cum = np.cumsum(weights)
            u0 = rng.uniform(0, 1.0 / n)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)
            pos = pos[idx] + 0.1 * rng.standard_normal(n)
            rate = rate[idx] + 0.001 * rng.standard_normal(n)
            weights = np.ones(n) / n
            n_resample += 1

        pred[i] = float(np.dot(weights, pos - z_v[i]))
        prev_md = md_v[i]

    out[list(eval_rows.index)] = pred
    meta = {"entropy": float(np.mean(entropies)) if entropies else 0.0, "n_resample": n_resample}
    return (out, log_lik, meta) if return_meta else (out, log_lik)


def pf_path_library(hw: pd.DataFrame, tw: pd.DataFrame, cfg: PfConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    paths = []
    log_liks = []
    entropies = []
    for seed in range(cfg.n_seeds):
        path, ll, meta = run_particle_filter(hw, tw, cfg=cfg, seed=seed, return_meta=True)
        paths.append(path)
        log_liks.append(ll)
        entropies.append(meta["entropy"])
    return np.stack(paths, axis=0), np.asarray(log_liks), {"entropy": float(np.mean(entropies))}


def pf_scales(paths: np.ndarray, log_liks: np.ndarray, eval_mask: np.ndarray, meta: dict) -> tuple[dict[str, np.ndarray], dict]:
    centered = log_liks - log_liks.max()
    out = {}
    for scale in PF_SCALES:
        weights = np.exp(centered / scale)
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = np.sum(paths * weights[:, None], axis=0)
    out["pf_mean"] = paths.mean(axis=0)
    meta = dict(meta)
    meta["path_std"] = float(np.mean(np.std(paths[:, eval_mask], axis=0)))
    return out, meta


def score_path_full_sequence(path: np.ndarray, hw: pd.DataFrame, tw: pd.DataFrame) -> float:
    tw_tvt, tw_gr = typewell_arrays(tw)
    eval_mask = hw["TVT_input"].isna().to_numpy()
    if not eval_mask.any():
        return 0.0
    gr = interpolate_gr(hw, float(np.nanmean(tw_gr)))[eval_mask]
    expected = np.interp(path[eval_mask], tw_tvt, tw_gr)
    known = hw[hw["TVT_input"].notna()]
    tw_at_known = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0).to_numpy(dtype=float) - tw_at_known), 10.0, 60.0))
    residual = (gr - expected) / gr_sigma
    emission = -0.5 * np.mean(np.minimum(residual**2, 100.0))
    tvt_eval = path[eval_mask]
    smooth = -0.02 * float(np.std(np.diff(tvt_eval))) if len(tvt_eval) > 2 else 0.0
    anchor = float(known["TVT_input"].iloc[-1])
    hold = -0.0005 * abs(float(tvt_eval[-1] - anchor))
    return emission + smooth + hold


def path_rerank(paths: np.ndarray, log_liks: np.ndarray, hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    full_scores = np.array([score_path_full_sequence(path, hw, tw) for path in paths])
    combined = 0.35 * (log_liks - log_liks.max()) / max(1.0, abs(log_liks.std())) + full_scores
    top_k = min(32, len(paths))
    top = np.argsort(combined)[-top_k:]
    weights = np.exp(combined[top] - combined[top].max())
    weights /= weights.sum()
    return np.sum(paths[top] * weights[:, None], axis=0)


def event_strength(values: np.ndarray) -> np.ndarray:
    x = pd.Series(values).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)
    if len(x) > 9:
        win = min(31, len(x) if len(x) % 2 == 1 else len(x) - 1)
        x = savgol_filter(x, win, min(3, win - 1))
    grad = np.abs(np.gradient(x))
    q80 = np.quantile(grad, 0.80) if len(grad) else 0.0
    q95 = np.quantile(grad, 0.95) if len(grad) else 1.0
    return np.clip((grad - q80) / max(q95 - q80, 1e-6), 0.0, 1.0)


def beam_search(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_tvt: float,
    bs: int,
    move_cost: float,
    err_scale: float,
    smooth_radius: int,
    event_weighted: bool = False,
) -> np.ndarray:
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if smooth_radius > 0 and n > max(3, 2 * smooth_radius + 1):
        win = min(2 * smooth_radius + 1, n if n % 2 == 1 else n - 1)
        obs = savgol_filter(hgr, win, min(2, win - 1))
    else:
        obs = hgr.copy()
    ev = event_strength(obs) if event_weighted else np.zeros(n)
    start_idx = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    move_penalty = move_cost * np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    beam_idx = np.full(bs, start_idx, dtype=np.int64)
    beam_cost = np.full(bs, np.inf)
    beam_cost[0] = 0.0
    beam_n = 1
    result = np.zeros(n)
    for step in range(n):
        next_idx = beam_idx[:beam_n, None] + moves[None, :]
        clipped = np.clip(next_idx, 0, nt - 1)
        valid = (next_idx >= 0) & (next_idx < nt)
        local_scale = err_scale / (1.0 + 3.0 * ev[step])
        gr_error = (obs[step] - tw_gr[clipped]) ** 2 / max(local_scale, 1e-6)
        total = beam_cost[:beam_n, None] + gr_error + move_penalty[None, :]
        total = np.where(valid, total, np.inf)
        flat_idx = next_idx.flatten()
        flat_total = total.flatten()
        flat_valid = valid.flatten()
        flat_idx = flat_idx[flat_valid]
        flat_total = flat_total[flat_valid]
        order = np.argsort(flat_total)
        idx_sorted = flat_idx[order]
        total_sorted = flat_total[order]
        _, first = np.unique(idx_sorted, return_index=True)
        idx_unique = idx_sorted[first]
        total_unique = total_sorted[first]
        kept = min(bs, len(idx_unique))
        top = np.argpartition(total_unique, min(kept - 1, len(total_unique) - 1))[:kept]
        top = top[np.argsort(total_unique[top])]
        beam_idx[:kept] = idx_unique[top]
        beam_cost[:kept] = total_unique[top]
        if kept < bs:
            beam_idx[kept:] = beam_idx[kept - 1]
            beam_cost[kept:] = np.inf
        beam_n = kept
        result[step] = tw_tvt[beam_idx[0]]
    return result


def beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame, event_weighted: bool) -> np.ndarray:
    known = hw[hw["TVT_input"].notna()]
    eval_rows = hw[hw["TVT_input"].isna()]
    if len(eval_rows) == 0:
        return hw["TVT_input"].to_numpy(dtype=float).copy()
    last_tvt = float(known.iloc[-1]["TVT_input"])
    tw_tvt, tw_gr = typewell_arrays(tw)
    gr_all = interpolate_gr(hw, float(np.nanmean(tw_gr)))
    hgr = gr_all[eval_rows.index]
    paths = [
        beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r, event_weighted=event_weighted)
        for (bs, mc, es, r) in BEAM_CONFIGS
    ]
    out = hw["TVT_input"].to_numpy(dtype=float).copy()
    out[list(eval_rows.index)] = np.stack(paths, axis=0).mean(axis=0)
    return out


def selector_well_code(hw: pd.DataFrame) -> tuple[int, str, float, float]:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT), n_eval, z_span


def parse_selector_variant(name: str) -> tuple[float, float, float]:
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_weight = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return scale, beam_weight, hold_weight


def apply_selector_variant(name: str, pf_by_scale: dict[str, np.ndarray], beam: np.ndarray, last_known_tvt: float) -> np.ndarray:
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale.get("pf_scale_8", pf_by_scale["pf_mean"]))
    pred = (1.0 - beam_weight) * base + beam_weight * beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def parse_grid_variant(name: str) -> tuple[float, float, float]:
    match = re.fullmatch(r"grid_s([0-9]+(?:p[0-9]+)?)_b([0-9]+(?:p[0-9]+)?)_h([0-9]+(?:p[0-9]+)?)", name)
    if not match:
        raise ValueError(f"Bad grid variant name={name}")
    scale = float(match.group(1).replace("p", "."))
    beam_weight = float(match.group(2).replace("p", "."))
    hold_weight = float(match.group(3).replace("p", "."))
    return scale, beam_weight, hold_weight


def apply_grid_variant(name: str, pf_by_scale: dict[str, np.ndarray], beam: np.ndarray, last_known_tvt: float) -> np.ndarray:
    scale, beam_weight, hold_weight = parse_grid_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale["pf_mean"])
    pred = (1.0 - beam_weight) * base + beam_weight * beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def predict_variants(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    variants: list[str],
    cfg: PfConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    predictions = {}
    meta_out = {}
    needs_pf = any(
        v in {"pf_scale_8", "public_selector", "path_rerank", "event_beam", "uncertainty_selector"}
        or v.startswith("grid_s")
        or v in BIN_SELECTOR_VARIANTS
        for v in variants
    )
    paths = log_liks = None
    pf_by_scale = {}
    pf_meta = {}
    if needs_pf:
        paths, log_liks, pf_meta = pf_path_library(hw, tw, cfg)
        pf_by_scale, pf_meta = pf_scales(paths, log_liks, eval_mask, pf_meta)
        meta_out.update({f"pf_{k}": v for k, v in pf_meta.items()})

    beam_event = None
    if any(
        v in {"beam_event", "public_selector", "event_beam", "uncertainty_selector"}
        or v.startswith("grid_s")
        or v in BIN_SELECTOR_VARIANTS
        for v in variants
    ):
        beam_event = beam_ensemble(hw, tw, event_weighted=True)

    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])
    for variant in variants:
        if variant == "contact_physics":
            predictions[variant] = tvt_from_contacts(hw, tw)
        elif variant == "pf_scale_8":
            predictions[variant] = pf_by_scale["pf_scale_8"]
        elif variant == "beam_event":
            predictions[variant] = beam_event
        elif variant == "public_selector":
            _, selector, _, _ = selector_well_code(hw)
            predictions[variant] = apply_selector_variant(selector, pf_by_scale, beam_event, last_known_tvt)
        elif variant == "path_rerank":
            predictions[variant] = path_rerank(paths, log_liks, hw, tw)
        elif variant == "event_beam":
            predictions[variant] = 0.72 * beam_event + 0.20 * pf_by_scale["pf_scale_8"] + 0.08 * last_known_tvt
        elif variant == "uncertainty_selector":
            _, selector, _, _ = selector_well_code(hw)
            base = apply_selector_variant(selector, pf_by_scale, beam_event, last_known_tvt)
            pf_std = pf_meta["path_std"]
            entropy = pf_meta["entropy"]
            beam_delta = float(np.mean(np.abs(base[eval_mask] - beam_event[eval_mask]))) if eval_mask.any() else 0.0
            gr_nan = float(hw.loc[eval_mask, "GR"].isna().mean()) if eval_mask.any() else 0.0
            hold_weight = np.clip(0.04 + 0.002 * pf_std + 0.08 * gr_nan, 0.04, 0.28)
            beam_weight = np.clip(0.04 + 0.003 * beam_delta + 0.08 * (1.0 - entropy), 0.04, 0.35)
            pred = (1.0 - beam_weight) * pf_by_scale["pf_scale_8"] + beam_weight * beam_event
            pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
            predictions[variant] = pred
            meta_out["unc_hold_weight"] = float(hold_weight)
            meta_out["unc_beam_weight"] = float(beam_weight)
        elif variant.startswith("grid_s"):
            predictions[variant] = apply_grid_variant(variant, pf_by_scale, beam_event, last_known_tvt)
        elif variant in BIN_SELECTOR_VARIANTS:
            code, _, _, _ = selector_well_code(hw)
            selected = BIN_SELECTOR_VARIANTS[variant].get(code, "grid_s3_b0_h0p2")
            predictions[variant] = apply_grid_variant(selected, pf_by_scale, beam_event, last_known_tvt)
            meta_out[f"{variant}_code"] = float(code)
        else:
            raise ValueError(f"Unknown variant={variant}")
    return predictions, meta_out


def fmt_grid_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def selector_grid_variants() -> list[str]:
    scales = [3.0, 5.0, 8.0, 12.0]
    beam_weights = [0.0, 0.05, 0.10, 0.20, 0.30]
    hold_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    return [
        f"grid_s{fmt_grid_value(scale)}_b{fmt_grid_value(beam)}_h{fmt_grid_value(hold)}"
        for scale in scales
        for beam in beam_weights
        for hold in hold_weights
    ]


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in details.groupby("variant"):
        y_true = np.concatenate([np.asarray(x) for x in g["y_true"].to_list()])
        y_pred = np.concatenate([np.asarray(x) for x in g["y_pred"].to_list()])
        rows.append(
            {
                "variant": variant,
                "n_wells": int(g["well"].nunique()),
                "n_rows": int(sum(len(x) for x in g["y_true"])),
                "row_rmse": rmse(y_true, y_pred),
                "well_rmse_mean": float(g["rmse"].mean()),
                "well_rmse_median": float(g["rmse"].median()),
                "well_rmse_p75": float(g["rmse"].quantile(0.75)),
                "well_rmse_max": float(g["rmse"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--particles", type=int, default=160)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--variants",
        default="public_selector,path_rerank,event_beam,uncertainty_selector",
        help="Comma-separated variants.",
    )
    parser.add_argument("--selector-grid", action="store_true", help="Evaluate a PF scale / beam / hold weight grid.")
    parser.add_argument("--summary-output", type=Path, default=Path("docs/pf_variant_cv_summary.csv"))
    parser.add_argument("--detail-output", type=Path, default=Path("docs/pf_variant_cv_details.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if args.selector_grid:
        variants = ["public_selector", *selector_grid_variants()]
    wells = select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    print(
        f"Evaluating variants={variants} selection={args.selection} wells={len(wells)} "
        f"seeds={cfg.n_seeds} particles={cfg.n_particles}"
    )

    rows = []
    for i, wid in enumerate(wells, 1):
        hw, tw = load_well(args.data_dir, wid, "train")
        eval_mask = hw["TVT_input"].isna().to_numpy()
        y_true = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        if len(y_true) == 0:
            continue
        profile = well_profile(hw)
        print(f"[{i:03d}/{len(wells):03d}] {wid} n_eval={profile['n_eval']:.0f} z_span={profile['z_span']:.1f}")
        preds, meta = predict_variants(hw, tw, variants, cfg)
        for variant, pred in preds.items():
            y_pred = pred[eval_mask]
            rows.append(
                {
                    "well": wid,
                    "variant": variant,
                    "rmse": rmse(y_true, y_pred),
                    "bias": float(np.mean(y_pred - y_true)),
                    "trend_true": float(y_true[-1] - y_true[0]),
                    "trend_pred": float(y_pred[-1] - y_pred[0]),
                    **profile,
                    **meta,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    details = pd.DataFrame(rows)
    summary = summarize(details)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    slim = details.drop(columns=["y_true", "y_pred"])
    slim.to_csv(args.detail_output, index=False)
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.summary_output}")
    print(f"Saved {args.detail_output}")


if __name__ == "__main__":
    main()
