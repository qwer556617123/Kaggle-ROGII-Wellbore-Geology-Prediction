"""
ROGII PF-family exploratory notebook.

Variants:
  - path_rerank: PF path library with full-sequence reranking.
  - event_beam: event-weighted beam ensemble with PF fallback blend.
  - uncertainty_selector: PF/beam/hold selector driven by uncertainty.

This is intentionally targetless at test time and notebook-submission friendly.
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")

VARIANT = os.getenv("ROGII_VARIANT", "bin_less_aggressive")
N_PARTICLES = int(os.getenv("ROGII_N_PARTICLES", "160"))
N_SEEDS = int(os.getenv("ROGII_N_SEEDS", "64"))
PF_SCALES = tuple(float(x) for x in os.getenv("ROGII_PF_SCALES", "3,5,8,12").split(","))
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
if not OUTPUT_PATH.parent.exists():
    OUTPUT_PATH = Path("submission.csv")
USE_VISIBLE_PHYSICAL = os.getenv("ROGII_USE_VISIBLE_PHYSICAL", "1") == "1"

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


def find_input_dir() -> str:
    for candidate in [
        "/kaggle/input/rogii-wellbore-geology-prediction",
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        ".",
    ]:
        if os.path.isdir(os.path.join(candidate, "train")) and os.path.isdir(os.path.join(candidate, "test")):
            print(f"INPUT_DIR={candidate}")
            return candidate
    hits = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    if hits:
        root = os.path.dirname(hits[0])
        print(f"Discovered INPUT_DIR={root}")
        return root
    raise FileNotFoundError("Cannot locate competition data")


INPUT_DIR = find_input_dir()
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
TEST_WELLS = [
    os.path.basename(path).split("__")[0]
    for path in sorted(glob.glob(os.path.join(TEST_DIR, "*__horizontal_well.csv")))
]


def load_well(wid: str, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv"))
    return hw, tw


def interpolate_gr(hw: pd.DataFrame, fallback: float) -> np.ndarray:
    return hw["GR"].interpolate(limit_direction="both").fillna(fallback).to_numpy(dtype=float)


def typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(dtype=float)
    return tw_tvt, tw_gr


def tvt_from_contacts(hw_tr: pd.DataFrame, tw_tr: pd.DataFrame, ref_col: str = "EGFDU") -> pd.Series:
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


def run_particle_filter(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = N_PARTICLES,
    seed: int = 42,
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
    n = n_particles
    pos = last_tvt + last_z + 2.0 * rng.standard_normal(n)
    rate = init_rate + 0.01 * rng.standard_normal(n)
    weights = np.ones(n) / n

    momentum = 0.998
    velocity_noise = 0.002
    position_noise = 0.005
    resample_pos_noise = 0.1
    resample_rate_noise = 0.001
    resample_threshold = 0.5

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
        rate = momentum * rate + velocity_noise * rng.standard_normal(n)
        pos = pos + rate * dm_step + position_noise * rng.standard_normal(n)
        tvt_particles = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_particles + z_v[i]

        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        residual = (gr_v[i] - expected_gr) / gr_sigma
        likelihood = np.exp(-0.5 * np.minimum(residual**2, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        avg_likelihood = float((weights * likelihood).sum())
        log_lik += np.log(max(avg_likelihood, 1e-300))
        weights = weights * likelihood
        w_sum = weights.sum()
        weights = weights / w_sum if w_sum > 0 else np.ones(n) / n

        entropy = float(-np.sum(weights * np.log(weights + 1e-300)) / np.log(n))
        entropies.append(entropy)
        n_eff = 1.0 / np.sum(weights**2)
        if n_eff < resample_threshold * n:
            cum = np.cumsum(weights)
            u0 = rng.uniform(0, 1.0 / n)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)
            pos = pos[idx] + resample_pos_noise * rng.standard_normal(n)
            rate = rate[idx] + resample_rate_noise * rng.standard_normal(n)
            weights = np.ones(n) / n
            n_resample += 1

        pred[i] = float(np.dot(weights, pos - z_v[i]))
        prev_md = md_v[i]

    out[list(eval_rows.index)] = pred
    meta = {"entropy": float(np.mean(entropies)) if entropies else 0.0, "n_resample": n_resample}
    return (out, log_lik, meta) if return_meta else (out, log_lik)


def pf_path_library(hw: pd.DataFrame, tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    paths = []
    log_liks = []
    entropies = []
    for seed in range(N_SEEDS):
        path, ll, meta = run_particle_filter(hw, tw, seed=seed, return_meta=True)
        paths.append(path)
        log_liks.append(ll)
        entropies.append(meta["entropy"])
    return np.stack(paths, axis=0), np.asarray(log_liks), {"entropy": float(np.mean(entropies))}


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


def run_path_rerank(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    paths, online_ll, meta = pf_path_library(hw, tw)
    full_scores = np.array([score_path_full_sequence(path, hw, tw) for path in paths])
    combined = 0.35 * (online_ll - online_ll.max()) / max(1.0, abs(online_ll.std())) + full_scores
    top_k = min(32, len(paths))
    top = np.argsort(combined)[-top_k:]
    weights = np.exp(combined[top] - combined[top].max())
    weights /= weights.sum()
    pred = np.sum(paths[top] * weights[:, None], axis=0)
    print(f"  path_rerank entropy={meta['entropy']:.3f} top_score={combined[top].max():.3f}")
    return pred


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


def run_beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame, event_weighted: bool = False) -> np.ndarray:
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
    beam_mean = np.stack(paths, axis=0).mean(axis=0)
    out = hw["TVT_input"].to_numpy(dtype=float).copy()
    out[list(eval_rows.index)] = beam_mean
    return out


def run_pf_scales(hw: pd.DataFrame, tw: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    paths, log_liks, meta = pf_path_library(hw, tw)
    centered = log_liks - log_liks.max()
    out = {}
    for scale in PF_SCALES:
        weights = np.exp(centered / scale)
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = np.sum(paths * weights[:, None], axis=0)
    out["pf_mean"] = paths.mean(axis=0)
    meta["path_std"] = float(np.mean(np.std(paths[:, hw["TVT_input"].isna().to_numpy()], axis=0)))
    return out, meta


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


def run_public_selector(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    pf_by_scale, meta = run_pf_scales(hw, tw)
    beam = run_beam_ensemble(hw, tw, event_weighted=True)
    code, base_variant, n_eval, z_span = selector_well_code(hw)
    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])
    pred = apply_selector_variant(base_variant, pf_by_scale, beam, last_known_tvt)
    print(
        f"  public_selector code={code} base={base_variant} n_eval={n_eval:.0f} "
        f"z_span={z_span:.2f} entropy={meta['entropy']:.3f} path_std={meta['path_std']:.2f}"
    )
    return pred


def run_grid_selector(hw: pd.DataFrame, tw: pd.DataFrame, variant: str) -> np.ndarray:
    scale, beam_weight, hold_weight = parse_grid_variant(variant)
    pf_by_scale, meta = run_pf_scales(hw, tw)
    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])
    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale["pf_mean"])
    if beam_weight > 0:
        beam = run_beam_ensemble(hw, tw, event_weighted=True)
        pred = (1.0 - beam_weight) * base + beam_weight * beam
    else:
        pred = base
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    print(
        f"  grid_selector scale={scale:g} beam={beam_weight:.3f} hold={hold_weight:.3f} "
        f"entropy={meta['entropy']:.3f} path_std={meta['path_std']:.2f}"
    )
    return pred


def run_bin_selector(hw: pd.DataFrame, tw: pd.DataFrame, variant: str) -> np.ndarray:
    code, _, n_eval, z_span = selector_well_code(hw)
    selected = BIN_SELECTOR_VARIANTS[variant].get(code, "grid_s3_b0_h0p2")
    print(f"  bin_selector variant={variant} code={code} selected={selected} n_eval={n_eval:.0f} z_span={z_span:.2f}")
    return run_grid_selector(hw, tw, selected)


def run_uncertainty_selector(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    pf_by_scale, meta = run_pf_scales(hw, tw)
    beam = run_beam_ensemble(hw, tw, event_weighted=True)
    code, base_variant, n_eval, z_span = selector_well_code(hw)
    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])
    base = apply_selector_variant(base_variant, pf_by_scale, beam, last_known_tvt)

    eval_mask = hw["TVT_input"].isna().to_numpy()
    pf_std = meta["path_std"]
    entropy = meta["entropy"]
    beam_delta = float(np.mean(np.abs(base[eval_mask] - beam[eval_mask]))) if eval_mask.any() else 0.0
    gr_nan = float(hw.loc[eval_mask, "GR"].isna().mean()) if eval_mask.any() else 0.0

    hold_weight = np.clip(0.04 + 0.002 * pf_std + 0.08 * gr_nan, 0.04, 0.28)
    beam_weight = np.clip(0.04 + 0.003 * beam_delta + 0.08 * (1.0 - entropy), 0.04, 0.35)
    stable_pf = pf_by_scale.get("pf_scale_8", pf_by_scale["pf_mean"])
    pred = (1.0 - beam_weight) * stable_pf + beam_weight * beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    print(
        f"  uncertainty code={code} base={base_variant} n_eval={n_eval:.0f} z_span={z_span:.2f} "
        f"pf_std={pf_std:.2f} entropy={entropy:.3f} beam_delta={beam_delta:.2f} "
        f"gr_nan={gr_nan:.3f} hold={hold_weight:.3f} beam={beam_weight:.3f}"
    )
    return pred


def predict_well(wid: str, hw_te: pd.DataFrame, tw_te: pd.DataFrame, train_wids: set[str]) -> np.ndarray:
    hw_tr = None
    tw_tr = None
    tvt_phys = None
    if USE_VISIBLE_PHYSICAL and wid in train_wids:
        try:
            hw_tr, tw_tr = load_well(wid, "train")
            hw_te = hw_te.copy()
            hw_te["TVT_input"] = hw_tr["TVT_input"].to_numpy()
            tvt_phys = tvt_from_contacts(hw_tr, tw_tr).to_numpy(dtype=float)
            print("  physical visible-well model OK")
        except Exception as exc:
            print(f"  physical model skipped: {exc}")
            tvt_phys = None

    tw_ref = tw_tr if tw_tr is not None else tw_te
    if tvt_phys is not None:
        return tvt_phys
    if VARIANT == "path_rerank":
        return run_path_rerank(hw_te, tw_ref)
    if VARIANT == "public_selector":
        return run_public_selector(hw_te, tw_ref)
    if VARIANT == "event_beam":
        beam = run_beam_ensemble(hw_te, tw_ref, event_weighted=True)
        pf_by_scale, meta = run_pf_scales(hw_te, tw_ref)
        last_known_tvt = float(hw_te["TVT_input"].dropna().iloc[-1])
        pred = 0.72 * beam + 0.20 * pf_by_scale.get("pf_scale_8", pf_by_scale["pf_mean"]) + 0.08 * last_known_tvt
        print(f"  event_beam entropy={meta['entropy']:.3f} path_std={meta['path_std']:.2f}")
        return pred
    if VARIANT == "uncertainty_selector":
        return run_uncertainty_selector(hw_te, tw_ref)
    if VARIANT.startswith("grid_s"):
        return run_grid_selector(hw_te, tw_ref, VARIANT)
    if VARIANT in BIN_SELECTOR_VARIANTS:
        return run_bin_selector(hw_te, tw_ref, VARIANT)
    raise ValueError(f"Unknown ROGII_VARIANT={VARIANT}")


print(
    f"VARIANT={VARIANT} N_SEEDS={N_SEEDS} N_PARTICLES={N_PARTICLES} "
    f"PF_SCALES={PF_SCALES} USE_VISIBLE_PHYSICAL={USE_VISIBLE_PHYSICAL}"
)
print(f"TEST_WELLS={TEST_WELLS}")
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
sample["well"] = sample["id"].str[:8]
sample["row_idx"] = sample["id"].str[9:].astype(int)
train_wids = {
    os.path.basename(path).split("__")[0]
    for path in glob.glob(os.path.join(TRAIN_DIR, "*__horizontal_well.csv"))
}

rows = []
for wid in TEST_WELLS:
    print(f"\nProcessing {wid}...")
    hw_te, tw_te = load_well(wid, "test")
    tvt_pred = predict_well(wid, hw_te, tw_te, train_wids)
    well_sample = sample[sample["well"] == wid]
    for _, row in well_sample.iterrows():
        idx = int(row["row_idx"])
        rows.append({"id": row["id"], "tvt": float(tvt_pred[idx])})
    eval_vals = np.array([rows[-len(well_sample) + i]["tvt"] for i in range(len(well_sample))])
    print(f"  Added {len(well_sample)} rows pred=[{eval_vals.min():.2f}, {eval_vals.max():.2f}] trend={eval_vals[-1] - eval_vals[0]:.2f}")

submission = pd.DataFrame(rows)
submission = sample[["id"]].merge(submission, on="id", how="left")
if submission["tvt"].isna().any():
    raise RuntimeError("NaN predictions in submission.")
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {OUTPUT_PATH} rows={len(submission)}")
print(submission.groupby(submission["id"].str[:8])["tvt"].agg(["count", "min", "max", "mean"]).to_string())
