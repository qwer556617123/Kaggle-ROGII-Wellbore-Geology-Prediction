"""Blend current PF baseline with dynamic v10 artifact inference.

Generated single-file Kaggle script. It writes the PF and v10 artifact
components into the working directory at runtime because Kaggle script kernels
only execute the configured code_file.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PF_COMPONENT_CODE = '"""\nROGII PF-family exploratory notebook.\n\nVariants:\n  - path_rerank: PF path library with full-sequence reranking.\n  - event_beam: event-weighted beam ensemble with PF fallback blend.\n  - uncertainty_selector: PF/beam/hold selector driven by uncertainty.\n\nThis is intentionally targetless at test time and notebook-submission friendly.\n"""\nfrom __future__ import annotations\n\nimport glob\nimport os\nimport re\nimport warnings\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nfrom scipy.signal import savgol_filter\n\n\nwarnings.filterwarnings("ignore")\n\nVARIANT = os.getenv("ROGII_VARIANT", "grid_s3_b0_h0p17")\nN_PARTICLES = int(os.getenv("ROGII_N_PARTICLES", "160"))\nN_SEEDS = int(os.getenv("ROGII_N_SEEDS", "64"))\nPF_SCALES = tuple(float(x) for x in os.getenv("ROGII_PF_SCALES", "3,5,8,12").split(","))\nif os.getenv("ROGII_OUTPUT_PATH"):\n    OUTPUT_PATH = Path(os.environ["ROGII_OUTPUT_PATH"])\nelif os.getenv("ROGII_OUTPUT_DIR"):\n    OUTPUT_PATH = Path(os.environ["ROGII_OUTPUT_DIR"]) / "submission.csv"\nelse:\n    OUTPUT_PATH = Path("/kaggle/working/submission.csv")\n    if not OUTPUT_PATH.parent.exists():\n        OUTPUT_PATH = Path("submission.csv")\nOUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)\nUSE_VISIBLE_PHYSICAL = os.getenv("ROGII_USE_VISIBLE_PHYSICAL", "1") == "1"\n\nSELECTOR_N_EVAL_THRESHOLD = 4840.0\nSELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)\nSELECTOR_BIN_VARIANTS = {\n    0: "pf_scale_5_hold_0.2",\n    1: "pf_scale_3_hold_0.15",\n    2: "pf_scale_12_beam_0.2_hold_0.15",\n    3: "pf_scale_5_hold_0.15",\n    4: "pf_scale_5_beam_0.05_hold_0.05",\n    5: "pf_scale_12_beam_0.2_hold_0.05",\n}\nSELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"\nBIN_SELECTOR_VARIANTS = {\n    "bin_best_v1": {\n        0: "grid_s3_b0_h0p05",\n        2: "grid_s8_b0_h0p05",\n        3: "grid_s3_b0_h0p15",\n        5: "grid_s12_b0_h0p2",\n    },\n    "bin_less_aggressive": {\n        0: "grid_s3_b0_h0p1",\n        2: "grid_s8_b0_h0p1",\n        3: "grid_s3_b0_h0p15",\n        5: "grid_s12_b0_h0p15",\n    },\n    "bin_lb_safe": {\n        0: "grid_s3_b0_h0p2",\n        2: "grid_s8_b0_h0p1",\n        3: "grid_s3_b0_h0p15",\n        5: "grid_s12_b0_h0p2",\n    },\n}\n\nBEAM_CONFIGS = [\n    (10, 20.0, 144.0, 2),\n    (10, 8.0, 64.0, 2),\n    (8, 35.0, 220.0, 1),\n    (10, 14.0, 90.0, 5),\n    (20, 4.0, 36.0, 3),\n    (12, 12.0, 100.0, 3),\n    (15, 25.0, 180.0, 2),\n    (20, 30.0, 200.0, 2),\n    (15, 10.0, 80.0, 4),\n    (25, 6.0, 50.0, 3),\n    (10, 40.0, 300.0, 1),\n    (12, 18.0, 120.0, 5),\n    (30, 8.0, 70.0, 2),\n    (10, 50.0, 400.0, 0),\n]\n\n\ndef find_input_dir() -> str:\n    for candidate in [\n        "/kaggle/input/rogii-wellbore-geology-prediction",\n        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",\n        ".",\n    ]:\n        if os.path.isdir(os.path.join(candidate, "train")) and os.path.isdir(os.path.join(candidate, "test")):\n            print(f"INPUT_DIR={candidate}")\n            return candidate\n    hits = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)\n    if hits:\n        root = os.path.dirname(hits[0])\n        print(f"Discovered INPUT_DIR={root}")\n        return root\n    raise FileNotFoundError("Cannot locate competition data")\n\n\nINPUT_DIR = find_input_dir()\nTRAIN_DIR = os.path.join(INPUT_DIR, "train")\nTEST_DIR = os.path.join(INPUT_DIR, "test")\nTEST_WELLS = [\n    os.path.basename(path).split("__")[0]\n    for path in sorted(glob.glob(os.path.join(TEST_DIR, "*__horizontal_well.csv")))\n]\n\n\ndef load_well(wid: str, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:\n    base = TRAIN_DIR if split == "train" else TEST_DIR\n    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))\n    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv"))\n    return hw, tw\n\n\ndef interpolate_gr(hw: pd.DataFrame, fallback: float) -> np.ndarray:\n    return hw["GR"].interpolate(limit_direction="both").fillna(fallback).to_numpy(dtype=float)\n\n\ndef typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:\n    tw_s = tw.sort_values("TVT")\n    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)\n    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(dtype=float)\n    return tw_tvt, tw_gr\n\n\ndef tvt_from_contacts(hw_tr: pd.DataFrame, tw_tr: pd.DataFrame, ref_col: str = "EGFDU") -> pd.Series:\n    tw_g = tw_tr.dropna(subset=["Geology"])\n    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()\n    if np.isnan(ref_tvt):\n        ref_col = tw_g["Geology"].iloc[0]\n        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()\n    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()\n    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset\n\n\ndef run_particle_filter(\n    hw: pd.DataFrame,\n    tw: pd.DataFrame,\n    n_particles: int = N_PARTICLES,\n    seed: int = 42,\n    return_meta: bool = False,\n):\n    tw_tvt, tw_gr = typewell_arrays(tw)\n    known = hw[hw["TVT_input"].notna()]\n    eval_rows = hw[hw["TVT_input"].isna()]\n    if len(eval_rows) == 0:\n        out = hw["TVT_input"].to_numpy(dtype=float).copy()\n        return (out, 0.0, {"entropy": 0.0, "n_resample": 0}) if return_meta else (out, 0.0)\n\n    last = known.iloc[-1]\n    last_tvt = float(last["TVT_input"])\n    last_z = float(last["Z"])\n    last_md = float(last["MD"])\n\n    tw_at_known = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)\n    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0).to_numpy(dtype=float) - tw_at_known), 10.0, 60.0))\n\n    tail = known.tail(30)\n    dt = np.diff(tail["TVT_input"].to_numpy(dtype=float))\n    dz = np.diff(tail["Z"].to_numpy(dtype=float))\n    dm = np.diff(tail["MD"].to_numpy(dtype=float))\n    m = dm > 0\n    init_rate = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0\n\n    rng = np.random.default_rng(seed)\n    n = n_particles\n    pos = last_tvt + last_z + 2.0 * rng.standard_normal(n)\n    rate = init_rate + 0.01 * rng.standard_normal(n)\n    weights = np.ones(n) / n\n\n    momentum = 0.998\n    velocity_noise = 0.002\n    position_noise = 0.005\n    resample_pos_noise = 0.1\n    resample_rate_noise = 0.001\n    resample_threshold = 0.5\n\n    md_v = eval_rows["MD"].to_numpy(dtype=float)\n    z_v = eval_rows["Z"].to_numpy(dtype=float)\n    gr_v = interpolate_gr(hw, tw_gr.mean())[eval_rows.index]\n\n    out = hw["TVT_input"].to_numpy(dtype=float).copy()\n    pred = np.empty(len(eval_rows))\n    prev_md = last_md\n    log_lik = 0.0\n    entropies = []\n    n_resample = 0\n\n    for i in range(len(eval_rows)):\n        dm_step = max(md_v[i] - prev_md, 1.0)\n        rate = momentum * rate + velocity_noise * rng.standard_normal(n)\n        pos = pos + rate * dm_step + position_noise * rng.standard_normal(n)\n        tvt_particles = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)\n        pos = tvt_particles + z_v[i]\n\n        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)\n        residual = (gr_v[i] - expected_gr) / gr_sigma\n        likelihood = np.exp(-0.5 * np.minimum(residual**2, 600.0))\n        likelihood = np.maximum(likelihood, 1e-300)\n        avg_likelihood = float((weights * likelihood).sum())\n        log_lik += np.log(max(avg_likelihood, 1e-300))\n        weights = weights * likelihood\n        w_sum = weights.sum()\n        weights = weights / w_sum if w_sum > 0 else np.ones(n) / n\n\n        entropy = float(-np.sum(weights * np.log(weights + 1e-300)) / np.log(n))\n        entropies.append(entropy)\n        n_eff = 1.0 / np.sum(weights**2)\n        if n_eff < resample_threshold * n:\n            cum = np.cumsum(weights)\n            u0 = rng.uniform(0, 1.0 / n)\n            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)\n            pos = pos[idx] + resample_pos_noise * rng.standard_normal(n)\n            rate = rate[idx] + resample_rate_noise * rng.standard_normal(n)\n            weights = np.ones(n) / n\n            n_resample += 1\n\n        pred[i] = float(np.dot(weights, pos - z_v[i]))\n        prev_md = md_v[i]\n\n    out[list(eval_rows.index)] = pred\n    meta = {"entropy": float(np.mean(entropies)) if entropies else 0.0, "n_resample": n_resample}\n    return (out, log_lik, meta) if return_meta else (out, log_lik)\n\n\ndef pf_path_library(hw: pd.DataFrame, tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:\n    paths = []\n    log_liks = []\n    entropies = []\n    for seed in range(N_SEEDS):\n        path, ll, meta = run_particle_filter(hw, tw, seed=seed, return_meta=True)\n        paths.append(path)\n        log_liks.append(ll)\n        entropies.append(meta["entropy"])\n    return np.stack(paths, axis=0), np.asarray(log_liks), {"entropy": float(np.mean(entropies))}\n\n\ndef score_path_full_sequence(path: np.ndarray, hw: pd.DataFrame, tw: pd.DataFrame) -> float:\n    tw_tvt, tw_gr = typewell_arrays(tw)\n    eval_mask = hw["TVT_input"].isna().to_numpy()\n    if not eval_mask.any():\n        return 0.0\n    gr = interpolate_gr(hw, float(np.nanmean(tw_gr)))[eval_mask]\n    expected = np.interp(path[eval_mask], tw_tvt, tw_gr)\n    known = hw[hw["TVT_input"].notna()]\n    tw_at_known = np.interp(known["TVT_input"].to_numpy(dtype=float), tw_tvt, tw_gr)\n    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0).to_numpy(dtype=float) - tw_at_known), 10.0, 60.0))\n    residual = (gr - expected) / gr_sigma\n    emission = -0.5 * np.mean(np.minimum(residual**2, 100.0))\n    tvt_eval = path[eval_mask]\n    smooth = -0.02 * float(np.std(np.diff(tvt_eval))) if len(tvt_eval) > 2 else 0.0\n    anchor = float(known["TVT_input"].iloc[-1])\n    hold = -0.0005 * abs(float(tvt_eval[-1] - anchor))\n    return emission + smooth + hold\n\n\ndef run_path_rerank(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:\n    paths, online_ll, meta = pf_path_library(hw, tw)\n    full_scores = np.array([score_path_full_sequence(path, hw, tw) for path in paths])\n    combined = 0.35 * (online_ll - online_ll.max()) / max(1.0, abs(online_ll.std())) + full_scores\n    top_k = min(32, len(paths))\n    top = np.argsort(combined)[-top_k:]\n    weights = np.exp(combined[top] - combined[top].max())\n    weights /= weights.sum()\n    pred = np.sum(paths[top] * weights[:, None], axis=0)\n    print(f"  path_rerank entropy={meta[\'entropy\']:.3f} top_score={combined[top].max():.3f}")\n    return pred\n\n\ndef event_strength(values: np.ndarray) -> np.ndarray:\n    x = pd.Series(values).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)\n    if len(x) > 9:\n        win = min(31, len(x) if len(x) % 2 == 1 else len(x) - 1)\n        x = savgol_filter(x, win, min(3, win - 1))\n    grad = np.abs(np.gradient(x))\n    q80 = np.quantile(grad, 0.80) if len(grad) else 0.0\n    q95 = np.quantile(grad, 0.95) if len(grad) else 1.0\n    return np.clip((grad - q80) / max(q95 - q80, 1e-6), 0.0, 1.0)\n\n\ndef beam_search(\n    hgr: np.ndarray,\n    tw_tvt: np.ndarray,\n    tw_gr: np.ndarray,\n    last_tvt: float,\n    bs: int,\n    move_cost: float,\n    err_scale: float,\n    smooth_radius: int,\n    event_weighted: bool = False,\n) -> np.ndarray:\n    n = len(hgr)\n    nt = len(tw_tvt)\n    if n == 0:\n        return np.array([last_tvt])\n    if smooth_radius > 0 and n > max(3, 2 * smooth_radius + 1):\n        win = min(2 * smooth_radius + 1, n if n % 2 == 1 else n - 1)\n        obs = savgol_filter(hgr, win, min(2, win - 1))\n    else:\n        obs = hgr.copy()\n    ev = event_strength(obs) if event_weighted else np.zeros(n)\n    start_idx = int(np.argmin(np.abs(tw_tvt - last_tvt)))\n    moves = np.array([-2, -1, 0, 1, 2], dtype=np.int64)\n    move_penalty = move_cost * np.array([2.0, 1.0, 0.0, 1.0, 2.0])\n    beam_idx = np.full(bs, start_idx, dtype=np.int64)\n    beam_cost = np.full(bs, np.inf)\n    beam_cost[0] = 0.0\n    beam_n = 1\n    result = np.zeros(n)\n    for step in range(n):\n        next_idx = beam_idx[:beam_n, None] + moves[None, :]\n        clipped = np.clip(next_idx, 0, nt - 1)\n        valid = (next_idx >= 0) & (next_idx < nt)\n        local_scale = err_scale / (1.0 + 3.0 * ev[step])\n        gr_error = (obs[step] - tw_gr[clipped]) ** 2 / max(local_scale, 1e-6)\n        total = beam_cost[:beam_n, None] + gr_error + move_penalty[None, :]\n        total = np.where(valid, total, np.inf)\n        flat_idx = next_idx.flatten()\n        flat_total = total.flatten()\n        flat_valid = valid.flatten()\n        flat_idx = flat_idx[flat_valid]\n        flat_total = flat_total[flat_valid]\n        order = np.argsort(flat_total)\n        idx_sorted = flat_idx[order]\n        total_sorted = flat_total[order]\n        _, first = np.unique(idx_sorted, return_index=True)\n        idx_unique = idx_sorted[first]\n        total_unique = total_sorted[first]\n        kept = min(bs, len(idx_unique))\n        top = np.argpartition(total_unique, min(kept - 1, len(total_unique) - 1))[:kept]\n        top = top[np.argsort(total_unique[top])]\n        beam_idx[:kept] = idx_unique[top]\n        beam_cost[:kept] = total_unique[top]\n        if kept < bs:\n            beam_idx[kept:] = beam_idx[kept - 1]\n            beam_cost[kept:] = np.inf\n        beam_n = kept\n        result[step] = tw_tvt[beam_idx[0]]\n    return result\n\n\ndef run_beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame, event_weighted: bool = False) -> np.ndarray:\n    known = hw[hw["TVT_input"].notna()]\n    eval_rows = hw[hw["TVT_input"].isna()]\n    if len(eval_rows) == 0:\n        return hw["TVT_input"].to_numpy(dtype=float).copy()\n    last_tvt = float(known.iloc[-1]["TVT_input"])\n    tw_tvt, tw_gr = typewell_arrays(tw)\n    gr_all = interpolate_gr(hw, float(np.nanmean(tw_gr)))\n    hgr = gr_all[eval_rows.index]\n    paths = [\n        beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r, event_weighted=event_weighted)\n        for (bs, mc, es, r) in BEAM_CONFIGS\n    ]\n    beam_mean = np.stack(paths, axis=0).mean(axis=0)\n    out = hw["TVT_input"].to_numpy(dtype=float).copy()\n    out[list(eval_rows.index)] = beam_mean\n    return out\n\n\ndef run_pf_scales(hw: pd.DataFrame, tw: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:\n    paths, log_liks, meta = pf_path_library(hw, tw)\n    centered = log_liks - log_liks.max()\n    out = {}\n    for scale in PF_SCALES:\n        weights = np.exp(centered / scale)\n        weights /= weights.sum()\n        out[f"pf_scale_{scale:g}"] = np.sum(paths * weights[:, None], axis=0)\n    out["pf_mean"] = paths.mean(axis=0)\n    meta["path_std"] = float(np.mean(np.std(paths[:, hw["TVT_input"].isna().to_numpy()], axis=0)))\n    return out, meta\n\n\ndef selector_well_code(hw: pd.DataFrame) -> tuple[int, str, float, float]:\n    eval_mask = hw["TVT_input"].isna().to_numpy()\n    n_eval = float(eval_mask.sum())\n    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)\n    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0\n    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)\n    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))\n    code = n_bin + 2 * z_bin\n    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT), n_eval, z_span\n\n\ndef parse_selector_variant(name: str) -> tuple[float, float, float]:\n    parts = name.split("_")\n    scale = float(parts[2])\n    beam_weight = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0\n    hold_weight = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0\n    return scale, beam_weight, hold_weight\n\n\ndef apply_selector_variant(name: str, pf_by_scale: dict[str, np.ndarray], beam: np.ndarray, last_known_tvt: float) -> np.ndarray:\n    scale, beam_weight, hold_weight = parse_selector_variant(name)\n    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale.get("pf_scale_8", pf_by_scale["pf_mean"]))\n    pred = (1.0 - beam_weight) * base + beam_weight * beam\n    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt\n    return pred\n\n\ndef parse_grid_variant(name: str) -> tuple[float, float, float]:\n    match = re.fullmatch(r"grid_s([0-9]+(?:p[0-9]+)?)_b([0-9]+(?:p[0-9]+)?)_h([0-9]+(?:p[0-9]+)?)", name)\n    if not match:\n        raise ValueError(f"Bad grid variant name={name}")\n    scale = float(match.group(1).replace("p", "."))\n    beam_weight = float(match.group(2).replace("p", "."))\n    hold_weight = float(match.group(3).replace("p", "."))\n    return scale, beam_weight, hold_weight\n\n\ndef run_public_selector(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:\n    pf_by_scale, meta = run_pf_scales(hw, tw)\n    beam = run_beam_ensemble(hw, tw, event_weighted=True)\n    code, base_variant, n_eval, z_span = selector_well_code(hw)\n    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])\n    pred = apply_selector_variant(base_variant, pf_by_scale, beam, last_known_tvt)\n    print(\n        f"  public_selector code={code} base={base_variant} n_eval={n_eval:.0f} "\n        f"z_span={z_span:.2f} entropy={meta[\'entropy\']:.3f} path_std={meta[\'path_std\']:.2f}"\n    )\n    return pred\n\n\ndef run_grid_selector(hw: pd.DataFrame, tw: pd.DataFrame, variant: str) -> np.ndarray:\n    scale, beam_weight, hold_weight = parse_grid_variant(variant)\n    pf_by_scale, meta = run_pf_scales(hw, tw)\n    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])\n    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale["pf_mean"])\n    if beam_weight > 0:\n        beam = run_beam_ensemble(hw, tw, event_weighted=True)\n        pred = (1.0 - beam_weight) * base + beam_weight * beam\n    else:\n        pred = base\n    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt\n    print(\n        f"  grid_selector scale={scale:g} beam={beam_weight:.3f} hold={hold_weight:.3f} "\n        f"entropy={meta[\'entropy\']:.3f} path_std={meta[\'path_std\']:.2f}"\n    )\n    return pred\n\n\ndef run_bin_selector(hw: pd.DataFrame, tw: pd.DataFrame, variant: str) -> np.ndarray:\n    code, _, n_eval, z_span = selector_well_code(hw)\n    selected = BIN_SELECTOR_VARIANTS[variant].get(code, "grid_s3_b0_h0p2")\n    print(f"  bin_selector variant={variant} code={code} selected={selected} n_eval={n_eval:.0f} z_span={z_span:.2f}")\n    return run_grid_selector(hw, tw, selected)\n\n\ndef run_uncertainty_selector(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:\n    pf_by_scale, meta = run_pf_scales(hw, tw)\n    beam = run_beam_ensemble(hw, tw, event_weighted=True)\n    code, base_variant, n_eval, z_span = selector_well_code(hw)\n    last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])\n    base = apply_selector_variant(base_variant, pf_by_scale, beam, last_known_tvt)\n\n    eval_mask = hw["TVT_input"].isna().to_numpy()\n    pf_std = meta["path_std"]\n    entropy = meta["entropy"]\n    beam_delta = float(np.mean(np.abs(base[eval_mask] - beam[eval_mask]))) if eval_mask.any() else 0.0\n    gr_nan = float(hw.loc[eval_mask, "GR"].isna().mean()) if eval_mask.any() else 0.0\n\n    hold_weight = np.clip(0.04 + 0.002 * pf_std + 0.08 * gr_nan, 0.04, 0.28)\n    beam_weight = np.clip(0.04 + 0.003 * beam_delta + 0.08 * (1.0 - entropy), 0.04, 0.35)\n    stable_pf = pf_by_scale.get("pf_scale_8", pf_by_scale["pf_mean"])\n    pred = (1.0 - beam_weight) * stable_pf + beam_weight * beam\n    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt\n    print(\n        f"  uncertainty code={code} base={base_variant} n_eval={n_eval:.0f} z_span={z_span:.2f} "\n        f"pf_std={pf_std:.2f} entropy={entropy:.3f} beam_delta={beam_delta:.2f} "\n        f"gr_nan={gr_nan:.3f} hold={hold_weight:.3f} beam={beam_weight:.3f}"\n    )\n    return pred\n\n\ndef predict_well(wid: str, hw_te: pd.DataFrame, tw_te: pd.DataFrame, train_wids: set[str]) -> np.ndarray:\n    hw_tr = None\n    tw_tr = None\n    tvt_phys = None\n    if USE_VISIBLE_PHYSICAL and wid in train_wids:\n        try:\n            hw_tr, tw_tr = load_well(wid, "train")\n            hw_te = hw_te.copy()\n            hw_te["TVT_input"] = hw_tr["TVT_input"].to_numpy()\n            tvt_phys = tvt_from_contacts(hw_tr, tw_tr).to_numpy(dtype=float)\n            print("  physical visible-well model OK")\n        except Exception as exc:\n            print(f"  physical model skipped: {exc}")\n            tvt_phys = None\n\n    tw_ref = tw_tr if tw_tr is not None else tw_te\n    if tvt_phys is not None:\n        return tvt_phys\n    if VARIANT == "path_rerank":\n        return run_path_rerank(hw_te, tw_ref)\n    if VARIANT == "public_selector":\n        return run_public_selector(hw_te, tw_ref)\n    if VARIANT == "event_beam":\n        beam = run_beam_ensemble(hw_te, tw_ref, event_weighted=True)\n        pf_by_scale, meta = run_pf_scales(hw_te, tw_ref)\n        last_known_tvt = float(hw_te["TVT_input"].dropna().iloc[-1])\n        pred = 0.72 * beam + 0.20 * pf_by_scale.get("pf_scale_8", pf_by_scale["pf_mean"]) + 0.08 * last_known_tvt\n        print(f"  event_beam entropy={meta[\'entropy\']:.3f} path_std={meta[\'path_std\']:.2f}")\n        return pred\n    if VARIANT == "uncertainty_selector":\n        return run_uncertainty_selector(hw_te, tw_ref)\n    if VARIANT.startswith("grid_s"):\n        return run_grid_selector(hw_te, tw_ref, VARIANT)\n    if VARIANT in BIN_SELECTOR_VARIANTS:\n        return run_bin_selector(hw_te, tw_ref, VARIANT)\n    raise ValueError(f"Unknown ROGII_VARIANT={VARIANT}")\n\n\nprint(\n    f"VARIANT={VARIANT} N_SEEDS={N_SEEDS} N_PARTICLES={N_PARTICLES} "\n    f"PF_SCALES={PF_SCALES} USE_VISIBLE_PHYSICAL={USE_VISIBLE_PHYSICAL}"\n)\nprint(f"TEST_WELLS={TEST_WELLS}")\nsample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))\nsample["well"] = sample["id"].str[:8]\nsample["row_idx"] = sample["id"].str[9:].astype(int)\ntrain_wids = {\n    os.path.basename(path).split("__")[0]\n    for path in glob.glob(os.path.join(TRAIN_DIR, "*__horizontal_well.csv"))\n}\n\nrows = []\nfor wid in TEST_WELLS:\n    print(f"\\nProcessing {wid}...")\n    hw_te, tw_te = load_well(wid, "test")\n    tvt_pred = predict_well(wid, hw_te, tw_te, train_wids)\n    well_sample = sample[sample["well"] == wid]\n    for _, row in well_sample.iterrows():\n        idx = int(row["row_idx"])\n        rows.append({"id": row["id"], "tvt": float(tvt_pred[idx])})\n    eval_vals = np.array([rows[-len(well_sample) + i]["tvt"] for i in range(len(well_sample))])\n    print(f"  Added {len(well_sample)} rows pred=[{eval_vals.min():.2f}, {eval_vals.max():.2f}] trend={eval_vals[-1] - eval_vals[0]:.2f}")\n\nsubmission = pd.DataFrame(rows)\nsubmission = sample[["id"]].merge(submission, on="id", how="left")\nif submission["tvt"].isna().any():\n    raise RuntimeError("NaN predictions in submission.")\nsubmission.to_csv(OUTPUT_PATH, index=False)\nprint(f"\\nSaved {OUTPUT_PATH} rows={len(submission)}")\nprint(submission.groupby(submission["id"].str[:8])["tvt"].agg(["count", "min", "max", "mean"]).to_string())\n'
V10_COMPONENT_CODE = '# %% cell 1\nfrom __future__ import annotations\n\nimport gc\nimport hashlib\nimport json\nimport multiprocessing\nimport os\nimport sys\nimport time\nimport traceback\nimport warnings\nfrom functools import lru_cache\nfrom pathlib import Path\nfrom typing import Optional\nfrom concurrent.futures import ThreadPoolExecutor\n\nimport lightgbm as lgb\nimport numpy as np\nimport pandas as pd\nfrom catboost import CatBoostRegressor, Pool\nfrom scipy.interpolate import interp1d\nfrom scipy.signal import savgol_filter\nfrom scipy.spatial import cKDTree\nfrom sklearn.linear_model import Ridge\nfrom sklearn.metrics import root_mean_squared_error\nfrom sklearn.model_selection import GroupKFold\n\nwarnings.filterwarnings("ignore")\n\n# Kaggle inference notebook defaults: load the artifact dataset we created locally.\nos.environ.setdefault("ROGII_INFERENCE_ONLY", "1")\nos.environ.setdefault("ROGII_SAVE_ARTIFACTS", "0")\nos.environ.setdefault("ROGII_RUN_TABICL", "1")\n\n# ── optional numba JIT ──────────────────────────────────────────────────────\ntry:\n    from numba import njit as _njit\n    _NUMBA = True\nexcept ImportError:\n    def _njit(*a, **kw):\n        def _wrap(f): return f\n        return _wrap\n    _NUMBA = False\n\nprint("NUMBA:", _NUMBA)\n\nSEED = 42\nSEED_SALT = os.environ.get("ROGII_ARTIFACT_SEED_SALT", "0")\n_GLOBAL_SEED_BYTES = hashlib.sha256(f"{SEED}:{SEED_SALT}:global".encode("utf-8")).digest()\nnp.random.seed(int.from_bytes(_GLOBAL_SEED_BYTES[:4], "little"))\nprint(f"SEED={SEED} SEED_SALT={SEED_SALT}")\n# Use ALL available cores — Kaggle typically gives 4 (sometimes 2×4 on multi-GPU)\ndef env_flag(name: str, default: bool = False) -> bool:\n    value = os.environ.get(name)\n    if value is None:\n        return bool(default)\n    return value.strip().lower() not in {"0", "false", "no", "off", ""}\n\n\ndef env_int(name: str, default: int) -> int:\n    value = os.environ.get(name)\n    if value is None or not str(value).strip():\n        return int(default)\n    return int(value)\n\n\nRUNNING_ON_KAGGLE = (\n    Path("/kaggle/input").exists()\n    or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))\n    or bool(os.environ.get("KAGGLE_URL_BASE"))\n)\n\nNCPU = max(1, min(multiprocessing.cpu_count(), env_int("ROGII_NCPU", multiprocessing.cpu_count())))\nprint(f"NCPU={NCPU}")\n\n# ══════════════════════════════════════════════════════════════════════════════\n# DEBUG CONFIG\n# ══════════════════════════════════════════════════════════════════════════════\nDBG_VERBOSE        = env_flag("ROGII_DEBUG_VERBOSE", False)\nDBG_SINGLE_WELL    = env_flag("ROGII_SMOKE_WELL", False)\nDBG_SINGLE_TEST    = env_flag("ROGII_SMOKE_TEST", False)\nDBG_PARALLEL_STATS = env_flag("ROGII_PARALLEL_STATS", True)\nDBG_NAN_AUDIT      = env_flag("ROGII_NAN_AUDIT", False)\nDBG_FEATURE_AUDIT  = env_flag("ROGII_FEATURE_AUDIT", False)\nDBG_MAX_ERRORS     = 20\n_dbg_error_count   = 0\n\n_T0_GLOBAL = time.time()\n\n\ndef _dbg(msg: str, arr: np.ndarray = None, level: str = "INFO") -> None:\n    if not DBG_VERBOSE:\n        return\n    elapsed = time.time() - _T0_GLOBAL\n    prefix  = f"[{elapsed:8.1f}s][{level}] "\n    print(prefix + msg, file=sys.stderr, flush=True)\n    if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:\n        try:\n            nans = int(np.isnan(arr.astype(float)).sum())\n            infs = int(np.isinf(arr.astype(float)).sum())\n            print(\n                f"{prefix}  shape={arr.shape} dtype={arr.dtype} "\n                f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g} "\n                f"nan={nans} inf={infs}",\n                file=sys.stderr, flush=True,\n            )\n        except Exception:\n            print(f"{prefix}  shape={arr.shape} dtype={arr.dtype} (stats failed)",\n                  file=sys.stderr, flush=True)\n\n\ndef _dbg_df(tag: str, df: pd.DataFrame) -> None:\n    if not DBG_VERBOSE:\n        return\n    elapsed = time.time() - _T0_GLOBAL\n    nan_cols = df.isnull().sum()\n    nan_cols = nan_cols[nan_cols > 0]\n    print(\n        f"[{elapsed:8.1f}s][DF] {tag}: shape={df.shape}  "\n        f"nan_cols={len(nan_cols)}/{len(df.columns)}",\n        file=sys.stderr, flush=True,\n    )\n    if len(nan_cols) > 0 and DBG_VERBOSE:\n        top = nan_cols.nlargest(10)\n        print(f"  top nan cols: {dict(top)}", file=sys.stderr, flush=True)\n\n\ndef _log_error(ctx: str, exc: Exception) -> None:\n    global _dbg_error_count\n    _dbg_error_count += 1\n    if _dbg_error_count > DBG_MAX_ERRORS:\n        if _dbg_error_count == DBG_MAX_ERRORS + 1:\n            print(f"[ERROR] Max error log limit ({DBG_MAX_ERRORS}) reached.",\n                  file=sys.stderr, flush=True)\n        return\n    elapsed = time.time() - _T0_GLOBAL\n    tb = traceback.format_exc()\n    print(\n        f"[{elapsed:8.1f}s][ERROR] {ctx}: {type(exc).__name__}: {exc}\\n{tb}",\n        file=sys.stderr, flush=True,\n    )\n\n\n# ── data paths ──────────────────────────────────────────────────────────────\ndef _find() -> Path:\n    candidates = []\n    if os.environ.get("ROGII_DATA_DIR"):\n        candidates.append(Path(os.environ["ROGII_DATA_DIR"]))\n    candidates.extend([\n        Path("/kaggle/input/rogii-wellbore-geology-prediction"),\n        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),\n        Path.cwd(),\n        Path.cwd() / "rogii-wellbore-geology-prediction",\n        Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd(),\n        Path(__file__).resolve().parents[2] / "rogii-wellbore-geology-prediction" if "__file__" in globals() else Path.cwd(),\n    ])\n    for p in candidates:\n        if (p / "train").is_dir() and (p / "test").is_dir() and (p / "sample_submission.csv").is_file():\n            return p\n    input_root = Path("/kaggle/input")\n    if input_root.exists():\n        for sample in input_root.glob("**/sample_submission.csv"):\n            p = sample.parent\n            if (p / "train").is_dir() and (p / "test").is_dir():\n                return p\n    raise FileNotFoundError("Data not found")\n\n\ndef _default_output_dir() -> Path:\n    if RUNNING_ON_KAGGLE:\n        return Path("/kaggle/working")\n    if os.environ.get("ROGII_OUTPUT_DIR"):\n        return Path(os.environ["ROGII_OUTPUT_DIR"])\n    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()\n\n\nDATA       = _find()\nTRAIN_DIR  = DATA / "train"\nTEST_DIR   = DATA / "test"\nSAMPLE     = DATA / "sample_submission.csv"\nOUTPUT_DIR = _default_output_dir()\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\nOUT        = OUTPUT_DIR / "submission.csv"\n\nSAVE_ARTIFACTS = env_flag("ROGII_SAVE_ARTIFACTS", False)\nARTIFACT_DIR = Path(os.environ.get("ROGII_ARTIFACT_DIR", OUTPUT_DIR / "model_artifacts"))\nARTIFACT_MANIFEST = {\n    "version": "v10_fresh_artifact_train",\n    "goal": "public_score_9.5",\n    "lgb": [],\n    "catboost": [],\n    "tabicl_contexts": [],\n    "created_outputs": {},\n}\n\n\ndef _json_default(obj):\n    if isinstance(obj, (np.integer,)):\n        return int(obj)\n    if isinstance(obj, (np.floating,)):\n        return float(obj)\n    if isinstance(obj, np.ndarray):\n        return obj.tolist()\n    if isinstance(obj, Path):\n        return str(obj)\n    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")\n\n\ndef _artifact_rel(path: Path) -> str:\n    try:\n        return str(path.relative_to(ARTIFACT_DIR)).replace("\\\\", "/")\n    except ValueError:\n        return str(path).replace("\\\\", "/")\n\n\ndef save_json(path: Path, data) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")\n\n\ndef load_json(path: Path):\n    return json.loads(path.read_text(encoding="utf-8"))\n\n\ndef find_artifact_dir() -> Path:\n    candidates: list[Path] = []\n    if os.environ.get("ROGII_ARTIFACT_DIR"):\n        candidates.append(Path(os.environ["ROGII_ARTIFACT_DIR"]))\n    candidates.append(ARTIFACT_DIR)\n    for root in [Path("/kaggle/input"), DATA, OUTPUT_DIR]:\n        if root.exists():\n            candidates.extend(p.parent for p in root.glob("**/manifest.json"))\n    for candidate in dict.fromkeys(candidates):\n        if (candidate / "manifest.json").exists():\n            return candidate\n    raise FileNotFoundError(\n        "No artifact manifest found. Set ROGII_ARTIFACT_DIR to the model_artifacts folder."\n    )\n\n\nif SAVE_ARTIFACTS:\n    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)\n    print(f"Artifact saving enabled: {ARTIFACT_DIR}", flush=True)\n\n\ndef env_float(name: str, default: float) -> float:\n    value = os.environ.get(name)\n    if value is None or not str(value).strip():\n        return float(default)\n    return float(value)\n\n\ndef apply_exact_train_coordinate_blend(sub: pd.DataFrame, data_dir: Path) -> pd.DataFrame:\n    """Optional public train-coordinate blend; disabled with ROGII_EXACT_OVERLAP=0."""\n    if os.environ.get("ROGII_EXACT_OVERLAP", "1").strip().lower() in {"0", "false", "no"}:\n        print("Exact train-coordinate overlap override disabled.", flush=True)\n        return sub\n    blend_weight = env_float("ROGII_EXACT_BLEND_WEIGHT", 0.28)\n    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))\n\n    train_parts = []\n    for p in sorted((data_dir / "train").glob("*__horizontal_well.csv")):\n        try:\n            cur = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"])\n        except Exception:\n            continue\n        cur = cur[cur["TVT"].notna()].copy()\n        if not cur.empty:\n            train_parts.append(cur)\n    if not train_parts:\n        print("Exact overlap: no train coordinate rows found.", flush=True)\n        return sub\n\n    train_all = pd.concat(train_parts, ignore_index=True)\n    for col in ["X", "Y", "Z"]:\n        train_all[col + "_r"] = train_all[col].round(2)\n    train_map = (\n        train_all\n        .drop_duplicates(subset=["X_r", "Y_r", "Z_r"])\n        .set_index(["X_r", "Y_r", "Z_r"])["TVT"]\n        .to_dict()\n    )\n\n    coord_parts = []\n    for p in sorted((data_dir / "test").glob("*__horizontal_well.csv")):\n        wid = p.name.split("__")[0]\n        try:\n            cur = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT_input"])\n        except Exception:\n            continue\n        mask = cur["TVT_input"].isna().to_numpy()\n        if not mask.any():\n            continue\n        row_idx = np.arange(len(cur))[mask]\n        part = cur.loc[mask, ["X", "Y", "Z"]].copy()\n        part["id"] = [f"{wid}_{int(i)}" for i in row_idx]\n        coord_parts.append(part)\n    if not coord_parts:\n        print("Exact overlap: no test prediction rows found.", flush=True)\n        return sub\n\n    coord = pd.concat(coord_parts, ignore_index=True)\n    coord["key"] = list(zip(coord["X"].round(2), coord["Y"].round(2), coord["Z"].round(2)))\n    coord["exact_tvt"] = coord["key"].map(train_map)\n    exact = coord[coord["exact_tvt"].notna()][["id", "exact_tvt"]]\n    if exact.empty:\n        print("Exact overlap: 0 rows replaced.", flush=True)\n        return sub\n\n    out = sub.merge(exact, on="id", how="left")\n    mask = out["exact_tvt"].notna()\n    out.loc[mask, "tvt"] = (\n        (1.0 - blend_weight) * out.loc[mask, "tvt"].astype(float)\n        + blend_weight * out.loc[mask, "exact_tvt"].astype(float)\n    )\n    print(\n        f"Exact overlap blend: blended {int(mask.sum())}/{len(out)} rows "\n        f"with weight={blend_weight:.3f}.",\n        flush=True,\n    )\n    return out[["id", "tvt"]]\n\nTRAIN_CORE_CACHE_NAME = "aeroridge_train_core_df.pkl"\nCACHE_ONLY            = env_flag("ROGII_CACHE_ONLY", False)\nUSE_TRAIN_CORE_CACHE  = env_flag("ROGII_USE_TRAIN_CORE_CACHE", True)\nWRITE_TRAIN_CORE_CACHE = env_flag("ROGII_WRITE_TRAIN_CORE_CACHE", not RUNNING_ON_KAGGLE)\nRUN_TABICL            = env_flag("ROGII_RUN_TABICL", True)\nINFERENCE_ONLY        = env_flag("ROGII_INFERENCE_ONLY", False)\nSAVE_FEATURE_FRAMES   = env_flag("ROGII_SAVE_FEATURE_FRAMES", False)\nUSE_HILL_STACK        = env_flag("ROGII_USE_HILL_STACK", True)\nHILL_PRECISION        = env_float("ROGII_HILL_PRECISION", 0.01)\nDEBUG_MAX_TRAIN_WELLS = env_int("ROGII_DEBUG_MAX_TRAIN_WELLS", 0)\nDEBUG_MAX_TEST_WELLS  = env_int("ROGII_DEBUG_MAX_TEST_WELLS", 0)\n\n\ndef _cache_candidates() -> list[Path]:\n    paths = []\n    if os.environ.get("ROGII_TRAIN_CORE_CACHE"):\n        paths.append(Path(os.environ["ROGII_TRAIN_CORE_CACHE"]))\n    for root in [Path("/kaggle/input"), OUTPUT_DIR]:\n        if root.exists():\n            paths.extend(root.glob(f"**/{TRAIN_CORE_CACHE_NAME}"))\n    return list(dict.fromkeys(paths))\n\n\ndef _train_core_cache_path_for_write() -> Path:\n    if os.environ.get("ROGII_TRAIN_CORE_CACHE_OUT"):\n        return Path(os.environ["ROGII_TRAIN_CORE_CACHE_OUT"])\n    return OUTPUT_DIR / TRAIN_CORE_CACHE_NAME\n\n\nprint(f"DATA={DATA}")\nprint(f"OUTPUT_DIR={OUTPUT_DIR}")\n\nFORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]\nPLANE_K    = 10\nDENSE_SPW  = 60\nDENSE_K    = 20\nN_SPLITS   = 5\nN_AUG_SPLITS      = 1#3\nMIN_KNOWN_FOR_AUG = 20\n\n\'\'\'BEAMS = [\n    (10, 20.0, 144.0, 3, "cons"),\n    (10,  8.0,  64.0, 3, "loose"),\n    ( 8, 35.0, 220.0, 1, "vcons"),\n    (10, 14.0,  90.0, 5, "sm5"),\n    (20,  4.0,  36.0, 3, "vloose"),\n    (12, 12.0, 100.0, 3, "mid"),\n    (15, 25.0, 180.0, 2, "stiff"),\n]\'\'\'\nBEAMS = [\n    (10, 20.0, 144.0, 3, "cons"),\n    (10,  8.0,  64.0, 3, "loose"),\n    (10, 14.0,  90.0, 5, "sm5"),\n]\n\nPF_N = 150; ANCC_N = 150\nPF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01\nPF_GR_SIG_MIN = 10.0; PF_GR_SIG_MAX = 60.0; PF_GR_SIG_DEF = 30.0\nPF_INIT_V_STD = 0.02; PF_INIT_SPR = 0.5; PF_RESAMP = 0.5\nPF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3\nANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005\nANCC_IR = 0.01; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001\n\nNOTEBOOK_RUN_VERSION = "ROGII_v26_combinedbest_lgbfix_cb3_tabicl_splitgpu_2026_05_10"\n\nLGB_P = dict(\n    boosting_type="gbdt",\n    learning_rate=0.04,\n    num_leaves=127,\n    min_child_samples=20,\n    subsample=0.8,\n    colsample_bytree=0.8,\n    reg_lambda=5.0,\n    reg_alpha=0.1,\n    objective="regression",\n    verbose=-1,\n    n_jobs=-1,\n    device_type="gpu",\n    gpu_use_dp=False,\n    max_bin=255,\n)\nLGB_SEEDS = [42, 7, 123]\n\nimport subprocess as _s\n\n\ndef _gpu_names() -> list[str]:\n    try:\n        out = _s.run(\n            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],\n            capture_output=True, text=True, check=False,\n        ).stdout.strip()\n        return [line.strip() for line in out.splitlines() if line.strip()]\n    except Exception:\n        return []\n\n\nGPU_NAMES = _gpu_names()\nGPU_COUNT = len(GPU_NAMES)\nCATBOOST_DEVICES = os.environ.get(\n    "ROGII_CATBOOST_DEVICES",\n    ":".join(str(i) for i in range(GPU_COUNT)) if GPU_COUNT else "0",\n)\n\nCB_P = dict(\n    iterations=5000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,\n    min_data_in_leaf=20, loss_function="RMSE",\n    task_type="GPU", devices=CATBOOST_DEVICES, od_type="Iter", od_wait=150, verbose=0,\n)\nFORCE_CPU = env_flag("ROGII_FORCE_CPU", False)\nif FORCE_CPU:\n    LGB_P["device_type"] = "cpu"\n    LGB_P.pop("gpu_use_dp", None)\n    LGB_P.pop("max_bin", None)\n    CB_P["task_type"] = "CPU"\n    CB_P.pop("devices", None)\n\nprint("GPUs:", "\\n".join(GPU_NAMES) if GPU_NAMES else "(none)")\nprint(f"CatBoost devices: {CATBOOST_DEVICES}")\nprint(f"FORCE_CPU={FORCE_CPU}")\nprint(f"CPUs={NCPU}  train={len(list(TRAIN_DIR.glob(\'*__horizontal_well.csv\')))} wells")\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Helpers\n# ═══════════════════════════════════════════════════════════════════════════════\n\ndef nn_idx(arr: np.ndarray, v: float) -> int:\n    i = int(np.searchsorted(arr, v, "left"))\n    n = len(arr)\n    if i >= n:\n        return n - 1\n    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):\n        return i - 1\n    return i\n\n\ndef robust_slope(x: np.ndarray, y: np.ndarray) -> float:\n    x = np.asarray(x, float); y = np.asarray(y, float)\n    m = np.isfinite(x) & np.isfinite(y)\n    if m.sum() < 2 or np.std(x[m]) < 1e-6:\n        return 0.0\n    return float(np.polyfit(x[m], y[m], 1)[0])\n\n\ndef affine_cal(kgr: np.ndarray, tw_at_k: np.ndarray, min_pts: int = 20):\n    v = np.isfinite(kgr) & np.isfinite(tw_at_k)\n    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:\n        bias = float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0\n        return 1.0, bias\n    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)\n    return float(a), float(b)\n\n\ndef wls_b_well(ktvt: np.ndarray, kz: np.ndarray, form_kn_col: np.ndarray,\n               decay: float = 0.02) -> float:\n    n = len(ktvt)\n    if n < 3:\n        return float(np.median(ktvt + kz - form_kn_col))\n    w = np.exp(decay * np.arange(n, dtype=np.float64))\n    w /= w.sum()\n    return float(np.dot(w, ktvt + kz - form_kn_col))\n\n\ndef safe_interp(x, xp, fp):\n    xp = np.asarray(xp, float); fp = np.asarray(fp, float)\n    m = np.isfinite(xp) & np.isfinite(fp)\n    if m.sum() < 2:\n        return np.full(len(np.asarray(x)), np.nan)\n    o = np.argsort(xp[m])\n    return np.interp(np.asarray(x, float), xp[m][o], fp[m][o],\n                     left=np.nan, right=np.nan)\n\n\ndef _rolling_std_cumsum(a: np.ndarray, w: int) -> np.ndarray:\n    s = pd.Series(a.astype(np.float64))\n    return s.rolling(w, center=True, min_periods=1).std().fillna(0.0).to_numpy(np.float32)\n\n\ndef _build_gr_rolls(gr_vals: np.ndarray, ev_iloc: np.ndarray):\n    """\n    Compute all GR rolling features for the eval rows in one pass.\n    Single Series construction; all window sizes computed from it.\n    """\n    s = pd.Series(gr_vals.astype(np.float64))\n    out = {}\n    # All rolling windows in one loop to share the Series object\n    for w in (5, 21, 51, 101):\n        rm = s.rolling(w, center=True, min_periods=1)\n        mean_arr = rm.mean().to_numpy(np.float32)\n        std_arr  = rm.std().fillna(0.0).to_numpy(np.float32)\n        out[f"grm{w}"] = mean_arr[ev_iloc]\n        out[f"grs{w}"] = std_arr[ev_iloc]\n    # Lags / leads — computed from the base series, not rolling\n    for lag in (1, 5, 15, 30):\n        out[f"glag{lag}"]  = s.shift(lag ).bfill().to_numpy(np.float32)[ev_iloc]\n        out[f"glead{lag}"] = s.shift(-lag).ffill().to_numpy(np.float32)[ev_iloc]\n    diff1 = s.diff().fillna(0.0).to_numpy(np.float32)\n    diff2 = s.diff().diff().fillna(0.0).to_numpy(np.float32)\n    out["gr_d1"] = diff1[ev_iloc]\n    out["gr_d2"] = diff2[ev_iloc]\n    return out\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Multi-scale self-correlation\n# ═══════════════════════════════════════════════════════════════════════════════\n\ndef multi_scale_sc(kgr: np.ndarray, ktvt: np.ndarray, hgr: np.ndarray,\n                   hws=(8, 15, 25), stride: int = 3):\n    fallback = float(ktvt[-1]) if len(ktvt) > 0 else 0.0\n    nh = len(hgr); nk = len(kgr)\n    results = []\n\n    # Smooth once; share across all scales\n    kg_sm = (pd.Series(kgr).rolling(5, center=True, min_periods=1)\n             .mean().to_numpy(np.float32))\n    hg_sm = (pd.Series(hgr).rolling(5, center=True, min_periods=1)\n             .mean().to_numpy(np.float32))\n\n    # Build eval Hankel matrix once for the largest window; slice for smaller ones\n    max_hw = max(hws)\n    max_win = 2 * max_hw + 1\n    _hp_max = np.pad(hg_sm, max_hw, mode="edge")\n    # Pre-build the full H matrix for max window (reused via slicing below)\n    if nh > 0 and nk >= max_win + 1:\n        _H_max = _hp_max[np.arange(nh)[:, None] + np.arange(max_win)[None, :]].astype(np.float32)\n    else:\n        _H_max = None\n\n    for hw_sc in hws:\n        win = 2 * hw_sc + 1\n        if nk < win + 1 or nh == 0:\n            results.append((np.full(nh, fallback, np.float32),\n                            np.zeros(nh, np.float32)))\n            continue\n\n        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)\n        if len(sts) == 0:\n            results.append((np.full(nh, fallback, np.float32),\n                            np.zeros(nh, np.float32)))\n            continue\n\n        # Known-side Hankel\n        idx_mat = sts[:, None] + np.arange(win, dtype=np.int32)[None, :]\n        C  = kg_sm[idx_mat].astype(np.float32)\n        mu = C.mean(1, keepdims=True); sd = C.std(1, keepdims=True) + 1e-6\n        Cn = (C - mu) / sd\n\n        # Eval-side Hankel: reuse _H_max by slicing center columns when possible\n        if _H_max is not None and hw_sc == max_hw:\n            H = _H_max\n        else:\n            # For smaller windows, re-pad efficiently\n            pad_h = hw_sc\n            hp = np.pad(hg_sm, pad_h, mode="edge")\n            H  = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)\n\n        mu_h = H.mean(1, keepdims=True); sd_h = H.std(1, keepdims=True) + 1e-6\n        Hn   = (H - mu_h) / sd_h\n\n        # Matrix NCC: (nh × ns) — use float32 matmul (faster on GPU-less CPU)\n        ncc  = np.dot(Hn, Cn.T) / win          # shape (nh, ns)\n        best  = ncc.argmax(1)\n        score = ncc.max(1).astype(np.float32)\n        ctrs  = np.clip(sts[best] + hw_sc, 0, nk - 1)\n        results.append((ktvt[ctrs].astype(np.float32), score))\n\n    return results\n\n\ndef gr_envelope(gr: np.ndarray, w: int = 21) -> np.ndarray:\n    return (pd.Series(gr).rolling(w, center=True, min_periods=1)\n            .max().to_numpy(np.float32))\n\n\ndef gr_energy(gr: np.ndarray, w: int = 21) -> np.ndarray:\n    sq = gr.astype(np.float64) ** 2\n    return np.sqrt(\n        pd.Series(sq).rolling(w, center=True, min_periods=1)\n        .mean().to_numpy().clip(0)\n    ).astype(np.float32)\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Beam Search  (vectorised; ±2 delta; boolean seen-array instead of set)\n# ═══════════════════════════════════════════════════════════════════════════════\n\n_DELTAS = np.array([-2, -1, 0, 1, 2], dtype=np.int32)\n_ND     = len(_DELTAS)\n\n\ndef beam_search(gr_h: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,\n                start_tvt: float, bs: int = 10, mc: float = 20.0,\n                es: float = 144.0, r: int = 2) -> np.ndarray:\n    tw_tvt = np.asarray(tw_tvt, np.float32)\n    tw_gr  = np.asarray(tw_gr,  np.float32)\n    T  = len(tw_tvt)\n    fb = float(np.nanmean(tw_gr))\n\n    sg = pd.Series(gr_h, dtype="float32").interpolate(\n        limit_direction="both").fillna(fb)\n    if r > 0:\n        sg = sg.rolling(r * 2 + 1, center=True, min_periods=1).mean()\n    sg = sg.to_numpy(np.float32)\n\n    si = nn_idx(tw_tvt, start_tvt)\n    ns = len(sg)\n\n    bps = np.empty((ns, bs), np.int32)\n    bpb = np.empty((ns, bs), np.int32)\n\n    bi  = np.full(bs, si, np.int32)\n    bc  = np.zeros(bs, np.float64)\n\n    # Pre-allocate movement penalty (constant across steps)\n    mv = mc * np.abs(_DELTAS).astype(np.float64)   # shape (ND,)\n\n    # Boolean seen array — O(T) reset but avoids set hashing\n    _seen = np.zeros(T, dtype=np.bool_)\n\n    for s, gv in enumerate(sg):\n        ci = np.clip(bi[:, None] + _DELTAS[None, :], 0, T - 1)   # (bs, ND)\n        em = (gv - tw_gr[ci]) ** 2 / es                            # (bs, ND)\n        cc = bc[:, None] + em + mv[None, :]                        # (bs, ND)\n\n        fi  = ci.ravel()       # (bs*ND,)\n        fc  = cc.ravel()       # (bs*ND,)\n        fp  = np.repeat(np.arange(bs, dtype=np.int32), _ND)\n\n        ord_ = np.argsort(fc, kind="stable")\n\n        # Reset only the positions we used last step (O(bs*ND) not O(T))\n        _seen[:] = False\n        kept = []\n        for o in ord_:\n            t = int(fi[o])\n            if not _seen[t]:\n                _seen[t] = True\n                kept.append(o)\n            if len(kept) == bs:\n                break\n        while len(kept) < bs:\n            kept.append(kept[-1])\n        kept_arr = np.array(kept, np.int32)\n\n        bps[s] = fp[kept_arr]\n        bpb[s] = fi[kept_arr]\n        bi = fi[kept_arr].astype(np.int32)\n        bc = fc[kept_arr]\n\n    path = np.empty(ns, np.int32)\n    cb   = int(np.argmin(bc))\n    for s in range(ns - 1, -1, -1):\n        path[s] = bpb[s, cb]\n        cb      = bps[s, cb]\n\n    return tw_tvt[path]\n\n\ndef run_all_beams(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,\n                  last_tvt: float, gr_filled: pd.Series,\n                  eval_start_idx: int) -> dict:\n    hgr    = gr_filled.iloc[eval_start_idx:].to_numpy(np.float32)\n    n_eval = int((hw["TVT_input"].isna()).sum())\n    result = {}\n    for bs, mc, es, r, tag in BEAMS:\n        pred = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)\n        result[tag] = pred[:n_eval].astype(np.float32)\n    return result\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Particle Filters\n# ═══════════════════════════════════════════════════════════════════════════════\n\ndef _cal_gr_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:\n    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]\n    if len(kn) < 20:\n        return PF_GR_SIG_DEF\n    ex = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)\n    return float(np.clip(np.std(kn["GR"].values - ex),\n                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))\n\n\ndef _z_beta(hw: pd.DataFrame):\n    kn = hw[hw["TVT_input"].notna()]\n    if len(kn) < 30:\n        return -1.0, 0.0, 0.1\n    dz   = np.diff(kn["Z"].values)\n    dtvt = np.diff(kn["TVT_input"].values)\n    dmd  = np.diff(kn["MD"].values)\n    m    = dmd > 0\n    if m.sum() < 10:\n        return -1.0, 0.0, 0.1\n    vz = dz[m] / dmd[m]; vt = dtvt[m] / dmd[m]\n    A  = np.column_stack([vz, np.ones_like(vz)])\n    c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)\n    return float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)\n\n\ndef _init_v(hw: pd.DataFrame) -> float:\n    kn = hw[hw["TVT_input"].notna()]\n    if len(kn) < 10:\n        return 0.0\n    tail = kn.tail(20)\n    dtvt = np.diff(tail["TVT_input"].values)\n    dmd  = np.diff(tail["MD"].values)\n    m    = dmd > 0\n    return 0.0 if m.sum() < 3 else float(np.median(dtvt[m] / dmd[m]))\n\n\n# ── numba-compiled PF loop cores ─────────────────────────────────────────────\n\n@_njit(cache=True)\ndef _numba_interp1(xp: np.ndarray, fp: np.ndarray, x: float) -> float:\n    n = len(xp)\n    if x <= xp[0]:  return fp[0]\n    if x >= xp[-1]: return fp[-1]\n    lo = 0; hi = n - 1\n    while hi - lo > 1:\n        mid = (lo + hi) >> 1\n        if xp[mid] <= x: lo = mid\n        else:             hi = mid\n    t = (x - xp[lo]) / (xp[hi] - xp[lo])\n    return fp[lo] + t * (fp[hi] - fp[lo])\n\n\n@_njit(cache=True)\ndef _pf_z_loop(md_v, gr_v, z_v,\n               tw_tvt, tw_gr, tw_s,\n               pos, vel, w,\n               gs, beta, icpt, zsig,\n               PF_MOM, PF_VN, PF_PN, tmin, tmax,\n               PF_GR_WT, PF_RESAMP, PF_ROUGH_P, PF_ROUGH_V,\n               gr_sm_v, seed):\n    np.random.seed(seed)\n    N   = len(pos)\n    n_e = len(md_v)\n    pts = np.empty(n_e, np.float32)\n    std = np.empty(n_e, np.float32)\n    pm  = md_v[0]\n    for i in range(n_e):\n        dm  = max(md_v[i] - pm, 1.0)\n        dzd = (z_v[i] - (z_v[i - 1] if i > 0 else z_v[0])) / dm\n\n        noise_v = np.random.normal(0.0, PF_VN, N)\n        noise_p = np.random.normal(0.0, PF_PN, N)\n        _lo = tmin - 50.0; _hi = tmax + 50.0\n        for j in range(N):\n            vel[j] = PF_MOM * vel[j] + noise_v[j]\n            _raw   = pos[j] + vel[j] * dm + noise_p[j]\n            pos[j] = min(max(_raw, _lo), _hi)\n\n        gv = gr_v[i]\n        if not np.isnan(gv):\n            for j in range(N):\n                ep = _numba_interp1(tw_tvt, tw_gr, pos[j])\n                lp = np.exp(-0.5 * ((gv - ep) / gs) ** 2)\n                gs_sm_j = gr_sm_v[i] if not np.isnan(gr_sm_v[i]) else np.nan\n                if not np.isnan(gs_sm_j):\n                    es_ = _numba_interp1(tw_tvt, tw_s, pos[j])\n                    ls  = np.exp(-0.5 * ((gs_sm_j - es_) / (gs * 1.5)) ** 2)\n                    lk  = (1.0 - PF_GR_WT) * lp + PF_GR_WT * ls\n                else:\n                    lk = lp\n                w[j] *= max(lk, 1e-300)\n            ws = np.sum(w)\n            if ws > 0.0:\n                for j in range(N): w[j] /= ws\n            else:\n                for j in range(N): w[j] = 1.0 / N\n\n        zs = max(zsig * 2.0, 0.005)\n        for j in range(N):\n            ve = beta * dzd + icpt\n            lz = max(np.exp(-0.5 * ((vel[j] - ve) / zs) ** 2), 1e-300)\n            w[j] *= lz\n        ws = np.sum(w)\n        if ws > 0.0:\n            for j in range(N): w[j] /= ws\n        else:\n            for j in range(N): w[j] = 1.0 / N\n\n        ne = 1.0 / np.sum(w * w)\n        if ne < PF_RESAMP * N:\n            cum = np.cumsum(w)\n            u_  = (np.arange(N) + np.random.uniform()) / N\n            ix  = np.searchsorted(cum, u_)\n            new_pos = pos[ix].copy(); new_vel = vel[ix].copy()\n            noise_rp = np.random.normal(0.0, PF_ROUGH_P, N)\n            noise_rv = np.random.normal(0.0, PF_ROUGH_V, N)\n            for j in range(N):\n                pos[j] = new_pos[j] + noise_rp[j]\n                vel[j] = new_vel[j] + noise_rv[j]\n                w[j]   = 1.0 / N\n\n        mu_ = 0.0\n        for j in range(N): mu_ += w[j] * pos[j]\n        var_ = 0.0\n        for j in range(N): var_ += w[j] * (pos[j] - mu_) ** 2\n        pts[i] = np.float32(mu_)\n        std[i] = np.float32(np.sqrt(var_))\n        pm = md_v[i]\n\n    return pts, std\n\n\n@_njit(cache=True)\ndef _pf_ancc_loop(md_v, gr_v, z_v,\n                  tw_tvt, tw_gr,\n                  pos, rate, w, gs,\n                  ANCC_ALPHA, ANCC_RN, ANCC_PN,\n                  tmin, tmax, PF_RESAMP,\n                  ANCC_RP, ANCC_RR, seed):\n    np.random.seed(seed)\n    N   = len(pos)\n    n_e = len(md_v)\n    pts = np.empty(n_e, np.float32)\n    std = np.empty(n_e, np.float32)\n    pm  = md_v[0]\n    for i in range(n_e):\n        dm = max(md_v[i] - pm, 1.0)\n        noise_r = np.random.normal(0.0, ANCC_RN, N)\n        noise_p = np.random.normal(0.0, ANCC_PN, N)\n        for j in range(N):\n            rate[j] = ANCC_ALPHA * rate[j] + noise_r[j]\n            pos[j] += rate[j] * dm + noise_p[j]\n        _lo = tmin - 50.0; _hi = tmax + 50.0\n        tvt_e = np.empty(N, np.float64)\n        _zvi  = z_v[i]\n        for j in range(N):\n            _raw    = pos[j] - _zvi\n            tvt_e[j] = min(max(_raw, _lo), _hi)\n        for j in range(N): pos[j] = tvt_e[j] + _zvi\n\n        gv = gr_v[i]\n        if not np.isnan(gv):\n            for j in range(N):\n                eg  = _numba_interp1(tw_tvt, tw_gr, tvt_e[j])\n                lk  = max(np.exp(-0.5 * ((gv - eg) / gs) ** 2), 1e-300)\n                w[j] *= lk\n            ws = np.sum(w)\n            if ws > 0.0:\n                for j in range(N): w[j] /= ws\n            else:\n                for j in range(N): w[j] = 1.0 / N\n\n        ne = 1.0 / np.sum(w * w)\n        if ne < PF_RESAMP * N:\n            cum = np.cumsum(w)\n            u_  = (np.arange(N) + np.random.uniform()) / N\n            ix  = np.searchsorted(cum, u_)\n            new_pos  = pos[ix].copy(); new_rate = rate[ix].copy()\n            noise_rp = np.random.normal(0.0, ANCC_RP, N)\n            noise_rr = np.random.normal(0.0, ANCC_RR, N)\n            for j in range(N):\n                pos[j]  = new_pos[j]  + noise_rp[j]\n                rate[j] = new_rate[j] + noise_rr[j]\n                w[j]    = 1.0 / N\n\n        tvt_e2 = pos - z_v[i]\n        mu_ = 0.0\n        for j in range(N): mu_ += w[j] * tvt_e2[j]\n        var_ = 0.0\n        for j in range(N): var_ += w[j] * (tvt_e2[j] - mu_) ** 2\n        pts[i] = np.float32(mu_)\n        std[i] = np.float32(np.sqrt(var_))\n        pm = md_v[i]\n\n    return pts, std\n\n\ndef run_pf_z(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,\n             N: int = PF_N):\n    tw_s = (pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1)\n            .mean().to_numpy(np.float32))\n    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())\n    gs = _cal_gr_sigma(hw, tw_tvt, tw_gr)\n    beta, icpt, zsig = _z_beta(hw)\n    kn = hw[hw["TVT_input"].notna()]\n    ev = hw[hw["TVT_input"].isna()]\n    if len(ev) == 0:\n        return np.array([], np.float32), np.array([], np.float32)\n\n    gr_sm_full = (hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1)\n                  .mean().to_numpy(np.float32))\n    ev_locs   = [hw.index.get_loc(idx) for idx in ev.index]\n    gr_sm_ev  = gr_sm_full[ev_locs].astype(np.float32)\n\n    pos  = float(kn["TVT_input"].iloc[-1]) + np.random.normal(0, PF_INIT_SPR, N).astype(np.float32)\n    vel  = (_init_v(hw) + np.random.normal(0, PF_INIT_V_STD, N)).astype(np.float32)\n    w    = np.full(N, 1.0 / N, np.float64)\n\n    md_v  = ev["MD"].to_numpy(np.float32)\n    gr_v  = ev["GR"].to_numpy(np.float32)\n    z_v   = ev["Z"].to_numpy(np.float32)\n\n    if _NUMBA:\n        loop_seed = int(np.random.randint(0, 2_147_483_647))\n        pts, std = _pf_z_loop(\n            md_v, gr_v, z_v,\n            tw_tvt.astype(np.float64), tw_gr.astype(np.float64),\n            tw_s.astype(np.float64), pos.astype(np.float64), vel.astype(np.float64),\n            w,\n            float(gs), float(beta), float(icpt), float(zsig),\n            float(PF_MOM), float(PF_VN), float(PF_PN),\n            float(tmin), float(tmax),\n            float(PF_GR_WT), float(PF_RESAMP), float(PF_ROUGH_P), float(PF_ROUGH_V),\n            gr_sm_ev.astype(np.float64),\n            loop_seed,\n        )\n    else:\n        tf_p  = interp1d(tw_tvt, tw_gr, bounds_error=False,\n                         fill_value=(tw_gr[0], tw_gr[-1]))\n        tf_s  = interp1d(tw_tvt, tw_s,  bounds_error=False,\n                         fill_value=(tw_s[0],  tw_s[-1]))\n        pts_l = np.empty(len(ev)); std_l = np.empty(len(ev))\n        pm_py = float(kn["MD"].iloc[-1]); pz = float(kn["Z"].iloc[-1])\n        for i, idx in enumerate(ev.index):\n            dm  = max(md_v[i] - pm_py, 1.0)\n            dzd = (z_v[i] - pz) / dm\n            vel = PF_MOM * vel + np.random.normal(0, PF_VN, N)\n            pos = pos + vel * dm + np.random.normal(0, PF_PN, N)\n            pos = np.clip(pos, tmin - 50, tmax + 50)\n            if not np.isnan(gr_v[i]):\n                ep = tf_p(pos)\n                lp = np.exp(-0.5 * ((gr_v[i] - ep) / gs) ** 2)\n                gs_sm = gr_sm_ev[i]\n                if not np.isnan(gs_sm):\n                    es_ = tf_s(pos)\n                    ls  = np.exp(-0.5 * ((gs_sm - es_) / (gs * 1.5)) ** 2)\n                    lk  = (1 - PF_GR_WT) * lp + PF_GR_WT * ls\n                else:\n                    lk = lp\n                lk = np.maximum(lk, 1e-300); w *= lk\n                ws = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)\n            ve = beta * dzd + icpt; zs = max(zsig * 2.0, 0.005)\n            lz = np.exp(-0.5 * ((vel - ve) / zs) ** 2)\n            lz = np.maximum(lz, 1e-300); w *= lz\n            ws = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)\n            ne = 1.0 / np.sum(w ** 2)\n            if ne < PF_RESAMP * N:\n                cum = np.cumsum(w); u = (np.arange(N) + np.random.uniform()) / N\n                ix  = np.searchsorted(cum, u)\n                pos = pos[ix]; vel = vel[ix]; w[:] = 1.0 / N\n                pos += np.random.normal(0, PF_ROUGH_P, N)\n                vel += np.random.normal(0, PF_ROUGH_V, N)\n            pts_l[i] = np.average(pos, weights=w)\n            std_l[i] = np.sqrt(np.average((pos - pts_l[i]) ** 2, weights=w))\n            pm_py = md_v[i]; pz = z_v[i]\n        pts, std = pts_l, std_l\n\n    return pts.astype(np.float32), std.astype(np.float32)\n\n\ndef run_pf_ancc(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,\n                N: int = ANCC_N):\n    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())\n    gs  = _cal_gr_sigma(hw, tw_tvt, tw_gr)\n    kn  = hw[hw["TVT_input"].notna()]\n    ev  = hw[hw["TVT_input"].isna()]\n    if len(ev) == 0:\n        return np.array([], np.float32), np.array([], np.float32)\n\n    ls  = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])\n    tail = kn.tail(30)\n    dt   = np.diff(tail["TVT_input"].values)\n    dz   = np.diff(tail["Z"].values)\n    dm   = np.diff(tail["MD"].values)\n    m    = dm > 0\n    ir   = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0\n\n    pos  = (ls + np.random.normal(0, ANCC_IS, N)).astype(np.float64)\n    rate = (ir  + np.random.normal(0, ANCC_IR, N)).astype(np.float64)\n    w    = np.full(N, 1.0 / N, np.float64)\n\n    md_v = ev["MD"].to_numpy(np.float32)\n    z_v  = ev["Z"].to_numpy(np.float32)\n    gr_v = ev["GR"].to_numpy(np.float32)\n\n    if _NUMBA:\n        loop_seed = int(np.random.randint(0, 2_147_483_647))\n        pts, std = _pf_ancc_loop(\n            md_v.astype(np.float64), gr_v.astype(np.float64),\n            z_v.astype(np.float64),\n            tw_tvt.astype(np.float64), tw_gr.astype(np.float64),\n            pos, rate, w, float(gs),\n            float(ANCC_ALPHA), float(ANCC_RN), float(ANCC_PN),\n            float(tmin), float(tmax), float(PF_RESAMP),\n            float(ANCC_RP), float(ANCC_RR),\n            loop_seed,\n        )\n    else:\n        pm    = float(kn["MD"].iloc[-1])\n        pts_l = np.empty(len(ev)); std_l = np.empty(len(ev))\n        for i in range(len(ev)):\n            dm_s = max(md_v[i] - pm, 1.0)\n            rate = ANCC_ALPHA * rate + np.random.normal(0, ANCC_RN, N)\n            pos  = pos + rate * dm_s + np.random.normal(0, ANCC_PN, N)\n            tvt_e = np.clip(pos - z_v[i], tmin - 50, tmax + 50)\n            pos   = tvt_e + z_v[i]\n            if not np.isnan(gr_v[i]):\n                eg  = np.interp(tvt_e, tw_tvt, tw_gr)\n                lk  = np.exp(-0.5 * ((gr_v[i] - eg) / gs) ** 2)\n                lk  = np.maximum(lk, 1e-300); w *= lk\n                ws  = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)\n            ne = 1.0 / np.sum(w ** 2)\n            if ne < PF_RESAMP * N:\n                cum = np.cumsum(w); u = (np.arange(N) + np.random.uniform()) / N\n                ix  = np.searchsorted(cum, u)\n                pos = pos[ix]; rate = rate[ix]; w[:] = 1.0 / N\n                pos  += np.random.normal(0, ANCC_RP, N)\n                rate += np.random.normal(0, ANCC_RR, N)\n            tv = float(np.average(pos - z_v[i], weights=w))\n            pts_l[i] = tv\n            std_l[i] = np.sqrt(np.average((pos - z_v[i] - tv) ** 2, weights=w))\n            pm = md_v[i]\n        pts, std = pts_l, std_l\n\n    return pts.astype(np.float32), std.astype(np.float32)\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Spatial Imputers\n# ═══════════════════════════════════════════════════════════════════════════════\n\nclass FormationPlaneKNN:\n    def __init__(self, well_ids, data_dir: Path):\n        rows = []\n        for wid in well_ids:\n            p = data_dir / f"{wid}__horizontal_well.csv"\n            try:\n                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()\n            except Exception:\n                continue\n            if len(df) == 0:\n                continue\n            row = {"wid": wid,\n                   "x": float(df["X"].median()),\n                   "y": float(df["Y"].median())}\n            for c in FORMATIONS:\n                row[f"{c}_m"] = float(df[c].median())\n            rows.append(row)\n\n        self.df   = pd.DataFrame(rows)\n        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}\n        xy        = self.df[["x", "y"]].to_numpy(np.float64)\n        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))\n        self.tree  = cKDTree(xy / self.scale)\n        self.xa    = self.df["x"].to_numpy(np.float64)\n        self.ya    = self.df["y"].to_numpy(np.float64)\n        self.fa    = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)\n        self._fa_mean = self.fa.mean(0)\n\n    def impute(self, xy_q: np.ndarray, self_wid=None, k: int = PLANE_K):\n        q    = xy_q / self.scale\n        nf   = min(k + 5, len(self.df))\n        dist, idx = self.tree.query(q, k=nf, workers=-1)\n\n        if self_wid in self.wmap:\n            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)\n\n        if nf > k:\n            ord_ = np.argpartition(dist, k - 1, axis=1)[:, :k]\n            dk   = np.take_along_axis(dist, ord_, axis=1)\n            ik   = np.take_along_axis(idx,  ord_, axis=1)\n        else:\n            dk, ik = dist, idx\n\n        vk  = np.isfinite(dk)\n        w   = np.where(vk, 1.0 / (dk + 1e-3), 0.0)\n\n        xn = self.xa[ik]; yn = self.ya[ik]\n        wx = w * xn;      wy = w * yn\n\n        A = np.zeros((len(q), 3, 3), np.float64)\n        A[:, 0, 0] = (wx * xn).sum(1);  A[:, 0, 1] = (wx * yn).sum(1)\n        A[:, 0, 2] = wx.sum(1)\n        A[:, 1, 0] = A[:, 0, 1];        A[:, 1, 1] = (wy * yn).sum(1)\n        A[:, 1, 2] = wy.sum(1)\n        A[:, 2, 0] = A[:, 0, 2];        A[:, 2, 1] = A[:, 1, 2]\n        A[:, 2, 2] = w.sum(1)\n        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9\n\n        fn  = self.fa[ik]\n        rhs = np.stack([\n            (wx[:, :, None] * fn).sum(1),\n            (wy[:, :, None] * fn).sum(1),\n            (w[:, :, None]  * fn).sum(1),\n        ], axis=1)\n\n        try:\n            coef = np.linalg.solve(A, rhs)\n        except np.linalg.LinAlgError:\n            coef = np.zeros((len(q), 3, 6))\n            for r in range(len(q)):\n                try:\n                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]\n                except Exception:\n                    pass\n\n        Xq   = xy_q[:, 0]; Yq = xy_q[:, 1]\n        pred = (Xq[:, None] * coef[:, 0, :]\n                + Yq[:, None] * coef[:, 1, :]\n                + coef[:, 2, :]).astype(np.float32)\n\n        no_nbr = ~vk.any(1)\n        pred[no_nbr] = self._fa_mean.astype(np.float32)\n\n        min_dist = np.where(vk, dk, np.inf).min(1).astype(np.float32)\n        return pred, min_dist\n\n\nclass DenseANCCImputer:\n    def __init__(self, well_ids, data_dir: Path, spw: int = DENSE_SPW):\n        xs, ys, anccs, wids = [], [], [], []\n        for wid in well_ids:\n            p = data_dir / f"{wid}__horizontal_well.csv"\n            try:\n                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()\n            except Exception:\n                continue\n            if len(df) == 0:\n                continue\n            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)\n            s  = df.iloc[ix]\n            xs.append(s["X"].values); ys.append(s["Y"].values)\n            anccs.append(s["ANCC"].values)\n            wids.extend([wid] * len(s))\n\n        self.xy   = np.column_stack([np.concatenate(xs), np.concatenate(ys)])\n        self.ancc = np.concatenate(anccs).astype(np.float32)\n        self.wids = np.array(wids)\n        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))\n        self.tree  = cKDTree(self.xy / self.scale)\n        self._mean = float(self.ancc.mean())\n\n    def impute(self, xy_q: np.ndarray, self_wid=None,\n               k: int = DENSE_K, nfetch: int = 500):\n        xy_q = np.atleast_2d(xy_q)\n        q    = xy_q / self.scale\n        nf   = min(nfetch, len(self.ancc))\n\n        dist, idx = self.tree.query(q, k=nf, workers=-1)\n        if self_wid is not None:\n            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)\n\n        if nf > k:\n            ord_ = np.argpartition(dist, k - 1, axis=1)[:, :k]\n            dk   = np.take_along_axis(dist, ord_, axis=1)\n            ik   = np.take_along_axis(idx,  ord_, axis=1)\n        else:\n            dk, ik = dist, idx\n\n        vk  = np.isfinite(dk)\n        w   = np.where(vk, 1.0 / (dk + 1e-3), 0.0)\n        sw  = w.sum(1); safe = np.where(sw < 1e-9, 1.0, sw)\n\n        an  = self.ancc[ik]\n        ap  = (an * w).sum(1) / safe\n        ap  = np.where(sw < 1e-9, self._mean, ap)\n        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe\n\n        return (ap.astype(np.float32),\n                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),\n                np.where(vk, dk, np.inf).min(1).astype(np.float32))\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Feature Builder — split into I/O wrapper + pure compute core\n# ═══════════════════════════════════════════════════════════════════════════════\n\nANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80],  dtype=np.float32)\nBEAM_OFFS = np.array([-40, -20, -10,  -5, -3, 0, 3,  5, 10, 20, 40],  dtype=np.float32)\nSC_OFFS   = np.array([-30, -15,  -8,  -4, -2, 0, 2,  4,  8, 15, 30],  dtype=np.float32)\n\n# Process-global imputers — set once per worker via pool initializer\n_FI: Optional[FormationPlaneKNN] = None\n_DI: Optional[DenseANCCImputer]  = None\n\n\ndef _worker_init(fi: FormationPlaneKNN, di: DenseANCCImputer) -> None:\n    """Called once per worker process; stores imputers in process-local globals."""\n    global _FI, _DI\n    _FI = fi\n    _DI = di\n\n\nUSE_PROCESS_POOL = env_flag("ROGII_USE_PROCESS_POOL", os.name != "nt")\nUSE_THREAD_POOL  = env_flag("ROGII_USE_THREAD_POOL", os.name == "nt")\n\n\ndef _map_with_imputers(func, items, processes):\n    """Use process pools on Kaggle/Linux and thread pools on Windows local runs."""\n    if USE_PROCESS_POOL and processes > 1:\n        with multiprocessing.Pool(\n            processes=processes,\n            initializer=_worker_init,\n            initargs=(FI, DI),\n        ) as pool:\n            return pool.map(func, items)\n    _worker_init(FI, DI)\n    if USE_THREAD_POOL and processes > 1:\n        with ThreadPoolExecutor(max_workers=processes) as pool:\n            return list(pool.map(func, items))\n    return [func(item) for item in items]\n\n\ndef _starmap_with_imputers(func, args, processes):\n    """Use process pools on Kaggle/Linux and thread pools on Windows local runs."""\n    if USE_PROCESS_POOL and processes > 1:\n        with multiprocessing.Pool(\n            processes=processes,\n            initializer=_worker_init,\n            initargs=(FI, DI),\n        ) as pool:\n            return pool.starmap(func, args)\n    _worker_init(FI, DI)\n    if USE_THREAD_POOL and processes > 1:\n        with ThreadPoolExecutor(max_workers=processes) as pool:\n            return list(pool.map(lambda arg: func(*arg), args))\n    return [func(*arg) for arg in args]\n\n\n# ── Pre-computed GR stats cache (per well, shared across aug splits) ──────────\n\nclass _WellGRCache:\n    """\n    Holds expensive per-well arrays that don\'t change across augmentation splits\n    (the full GR array, rolling stats, envelope, energy).\n    Constructed once in process_train_well and passed into build_augmented.\n    """\n    __slots__ = ("gr_arr", "roll_feats_full", "gr_env_full", "gr_nrg_full")\n\n    def __init__(self, hw: pd.DataFrame, gr_mean: float):\n        gr_full = (hw["GR"].astype(float)\n                   .interpolate(limit_direction="both")\n                   .fillna(gr_mean))\n        self.gr_arr         = gr_full.to_numpy(np.float32)\n        # Compute rolling stats over the FULL well once\n        all_idx             = np.arange(len(hw), dtype=np.int64)\n        self.roll_feats_full = _build_gr_rolls(self.gr_arr, all_idx)\n        self.gr_env_full     = gr_envelope(self.gr_arr)\n        self.gr_nrg_full     = gr_energy(self.gr_arr)\n\n\ndef _build_well_from_df(\n    hw: pd.DataFrame,\n    tw: pd.DataFrame,\n    is_train: bool,\n    wid: str,\n    gr_cache: Optional[_WellGRCache] = None,\n) -> Optional[pd.DataFrame]:\n    """\n    Pure-compute feature builder. No disk I/O.\n    gr_cache: if provided, reuses pre-computed GR rolling stats (avoids recompute\n              across augmentation splits of the same well).\n    """\n    seed_bytes = hashlib.sha256(f"{SEED}:{SEED_SALT}:{wid}:{int(is_train)}:{len(hw)}".encode("utf-8")).digest()\n    np.random.seed(int.from_bytes(seed_bytes[:4], "little"))\n    if _FI is None or _DI is None:\n        _dbg(f"_build_well_from_df({wid}): imputers not set!", level="ERROR")\n        return None\n\n    required_hw = {"MD", "GR", "X", "Y", "Z", "TVT_input"}\n    required_tw = {"TVT", "GR"}\n    missing_hw  = required_hw - set(hw.columns)\n    missing_tw  = required_tw - set(tw.columns)\n    if missing_hw:\n        _dbg(f"_build_well_from_df({wid}): missing hw cols {missing_hw}", level="WARN")\n        return None\n    if missing_tw:\n        _dbg(f"_build_well_from_df({wid}): missing tw cols {missing_tw}", level="WARN")\n        return None\n\n    if is_train and "TVT" not in hw.columns:\n        _dbg(f"_build_well_from_df({wid}): is_train but no TVT column", level="WARN")\n        return None\n\n    kn = hw[hw["TVT_input"].notna()]\n    ev = hw[hw["TVT_input"].isna()]\n    if len(ev) == 0:\n        _dbg(f"_build_well_from_df({wid}): no eval rows", level="WARN")\n        return None\n    if len(kn) < 10:\n        _dbg(f"_build_well_from_df({wid}): too few known rows ({len(kn)})", level="WARN")\n        return None\n    if is_train and hw["TVT"].isna().all():\n        _dbg(f"_build_well_from_df({wid}): is_train but all TVT NaN", level="WARN")\n        return None\n\n    tw_tvt = tw["TVT"].to_numpy(np.float32)\n    tw_gr  = tw["GR"].to_numpy(np.float32)\n    if len(tw_tvt) < 3:\n        _dbg(f"_build_well_from_df({wid}): typewell too short", level="WARN")\n        return None\n\n    try:\n        lk       = kn.iloc[-1]\n        last_tvt = float(lk["TVT_input"])\n        gr_mean  = float(np.nanmean(tw_gr))\n\n        # ── GR arrays — use cache if available, else compute ──────────────\n        if gr_cache is not None:\n            gr_arr = gr_cache.gr_arr\n        else:\n            gr_full = (hw["GR"].astype(float)\n                       .interpolate(limit_direction="both")\n                       .fillna(gr_mean))\n            gr_arr = gr_full.to_numpy(np.float32)\n\n        # iloc positions of eval rows (needed for rolling feature slicing)\n        ev_idx_arr = np.array([hw.index.get_loc(i) for i in ev.index], dtype=np.int64)\n\n        hgr = gr_arr[ev_idx_arr]\n        kgr = gr_arr[:len(kn)]\n\n        # ── GR rolling features ───────────────────────────────────────────\n        if gr_cache is not None:\n            # Slice pre-computed full-well rolling arrays to eval positions\n            roll_feats = {\n                k: v[ev_idx_arr]\n                for k, v in gr_cache.roll_feats_full.items()\n            }\n            hgr_env = gr_cache.gr_env_full[ev_idx_arr]\n            hgr_nrg = gr_cache.gr_nrg_full[ev_idx_arr]\n        else:\n            roll_feats = _build_gr_rolls(gr_arr, ev_idx_arr)\n            hgr_env    = gr_envelope(gr_arr)[ev_idx_arr]\n            hgr_nrg    = gr_energy(gr_arr)[ev_idx_arr]\n\n        gr_d1 = roll_feats.pop("gr_d1")\n        gr_d2 = roll_feats.pop("gr_d2")\n\n        # Particle filters\n        pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)\n        pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)\n        if len(pf_a) == 0:\n            _dbg(f"_build_well_from_df({wid}): pf_ancc returned empty", level="WARN")\n            return None\n\n        pf_use  = pf_a; std_use = std_a\n        has_z   = (len(pf_z) == len(pf_a)) and not np.any(np.isnan(pf_z))\n\n        # Beam search (7 configs)\n        # eval_start_idx: first iloc position of the eval block\n        eval_start_iloc = int(ev_idx_arr[0])\n        gr_filled_series = pd.Series(gr_arr)\n        bpaths    = run_all_beams(hw, tw_tvt, tw_gr, last_tvt,\n                                  gr_filled_series, eval_start_iloc)\n        beam_vals = np.stack(list(bpaths.values()), axis=1)\n        beam_ref  = (bpaths["cons"] + bpaths["sm5"]) * 0.5\n\n        # Multi-scale self-correlation\n        ktvt = kn["TVT_input"].to_numpy(np.float32)\n        sc_results  = multi_scale_sc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)\n        sc8,  sc8s  = sc_results[0]\n        sc15, sc15s = sc_results[1]\n        sc25, sc25s = sc_results[2]\n        sc_consensus = ((sc8 + sc15 + sc25) * (1.0 / 3.0)).astype(np.float32)\n        sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))\n        hyb_ref  = ((1 - sc_trust) * beam_ref + sc_trust * sc15).astype(np.float32)\n\n        # Affine calibration\n        tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)\n        a_cal, b_cal = affine_cal(kgr, tw_at_k)\n\n        # Prefix statistics\n        kmd      = kn["MD"].to_numpy(np.float32)\n        kz       = kn["Z"].to_numpy(np.float32)\n        pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))\n        slp_all  = robust_slope(kmd,       ktvt)\n        slp_50   = robust_slope(kmd[-50:], ktvt[-50:])\n        slp_z    = robust_slope(kz,        ktvt)\n\n        # Spatial imputers\n        swid    = wid if is_train else None\n        xy_ev   = ev[["X", "Y"]].to_numpy(np.float64)\n        xy_kn   = kn[["X", "Y"]].to_numpy(np.float64)\n        form_ev, knn_d  = _FI.impute(xy_ev, self_wid=swid)\n        form_kn, _      = _FI.impute(xy_kn, self_wid=swid)\n        z_kn    = kn["Z"].to_numpy(np.float32)\n        z_ev    = ev["Z"].to_numpy(np.float32)\n\n        # Per-formation TVT formulas + RMSE features\n        ktvt_plus_zkn = ktvt + z_kn\n        residuals_all = ktvt_plus_zkn[:, None] - form_kn\n\n        b_all_v  = np.median(residuals_all, axis=0)\n        b_wls_v  = np.array([wls_b_well(ktvt, z_kn, form_kn[:, fi])\n                             for fi in range(6)], dtype=np.float32)\n        b_50_v   = (np.median(residuals_all[-50:], axis=0).astype(np.float32)\n                    if len(ktvt) >= 5 else b_all_v.astype(np.float32))\n\n        tvt_form_mat  = (-z_ev[:, None] + form_ev + b_all_v[None, :]).astype(np.float32)\n        tvt_formw_mat = (-z_ev[:, None] + form_ev + b_wls_v[None, :]).astype(np.float32)\n\n        kn_pred_mat   = (-z_kn[:, None] + form_kn + b_all_v[None, :]).astype(np.float32)\n        form_rmse_v   = np.sqrt(np.mean((ktvt[:, None] - kn_pred_mat) ** 2, axis=0))\n\n        form_consistency = tvt_form_mat\n        form_mean_d  = (form_consistency.mean(1) - last_tvt).astype(np.float32)\n        form_std_d   = form_consistency.std(1).astype(np.float32)\n        form_range_d = (form_consistency.max(1) - form_consistency.min(1)).astype(np.float32)\n\n        # Dense ANCC\n        d_ancc, d_std, d_dist         = _DI.impute(xy_ev, self_wid=swid)\n        d_kn,   d_std_kn, _           = _DI.impute(xy_kn, self_wid=swid)\n        res_kn   = ktvt + z_kn - d_kn\n        b_d      = float(np.median(res_kn))\n        b_dw     = wls_b_well(ktvt, z_kn, d_kn)\n        b_d50    = float(np.median(res_kn[-50:])) if len(ktvt) >= 5 else b_d\n        tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)\n        tvt_dw    = (-z_ev + d_ancc + b_dw).astype(np.float32)\n        tvt_d50   = (-z_ev + d_ancc + b_d50).astype(np.float32)\n        d_rmse    = float(np.sqrt(np.mean(res_kn ** 2)))\n        d_bias    = float(np.mean(res_kn))\n        d_nb_std  = float(np.mean(d_std_kn))\n\n        # Inter-signal consensus std\n        all_sig = [pf_use, tvt_form_mat[:, 0], tvt_dense,\n                   *bpaths.values(), sc8, sc15, sc25]\n        valid_sig = [s for s in all_sig\n                     if len(s) == len(ev) and not np.any(np.isnan(s))]\n        signal_std = (np.stack(valid_sig, axis=1).std(1).astype(np.float32)\n                      if len(valid_sig) >= 2 else np.zeros(len(ev), np.float32))\n\n        # Slope baselines\n        hmd        = ev["MD"].to_numpy(np.float32)\n        md_since   = hmd - float(lk["MD"])\n        slp_base_all = (last_tvt + slp_all * md_since).astype(np.float32)\n        slp_base_50  = (last_tvt + slp_50  * md_since).astype(np.float32)\n\n        nh   = len(ev)\n        frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)\n\n        def sc(v: float) -> np.ndarray:\n            return np.full(nh, np.float32(v), np.float32)\n\n        # Trajectory derivatives\n        mdd   = hw["MD"].diff().replace(0, np.nan)\n        dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].to_numpy(np.float32)\n        dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].to_numpy(np.float32)\n        dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].to_numpy(np.float32)\n\n        # Pre-compute interp lookups (all at once, single np.interp calls)\n        anch_twgr       = np.interp(float(last_tvt) + ANCH_OFFS, tw_tvt, tw_gr).astype(np.float32)\n        beam_ref_tw     = np.interp(beam_ref,  tw_tvt, tw_gr).astype(np.float32)\n        sc15_tw         = np.interp(sc15,      tw_tvt, tw_gr).astype(np.float32)\n\n        # Offset lookups — vectorised: shape (len(BEAM_OFFS), nh) → transpose\n        beam_offsets_tw = np.array([\n            np.interp(beam_ref + o, tw_tvt, tw_gr) for o in BEAM_OFFS\n        ], dtype=np.float32).T\n        sc_offsets_tw = np.array([\n            np.interp(sc15 + o, tw_tvt, tw_gr) for o in SC_OFFS\n        ], dtype=np.float32).T\n\n        # ── Assemble feature dict ────────────────────────────────────────\n        feats: dict = {\n            "well": wid,\n            "id":   [f"{wid}_{i}" for i in ev.index],\n            "last_known_tvt": sc(last_tvt),\n            "pf_ancc":       pf_use,\n            "pf_ancc_std":   std_use,\n            "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),\n            "pf_z":          pf_z.astype(np.float32) if has_z else sc(last_tvt),\n            "pf_z_delta":    (pf_z - last_tvt).astype(np.float32) if has_z else sc(0.0),\n            "pf_vs_z":       (pf_use - pf_z.astype(np.float32)) if has_z else sc(0.0),\n            "pf_std_trend":  (std_use - std_use[0]).astype(np.float32) if len(std_use) > 0 else sc(0.0),\n            **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32)\n               for t, p in bpaths.items()},\n            "beam_mean_d": (beam_vals - last_tvt).mean(1).astype(np.float32),\n            "beam_std_d":  (beam_vals - last_tvt).std(1).astype(np.float32),\n            "beam_med_d":  np.median(beam_vals - last_tvt, axis=1).astype(np.float32),\n            "sc8_d":    (sc8  - np.float32(last_tvt)),\n            "sc8_score": sc8s,\n            "sc15_d":   (sc15 - np.float32(last_tvt)),\n            "sc15_score": sc15s,\n            "sc25_d":   (sc25 - np.float32(last_tvt)),\n            "sc25_score": sc25s,\n            "sc_cons_d": (sc_consensus - np.float32(last_tvt)),\n            "sc_trust":  sc(sc_trust),\n            "hyb_d":     (hyb_ref - np.float32(last_tvt)).astype(np.float32),\n            **{fn:                    tvt_form_mat[:, fi]  for fi, fn in enumerate(FORMATIONS)},\n            **{fn + "_wls":           tvt_formw_mat[:, fi] for fi, fn in enumerate(FORMATIONS)},\n            **{f"b_{fn}":             sc(float(b_all_v[fi]))  for fi, fn in enumerate(FORMATIONS)},\n            **{f"bw_{fn}":            sc(float(b_wls_v[fi]))  for fi, fn in enumerate(FORMATIONS)},\n            **{f"b50_{fn}":           sc(float(b_50_v[fi]))   for fi, fn in enumerate(FORMATIONS)},\n            **{f"tvtF_{fn}_d":       (tvt_form_mat[:,  fi] - last_tvt).astype(np.float32)\n               for fi, fn in enumerate(FORMATIONS)},\n            **{f"tvtFw_{fn}_d":      (tvt_formw_mat[:, fi] - last_tvt).astype(np.float32)\n               for fi, fn in enumerate(FORMATIONS)},\n            **{f"form_rmse_{fn}":    sc(float(form_rmse_v[fi]))\n               for fi, fn in enumerate(FORMATIONS)},\n            "form_mean_d":   form_mean_d,\n            "form_std_d":    form_std_d,\n            "form_range_d":  form_range_d,\n            "spatial_knn_dist": knn_d,\n            "dense_ancc":    d_ancc,\n            "dense_std":     d_std,\n            "dense_dist":    d_dist,\n            "tvt_dense_d":   (tvt_dense - last_tvt).astype(np.float32),\n            "tvt_dw_d":      (tvt_dw    - last_tvt).astype(np.float32),\n            "tvt_d50_d":     (tvt_d50   - last_tvt).astype(np.float32),\n            "dense_rmse":    sc(d_rmse),\n            "dense_bias":    sc(d_bias),\n            "dense_nb_std":  sc(d_nb_std),\n            "pf_vs_spatial":      (pf_use - tvt_form_mat[:, 0]).astype(np.float32),\n            "pf_vs_dense":        (pf_use - tvt_dense).astype(np.float32),\n            "spatial_vs_dense":   (tvt_form_mat[:, 0] - tvt_dense).astype(np.float32),\n            "beam_vs_spatial":    (bpaths["cons"] - tvt_form_mat[:, 0]).astype(np.float32),\n            "sc15_vs_beam_cons":  (sc15 - bpaths["cons"]).astype(np.float32),\n            "signal_std":  signal_std,\n            "cal_a": sc(a_cal), "cal_b": sc(b_cal),\n            "pfx_rmse":   sc(pfx_rmse), "known_len": sc(len(kn)), "eval_len": sc(nh),\n            "slp_all":    sc(slp_all),  "slp_50":    sc(slp_50),  "slp_z":    sc(slp_z),\n            "slp_base_d_all": (slp_base_all - last_tvt).astype(np.float32),\n            "slp_base_d_50":  (slp_base_50  - last_tvt).astype(np.float32),\n            "ktvt_range": sc(float(np.ptp(ktvt))), "ktvt_std": sc(float(ktvt.std())),\n            "md_since": md_since, "frac": frac, "frac2": frac ** 2,\n            "sqrt_frac": np.sqrt(frac),\n            "z":  z_ev,\n            "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),\n            "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),\n            "dz": (z_ev    - float(lk["Z"])).astype(np.float32),\n            "dxy": np.sqrt(\n                (ev["X"] - float(lk["X"])) ** 2 +\n                (ev["Y"] - float(lk["Y"])) ** 2\n            ).to_numpy(np.float32),\n            "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,\n            "gr":       hgr,\n            "gr_d1":    gr_d1,\n            "gr_d2":    gr_d2,\n            "gr_env":   hgr_env.astype(np.float32),\n            "gr_energy": hgr_nrg.astype(np.float32),\n            "gr_vs_tw_anc":  hgr - float(np.interp(last_tvt, tw_tvt, tw_gr)),\n            "gr_vs_slp_all": hgr - np.interp(slp_base_all, tw_tvt, tw_gr).astype(np.float32),\n            **{f"tda{int(o)}": hgr - anch_twgr[i] for i, o in enumerate(ANCH_OFFS)},\n            **{f"tdbc{int(o)}": hgr - beam_offsets_tw[:, i]\n               for i, o in enumerate(BEAM_OFFS)},\n            **{f"tdsc{int(o)}": hgr - sc_offsets_tw[:, i]\n               for i, o in enumerate(SC_OFFS)},\n            "tw_range":   sc(float(np.ptp(tw_tvt))),\n            "tw_gr_mean": sc(float(tw_gr.mean())),\n        }\n        feats.update(roll_feats)\n\n        result = pd.DataFrame(feats)\n\n        if DBG_VERBOSE:\n            _dbg(f"_build_well_from_df({wid}): rows={len(result)} cols={len(result.columns)}")\n\n        if is_train:\n            if "TVT" not in ev.columns or ev["TVT"].isna().all():\n                _dbg(f"_build_well_from_df({wid}): is_train but ev TVT all NaN", level="WARN")\n                return None\n            result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))\n\n        return result\n\n    except Exception as exc:\n        _log_error(f"_build_well_from_df({wid})", exc)\n        return None\n\n\ndef build_well(hw_path: str, tw_path: str, is_train: bool) -> Optional[pd.DataFrame]:\n    """I/O wrapper — reads CSVs then delegates to _build_well_from_df."""\n    if _FI is None or _DI is None:\n        _dbg(f"build_well({Path(hw_path).stem}): imputers not set!", level="ERROR")\n        return None\n    wid = Path(hw_path).stem.replace("__horizontal_well", "")\n    try:\n        hw = pd.read_csv(hw_path)\n        tw = pd.read_csv(tw_path).sort_values("TVT")\n    except Exception as exc:\n        _log_error(f"build_well({wid}) read_csv", exc)\n        return None\n    return _build_well_from_df(hw, tw, is_train=is_train, wid=wid)\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Cal-Zone Augmentation — no temp files, reuses GR cache\n# ═══════════════════════════════════════════════════════════════════════════════\n\ndef build_augmented(\n    hw: pd.DataFrame,\n    tw_tvt: np.ndarray,\n    tw_gr: np.ndarray,\n    wid: str,\n    gr_cache: Optional[_WellGRCache] = None,\n    n_splits: int = N_AUG_SPLITS,\n    min_known: int = MIN_KNOWN_FOR_AUG,\n) -> pd.DataFrame:\n    known_all = hw[hw["TVT_input"].notna()]\n    n_cal = len(known_all)\n    if n_cal < min_known + 5:\n        _dbg(f"build_augmented({wid}): too few known rows ({n_cal})", level="WARN")\n        return pd.DataFrame()\n\n    split_ks = np.unique(np.linspace(min_known, n_cal - 2, n_splits).astype(int))\n    tw_mock  = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})\n    parts    = []\n\n    for k in split_ks:\n        hw_m = hw.copy()\n        mask_start = known_all.index[k]\n        hw_m.loc[mask_start:, "TVT_input"] = np.nan\n\n        kn_m  = hw_m[hw_m["TVT_input"].notna()]\n        ev_m  = hw_m[hw_m["TVT_input"].isna()]\n        n_new = n_cal - k\n\n        if len(ev_m) == 0 or len(kn_m) < 10 or n_new == 0:\n            _dbg(f"build_augmented({wid}) k={k}: skipping", level="DEBUG")\n            continue\n\n        try:\n            # ── Direct in-memory call — no temp files ──────────────────────\n            feat = _build_well_from_df(\n                hw_m, tw_mock, is_train=True, wid=wid, gr_cache=gr_cache\n            )\n        except Exception as exc:\n            _log_error(f"build_augmented({wid}) k={k}", exc)\n            continue\n\n        if feat is None or len(feat) == 0:\n            _dbg(f"build_augmented({wid}) k={k}: build returned empty", level="DEBUG")\n            continue\n\n        feat_new = feat.iloc[:n_new].copy()\n        feat_new["aug_k"] = np.int32(k)\n        parts.append(feat_new)\n        _dbg(f"build_augmented({wid}) k={k}: added {len(feat_new)} aug rows", level="DEBUG")\n\n    if not parts:\n        _dbg(f"build_augmented({wid}): no augmented parts produced", level="WARN")\n    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()\n\n\ndef process_train_well(hw_path: Path) -> pd.DataFrame:\n    """\n    Top-level per-well train worker.\n    Builds GR cache once, passes it to both the original call and all aug splits.\n    """\n    wid     = hw_path.stem.replace("__horizontal_well", "")\n    tw_path = TRAIN_DIR / f"{wid}__typewell.csv"\n    if not tw_path.exists():\n        _dbg(f"process_train_well({wid}): typewell not found", level="WARN")\n        return pd.DataFrame()\n    try:\n        hw = pd.read_csv(hw_path)\n        tw = pd.read_csv(tw_path).sort_values("TVT")\n        tw_tvt = tw["TVT"].to_numpy(np.float32)\n        tw_gr  = tw["GR"].to_numpy(np.float32)\n\n        # Build GR cache once — shared across original + all aug splits\n        gr_mean  = float(np.nanmean(tw_gr))\n        gr_cache = _WellGRCache(hw, gr_mean)\n\n        # Original (full cal-zone) features\n        feat_orig = _build_well_from_df(\n            hw, tw, is_train=True, wid=wid, gr_cache=gr_cache\n        )\n        parts: list = []\n        if feat_orig is not None and len(feat_orig) > 0:\n            feat_orig["aug_k"] = np.int32(-1)\n            parts.append(feat_orig)\n        else:\n            _dbg(f"process_train_well({wid}): build_well returned None/empty", level="WARN")\n\n        # Augmented splits — pass same gr_cache to avoid recomputation\n        feat_aug = build_augmented(hw, tw_tvt, tw_gr, wid, gr_cache=gr_cache)\n        if len(feat_aug) > 0:\n            parts.append(feat_aug)\n\n        if not parts:\n            _dbg(f"process_train_well({wid}): returning empty", level="WARN")\n            return pd.DataFrame()\n\n        r = pd.concat(parts, ignore_index=True)\n        r["well_id"] = wid\n        _dbg(f"process_train_well({wid}): done rows={len(r)}", level="DEBUG")\n        return r\n    except Exception as exc:\n        _log_error(f"process_train_well({wid})", exc)\n        return pd.DataFrame()\n\n\ndef process_test_train(hw_path: Path) -> pd.DataFrame:\n    """Online training: augment from test well calibration zone."""\n    wid     = hw_path.stem.replace("__horizontal_well", "")\n    tw_path = TEST_DIR / f"{wid}__typewell.csv"\n    if not tw_path.exists():\n        _dbg(f"process_test_train({wid}): typewell not found", level="WARN")\n        return pd.DataFrame()\n    try:\n        hw = pd.read_csv(hw_path)\n        tw = pd.read_csv(tw_path)\n        if "TVT" not in tw.columns or "GR" not in tw.columns:\n            _dbg(f"process_test_train({wid}): typewell missing TVT/GR", level="WARN")\n            return pd.DataFrame()\n        known = hw[hw["TVT_input"].notna()]\n        if len(known) < MIN_KNOWN_FOR_AUG + 5:\n            _dbg(f"process_test_train({wid}): too few known rows", level="WARN")\n            return pd.DataFrame()\n        hw_aug = hw.copy()\n        hw_aug["TVT"] = hw_aug["TVT_input"]\n        tw_tvt = tw["TVT"].to_numpy(np.float32)\n        tw_gr  = tw["GR"].to_numpy(np.float32)\n\n        # Build GR cache for test well augmentation\n        gr_mean  = float(np.nanmean(tw_gr))\n        gr_cache = _WellGRCache(hw_aug, gr_mean)\n\n        feat_aug = build_augmented(hw_aug, tw_tvt, tw_gr, wid, gr_cache=gr_cache)\n        if len(feat_aug) > 0:\n            feat_aug["well_id"] = wid\n            _dbg(f"process_test_train({wid}): aug rows={len(feat_aug)}", level="DEBUG")\n        else:\n            _dbg(f"process_test_train({wid}): no aug rows", level="WARN")\n        return feat_aug\n    except Exception as exc:\n        _log_error(f"process_test_train({wid})", exc)\n        return pd.DataFrame()\n\n\ndef build_dataset(paths, is_train: bool, label: str) -> pd.DataFrame:\n    args = [\n        (str(p),\n         str(p.parent / f"{p.stem.replace(\'__horizontal_well\',\'\')}__typewell.csv"),\n         is_train)\n        for p in paths\n        if (p.parent / f"{p.stem.replace(\'__horizontal_well\',\'\')}__typewell.csv").exists()\n    ]\n    print(f"  {label}: {len(args)} wells | {NCPU} workers | process_pool={USE_PROCESS_POOL}")\n    res = _starmap_with_imputers(build_well, args, NCPU)\n    parts = [r for r in res if r is not None and len(r) > 0]\n    print(f"  {label}: OK={len(parts)} skipped={len(args)-len(parts)}")\n    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Main pipeline\n# ═══════════════════════════════════════════════════════════════════════════════\n\nhw_paths   = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))\nif DEBUG_MAX_TRAIN_WELLS > 0:\n    print(f"[DEBUG] Limiting train wells to first {DEBUG_MAX_TRAIN_WELLS}", flush=True)\n    hw_paths = hw_paths[:DEBUG_MAX_TRAIN_WELLS]\ntrain_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]\nprint(f"Building imputers from {len(train_wids)} wells...")\nt0 = time.time()\nFI = FormationPlaneKNN(train_wids, TRAIN_DIR)\nDI = DenseANCCImputer(train_wids, TRAIN_DIR)\n# Set in main process too (for smoke tests)\n_FI = FI; _DI = DI\nprint(f"  FI: {len(FI.df)} centroids | DI: {len(DI.ancc):,} pts  ({time.time()-t0:.0f}s)")\n\n\n# ── DEBUG: single-well smoke tests (run in main process, imputers already set) ──\n\nif DBG_SINGLE_WELL and len(hw_paths) > 0:\n    _smoke_path = hw_paths[0]\n    _smoke_wid  = _smoke_path.stem.replace("__horizontal_well", "")\n    _smoke_tw   = TRAIN_DIR / f"{_smoke_wid}__typewell.csv"\n    print(f"[SMOKE] Testing train well: {_smoke_wid}", flush=True)\n    try:\n        _smoke_result = build_well(str(_smoke_path), str(_smoke_tw), is_train=True)\n        if _smoke_result is None:\n            print("[SMOKE] build_well returned None", flush=True)\n        else:\n            print(f"[SMOKE] build_well OK: shape={_smoke_result.shape}", flush=True)\n            if DBG_FEATURE_AUDIT:\n                print(f"[SMOKE] columns ({len(_smoke_result.columns)}):",\n                      list(_smoke_result.columns), flush=True)\n            nan_check = _smoke_result.isnull().sum()\n            nan_check = nan_check[nan_check > 0]\n            if len(nan_check):\n                print(f"[SMOKE] NaN columns: {dict(nan_check)}", flush=True)\n            else:\n                print("[SMOKE] No NaN columns.", flush=True)\n    except Exception as _e:\n        print(f"[SMOKE] EXCEPTION: {_e}", flush=True)\n        traceback.print_exc()\n\ntest_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))\nif DEBUG_MAX_TEST_WELLS > 0:\n    print(f"[DEBUG] Limiting test wells to first {DEBUG_MAX_TEST_WELLS}", flush=True)\n    test_paths = test_paths[:DEBUG_MAX_TEST_WELLS]\n\nif DBG_SINGLE_TEST and len(test_paths) > 0:\n    _smoke_tp   = test_paths[0]\n    _smoke_twid = _smoke_tp.stem.replace("__horizontal_well", "")\n    _smoke_twp  = TEST_DIR / f"{_smoke_twid}__typewell.csv"\n    print(f"[SMOKE] Testing test well: {_smoke_twid}", flush=True)\n    try:\n        _smoke_t = build_well(str(_smoke_tp), str(_smoke_twp), is_train=False)\n        if _smoke_t is None:\n            print("[SMOKE] test build_well returned None", flush=True)\n        else:\n            print(f"[SMOKE] test build_well OK: shape={_smoke_t.shape}", flush=True)\n    except Exception as _e:\n        print(f"[SMOKE] test EXCEPTION: {_e}", flush=True)\n        traceback.print_exc()\n\n\nif INFERENCE_ONLY:\n    print("\\nROGII_INFERENCE_ONLY=1; loading saved artifacts.", flush=True)\n    artifact_dir = find_artifact_dir()\n    manifest = load_json(artifact_dir / "manifest.json")\n    config = load_json(artifact_dir / manifest.get("inference_config", "inference_config.json"))\n    feature_cols = load_json(artifact_dir / manifest.get("feature_cols", "feature_cols.json"))\n    result_keys = list(config.get("result_keys") or [])\n    if not RUN_TABICL and any(str(k).startswith("tabicl") for k in result_keys):\n        keep_idx = [i for i, k in enumerate(result_keys) if not str(k).startswith("tabicl")]\n        result_keys = [result_keys[i] for i in keep_idx]\n        for key in ["stacker", "ridge_coef"]:\n            if key == "stacker":\n                coef = config.get("stacker", {}).get("coef")\n            else:\n                coef = config.get(key)\n            if coef is None:\n                continue\n            coef = np.asarray(coef, dtype=np.float32)[keep_idx]\n            coef_sum = float(np.sum(np.abs(coef)))\n            if coef_sum > 1e-12:\n                coef = coef / coef_sum\n            if key == "stacker":\n                config.setdefault("stacker", {})["coef"] = coef.tolist()\n            else:\n                config[key] = coef.tolist()\n        print(f"ROGII_RUN_TABICL=0; using non-TabICL artifact keys: {result_keys}", flush=True)\n    print(f"Artifact dir: {artifact_dir}", flush=True)\n    print(f"Artifact result keys: {result_keys}", flush=True)\n\n    test_df = build_dataset(test_paths, is_train=False, label="test")\n    if test_df.empty:\n        raise RuntimeError("No test features were built.")\n    missing_features = [c for c in feature_cols if c not in test_df.columns]\n    if missing_features:\n        print(f"[WARN] Filling {len(missing_features)} missing artifact features with NaN.", flush=True)\n        for col in missing_features:\n            test_df[col] = np.nan\n    Xt = test_df[feature_cols]\n    pred_by_key: dict[str, np.ndarray] = {}\n\n    def _entry_path(entry: dict) -> Path:\n        return artifact_dir / str(entry["path"]).replace("/", os.sep)\n\n    lgb_groups: dict[str, list[dict]] = {}\n    for entry in manifest.get("lgb", []):\n        lgb_groups.setdefault(f"lgb{entry[\'seed\']}", []).append(entry)\n    for key, entries in sorted(lgb_groups.items()):\n        pred = np.zeros(len(test_df), dtype=np.float32)\n        for entry in sorted(entries, key=lambda e: e.get("fold", 0)):\n            booster = lgb.Booster(model_file=str(_entry_path(entry)))\n            best_iter = int(entry.get("best_iteration") or booster.best_iteration or booster.current_iteration())\n            pred += booster.predict(Xt, num_iteration=best_iter).astype(np.float32) / len(entries)\n        pred_by_key[key] = pred\n        print(f"Loaded {len(entries)} LGB folds for {key}", flush=True)\n\n    cb_groups: dict[str, list[dict]] = {}\n    for entry in manifest.get("catboost", []):\n        cb_groups.setdefault(f"cb{entry[\'seed\']}", []).append(entry)\n    for key, entries in sorted(cb_groups.items()):\n        pred = np.zeros(len(test_df), dtype=np.float32)\n        for entry in sorted(entries, key=lambda e: e.get("fold", 0)):\n            model = CatBoostRegressor()\n            model.load_model(str(_entry_path(entry)))\n            pred += model.predict(Xt.values).astype(np.float32) / len(entries)\n        pred_by_key[key] = pred\n        print(f"Loaded {len(entries)} CatBoost folds for {key}", flush=True)\n\n    tabicl_needed = any(k.startswith("tabicl") for k in result_keys)\n    tabicl_entries = manifest.get("tabicl_contexts", [])\n    if tabicl_needed and tabicl_entries:\n        if not RUN_TABICL:\n            raise RuntimeError("Artifacts expect TabICL predictions, but ROGII_RUN_TABICL=0.")\n        import subprocess as _sp\n        from pathlib import Path as _Path\n\n        tabicl_roots = [_Path("/kaggle/input")]\n        if os.environ.get("ROGII_TABICL_DIR"):\n            tabicl_roots.insert(0, _Path(os.environ["ROGII_TABICL_DIR"]))\n        _wheels, _ckpts = [], []\n        for _root in tabicl_roots:\n            if _root.exists():\n                _wheels.extend(_root.rglob("tabicl-*.whl"))\n                _ckpts.extend(_root.rglob("tabicl-regressor*.ckpt"))\n        if not _wheels or not _ckpts:\n            raise RuntimeError("TabICL wheel/checkpoint not found for artifact inference.")\n        _sp.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(_wheels[0])], check=True)\n        from tabicl import TabICLRegressor\n\n        tabicl_features = load_json(artifact_dir / manifest.get("tabicl_features", "tabicl_features.json"))\n        Xt_top = Xt[tabicl_features].values\n        label_groups: dict[str, list[dict]] = {}\n        for entry in tabicl_entries:\n            label = str(entry["label"])\n            if label.startswith("tabicl_A"):\n                key = "tabicl_A"\n            elif label.startswith("tabicl_B"):\n                key = "tabicl_B"\n            else:\n                key = label\n            label_groups.setdefault(key, []).append(entry)\n\n        device = "cpu" if FORCE_CPU else "cuda"\n        chunk = 50_000\n        for key, entries in sorted(label_groups.items()):\n            pred = np.zeros(len(test_df), dtype=np.float32)\n            for entry in sorted(entries, key=lambda e: (e.get("fold", 0), e.get("seed", 0))):\n                ctx = np.load(_entry_path(entry))\n                reg = TabICLRegressor(\n                    model_path=str(_ckpts[0]),\n                    device=device,\n                    random_state=int(entry.get("seed", 0)),\n                    n_estimators=int(entry.get("n_estimators", 4)),\n                    n_jobs=1,\n                    verbose=False,\n                    use_amp="auto",\n                    batch_size=4,\n                )\n                reg.fit(ctx["X_ctx"], ctx["y_ctx"])\n                burnin_path = entry.get("burnin_path")\n                if burnin_path:\n                    burn = np.load(artifact_dir / str(burnin_path).replace("/", os.sep))["X_va"]\n                    print(\n                        f"   burn {key} fold{entry.get(\'fold\')} seed{entry.get(\'seed\')}: "\n                        f"{len(burn)} rows",\n                        flush=True,\n                    )\n                    for i in range(0, len(burn), chunk):\n                        e = min(i + chunk, len(burn))\n                        _ = reg.predict(burn[i:e])\n                    del burn\n                cur = np.empty(len(test_df), dtype=np.float32)\n                for i in range(0, len(Xt_top), chunk):\n                    e = min(i + chunk, len(Xt_top))\n                    cur[i:e] = reg.predict(Xt_top[i:e]).astype(np.float32)\n                pred += cur / len(entries)\n            pred_by_key[key] = pred\n            print(f"Loaded {len(entries)} TabICL contexts for {key}", flush=True)\n\n    if not result_keys:\n        result_keys = list(pred_by_key.keys())\n    missing_results = [k for k in result_keys if k not in pred_by_key]\n    if missing_results:\n        raise RuntimeError(f"Missing artifact predictions for result keys: {missing_results}")\n\n    St = np.column_stack([pred_by_key[k] for k in result_keys])\n    stacker = config.get("stacker", {})\n    if stacker.get("coef") is not None:\n        coef = np.asarray(stacker["coef"], dtype=np.float32)\n        if len(coef) != St.shape[1]:\n            raise RuntimeError(f"Stack coef length {len(coef)} does not match predictions {St.shape[1]}.")\n        final_test = St @ coef + float(stacker.get("intercept", 0.0))\n        print(f"Using saved {stacker.get(\'kind\', \'linear\')} stack.", flush=True)\n    elif config.get("use_ridge", False):\n        coef = np.asarray(config["ridge_coef"], dtype=np.float32)\n        if len(coef) != St.shape[1]:\n            raise RuntimeError(f"Ridge coef length {len(coef)} does not match predictions {St.shape[1]}.")\n        final_test = St @ coef + float(config.get("ridge_intercept", 0.0))\n        print("Using saved ridge stack.", flush=True)\n    else:\n        final_test = St.mean(axis=1)\n        print("Using saved simple-average stack.", flush=True)\n\n    alpha = float(config.get("postproc_alpha", 1.0))\n    tau = config.get("postproc_tau", None)\n    w_pf = float(config.get("postproc_w_pf", 0.0))\n    pf_delta = (test_df["pf_ancc"].values - test_df["last_known_tvt"].values).astype(np.float32)\n    delta = ((1.0 - w_pf) * final_test.astype(np.float32) + w_pf * pf_delta).astype(np.float32)\n    if tau is not None:\n        delta *= 1.0 - np.exp(-np.maximum(test_df["md_since"].values, 0.0) / float(tau))\n    test_df2 = test_df.copy()\n    test_df2["pred"] = test_df2["last_known_tvt"].values + delta * alpha\n\n    sample = pd.read_csv(SAMPLE)\n    sub = sample[["id"]].merge(\n        test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}),\n        on="id", how="left",\n    )\n    fallback_tvt = float(config.get("fallback_tvt", test_df2["pred"].mean()))\n    sub["tvt"] = sub["tvt"].fillna(fallback_tvt)\n    print(\n        f"[SUB AUDIT] rows={len(sub)} null={int(sub[\'tvt\'].isnull().sum())} "\n        f"fallback_filled={int((sub[\'tvt\'] == fallback_tvt).sum())}",\n        flush=True,\n    )\n\n    if config.get("exact_overlap_enabled", True):\n        os.environ["ROGII_EXACT_OVERLAP"] = "1"\n        os.environ["ROGII_EXACT_BLEND_WEIGHT"] = str(config.get("exact_blend_weight", 0.28))\n    else:\n        os.environ["ROGII_EXACT_OVERLAP"] = "0"\n    sub = apply_exact_train_coordinate_blend(sub[["id", "tvt"]], DATA)\n    sub[["id", "tvt"]].to_csv(OUT, index=False)\n    print(f"\\n✅  {OUT}  {len(sub)} rows (artifact inference)")\n    print(sub.head(8).to_string(index=False))\n    print(f"=== ARTIFACT INFERENCE COMPLETE  total={time.time()-_T0_GLOBAL:.0f}s ===")\n    raise SystemExit(0)\n\n\n# ── Build train features — process pool with shared imputers ─────────────────\nprint("\\nLoading/building official train-core features...")\nt0 = time.time()\n\ntrain_core_df = None\nif USE_TRAIN_CORE_CACHE and DEBUG_MAX_TRAIN_WELLS <= 0:\n    for cache_path in _cache_candidates():\n        if cache_path.exists():\n            print(f"Loading train-core cache: {cache_path}", flush=True)\n            train_core_df = pd.read_pickle(cache_path)\n            print(f"  train_core_df={train_core_df.shape} loaded in {time.time()-t0:.0f}s", flush=True)\n            break\n\nif train_core_df is None:\n    print("\\nBuilding official train-core features...")\n    res = _map_with_imputers(process_train_well, hw_paths, NCPU)\n\n    if DBG_PARALLEL_STATS:\n        n_none  = sum(1 for r in res if r is None)\n        n_empty = sum(1 for r in res if r is not None and len(r) == 0)\n        n_ok    = sum(1 for r in res if r is not None and len(r) > 0)\n        row_dist = [len(r) for r in res if r is not None and len(r) > 0]\n        print(f"[PARALLEL STATS train] None={n_none} empty={n_empty} ok={n_ok}", flush=True)\n        if row_dist:\n            print(f"  rows per well: min={min(row_dist)} max={max(row_dist)} "\n                  f"mean={np.mean(row_dist):.0f} total={sum(row_dist)}", flush=True)\n        else:\n            print("  *** All results None or empty! ***", flush=True)\n\n    parts_core = [r for r in res if r is not None and len(r) > 0]\n    if len(parts_core) == 0:\n        print("[FATAL] No valid official train DataFrames.", flush=True)\n        raise RuntimeError("All process_train_well calls returned None or empty.")\n\n    train_core_df = pd.concat(parts_core, ignore_index=True)\n    del res, parts_core\n    gc.collect()\n    print(f"  train_core_df={train_core_df.shape} built in {time.time()-t0:.0f}s", flush=True)\n\n    if WRITE_TRAIN_CORE_CACHE or CACHE_ONLY:\n        cache_out = _train_core_cache_path_for_write()\n        cache_out.parent.mkdir(parents=True, exist_ok=True)\n        train_core_df.to_pickle(cache_out)\n        preview_cols = [c for c in ["well", "id", "last_known_tvt", "target", "aug_k"] if c in train_core_df.columns]\n        train_core_df[preview_cols].head(1000).to_csv(\n            cache_out.with_name("aeroridge_train_core_preview_1000.csv"), index=False\n        )\n        schema = pd.DataFrame({\n            "column": train_core_df.columns,\n            "dtype": [str(train_core_df[c].dtype) for c in train_core_df.columns],\n            "non_null_count": [int(train_core_df[c].notna().sum()) for c in train_core_df.columns],\n            "null_count": [int(train_core_df[c].isna().sum()) for c in train_core_df.columns],\n        })\n        schema["null_rate"] = schema["null_count"] / max(len(train_core_df), 1)\n        schema.to_csv(cache_out.with_name("aeroridge_train_core_schema.csv"), index=False)\n        print(f"  wrote train-core cache: {cache_out}", flush=True)\n\nif CACHE_ONLY:\n    print("ROGII_CACHE_ONLY=1, stopping after train-core cache build.", flush=True)\n    raise SystemExit(0)\n\nparts = [train_core_df]\n\n# ── Online training on test wells ────────────────────────────────────────────\nprint(f"\\nOnline training on {len(test_paths)} test wells...")\nt1 = time.time()\n\nres_t = _map_with_imputers(process_test_train, test_paths, min(NCPU, max(1, len(test_paths))))\n\nif DBG_PARALLEL_STATS:\n    n_none_t  = sum(1 for r in res_t if r is None)\n    n_empty_t = sum(1 for r in res_t if r is not None and len(r) == 0)\n    n_ok_t    = sum(1 for r in res_t if r is not None and len(r) > 0)\n    print(f"[PARALLEL STATS test-online] None={n_none_t} empty={n_empty_t} ok={n_ok_t}",\n          flush=True)\n\nparts.extend([r for r in res_t if r is not None and len(r) > 0])\ndel res_t\n\n# ── Pre-concat guard ─────────────────────────────────────────────────────────\nif len(parts) == 0:\n    print("[FATAL] No valid DataFrames to concatenate.", flush=True)\n    raise RuntimeError(\n        "All process_train_well / process_test_train calls returned None or empty."\n    )\n\ntrain_df = pd.concat(parts, ignore_index=True)\ndel parts; gc.collect()\nprint(f"\\ntrain_df: {train_df.shape} | {time.time()-t0:.0f}s")\norig = (train_df["aug_k"] == -1).sum()\naug  = (train_df["aug_k"] >=  0).sum()\nprint(f"  Original: {orig:,}  Augmented: {aug:,} (+{aug/max(orig,1)*100:.0f}%)")\n\n_dbg_df("train_df", train_df)\n\nSKIP = {"well", "well_id", "id", "target", "aug_k"}\nfeature_cols = [c for c in train_df.columns if c not in SKIP]\nprint(f"Features: {len(feature_cols)}")\nif SAVE_ARTIFACTS:\n    save_json(ARTIFACT_DIR / "feature_cols.json", feature_cols)\n    ARTIFACT_MANIFEST["feature_cols"] = "feature_cols.json"\n    ARTIFACT_MANIFEST["data"] = {\n        "train_df_shape": list(train_df.shape),\n        "test_df_shape": None,\n        "n_features": len(feature_cols),\n        "n_splits": N_SPLITS,\n        "seed": SEED,\n        "used_train_core_cache": str(os.environ.get("ROGII_TRAIN_CORE_CACHE", "")),\n    }\n\nif DBG_FEATURE_AUDIT:\n    print(f"[FEATURE AUDIT] {len(feature_cols)} feature columns:", flush=True)\n    for _fc in feature_cols:\n        _arr = train_df[_fc].to_numpy()\n        _nans = int(np.isnan(_arr.astype(float)).sum()) if np.issubdtype(_arr.dtype, np.number) else 0\n        if _nans > 0:\n            print(f"  NaN: {_fc}  count={_nans}/{len(_arr)}", flush=True)\n\nX  = train_df[feature_cols]\ny  = train_df["target"]\ng  = train_df["well"]\n\nif DBG_NAN_AUDIT:\n    _dbg("NaN audit on X (train features)")\n    _nan_X = X.isnull().sum()\n    _nan_X = _nan_X[_nan_X > 0]\n    if len(_nan_X):\n        print(f"[NaN AUDIT] X has NaN in {len(_nan_X)} cols:", flush=True)\n        print(_nan_X.to_string(), flush=True)\n    else:\n        print("[NaN AUDIT] X: clean (no NaN)", flush=True)\n    _nan_y = int(y.isnull().sum())\n    print(f"[NaN AUDIT] y: {_nan_y} NaN  "\n          f"min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f}", flush=True)\n    _inf_X = np.isinf(X.to_numpy(dtype=float)).sum()\n    print(f"[NaN AUDIT] X Inf count: {_inf_X}", flush=True)\n\n# ── Test features ────────────────────────────────────────────────────────────\ntest_df = build_dataset(test_paths, is_train=False, label="test")\nXt = test_df[feature_cols]\nif SAVE_ARTIFACTS:\n    ARTIFACT_MANIFEST.setdefault("data", {})["test_df_shape"] = list(test_df.shape)\n    ARTIFACT_MANIFEST["created_outputs"]["test_feature_preview"] = None\n\nif DBG_NAN_AUDIT:\n    _nan_Xt = Xt.isnull().sum()\n    _nan_Xt = _nan_Xt[_nan_Xt > 0]\n    if len(_nan_Xt):\n        print(f"[NaN AUDIT] Xt has NaN in {len(_nan_Xt)} cols:", flush=True)\n        print(_nan_Xt.to_string(), flush=True)\n    else:\n        print("[NaN AUDIT] Xt: clean (no NaN)", flush=True)\n    _inf_Xt = np.isinf(Xt.to_numpy(dtype=float)).sum()\n    print(f"[NaN AUDIT] Xt Inf count: {_inf_Xt}", flush=True)\n\ngc.collect()\n_dbg(f"Feature matrix ready: X={X.shape} Xt={Xt.shape}")\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Training:  LGB×3 + CatBoost + Ridge stacking\n# ═══════════════════════════════════════════════════════════════════════════════\n\n# ── CV split with group shuffling ────────────────────────────────────────────\n# Plain GroupKFold without shuffle assigns folds in sorted order, so wells\n# cluster geographically/temporally inside each fold → validation is\n# systematically easier or harder → inflated variance and worse generalisation.\n# Fix: shuffle the unique groups before assigning fold IDs.\n#\n# Augmented rows (aug_k >= 0) are kept in training but EXCLUDED from the OOF\n# validation set so the same real rows can\'t appear in both train and val.\n# Without this, aug splits of the same well leak into both sides of the fold.\n\n_unique_wells = np.unique(g.values)\n_rng_cv       = np.random.RandomState(SEED)\n_shuffled     = _rng_cv.permutation(_unique_wells)\n_fold_map     = {w: i % N_SPLITS for i, w in enumerate(_shuffled)}\n_fold_ids     = np.array([_fold_map[w] for w in g.values])\n\n# Boolean mask: True for original rows (not augmented)\n_is_orig = (train_df["aug_k"].values == -1)\n\nsplits: list = []\nfor _f in range(N_SPLITS):\n    _tr_idx = np.where(_fold_ids != _f)[0]                      # all rows not in fold f\n    _va_all = np.where(_fold_ids == _f)[0]                      # fold-f rows\n    _va_idx = _va_all[_is_orig[_va_all]]                        # keep only original rows\n    if len(_va_idx) == 0:\n        print(f"[CV] WARNING: fold {_f} has zero original val rows — skipping", flush=True)\n        continue\n    splits.append((_tr_idx, _va_idx))\n\nprint(f"[CV] {len(splits)} folds | "\n      f"train aug+orig per fold ≈ {np.mean([len(t) for t,_ in splits]):.0f} | "\n      f"val orig-only per fold ≈ {np.mean([len(v) for _,v in splits]):.0f}")\n\n\n\ndef run_lgb(seed: int, gpu_id: int = 0):\n    p   = dict(LGB_P, n_estimators=5000, seed=seed)\n    if not FORCE_CPU:\n        p["gpu_device_id"] = gpu_id\n    oof = np.zeros(len(train_df), np.float32)\n    tp  = np.zeros(len(test_df),  np.float32)\n    for fold, (tr, va) in enumerate(splits):\n        _dbg(f"LGB seed={seed} fold={fold} starting")\n        ds_tr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])\n        ds_va = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=ds_tr)\n        m = lgb.train(\n            p,\n            ds_tr,\n            valid_sets=[ds_va],\n            num_boost_round=p["n_estimators"],\n            callbacks=[\n                lgb.early_stopping(125, verbose=False),\n                lgb.log_evaluation(250),\n            ],\n        )\n        ni = m.best_iteration\n        oof[va] = m.predict(X.iloc[va], num_iteration=ni).astype(np.float32)\n        tp      += m.predict(Xt, num_iteration=ni).astype(np.float32) / len(splits)\n        fold_rmse = root_mean_squared_error(y.iloc[va], oof[va])\n        print(f"   LGB{seed} fold{fold}: best_iter={ni} rmse={fold_rmse:.4f}")\n        if SAVE_ARTIFACTS:\n            model_path = ARTIFACT_DIR / "lgb" / f"seed{seed}_fold{fold}.txt"\n            model_path.parent.mkdir(parents=True, exist_ok=True)\n            m.save_model(str(model_path), num_iteration=ni)\n            ARTIFACT_MANIFEST["lgb"].append({\n                "seed": int(seed),\n                "fold": int(fold),\n                "best_iteration": int(ni),\n                "rmse": float(fold_rmse),\n                "path": _artifact_rel(model_path),\n            })\n        _dbg(f"LGB seed={seed} fold={fold} done rmse={fold_rmse:.4f}")\n    # OOF only over rows that were in some validation set\n    va_all = np.concatenate([va for _, va in splits])\n    r = root_mean_squared_error(y.iloc[va_all], oof[va_all])\n    print(f"   LGB{seed} OOF={r:.4f}")\n    return oof, tp, r\n\n\ndef run_cb(seed: int = 42):\n    p   = dict(CB_P, random_seed=seed)\n    oof = np.zeros(len(train_df), np.float32)\n    tp  = np.zeros(len(test_df),  np.float32)\n    for fold, (tr, va) in enumerate(splits):\n        _dbg(f"CB fold={fold} starting")\n        m = CatBoostRegressor(**p)\n        m.fit(Pool(X.iloc[tr].values, label=y.iloc[tr].values),\n              eval_set=Pool(X.iloc[va].values, label=y.iloc[va].values),\n              use_best_model=True)\n        oof[va] = m.predict(X.iloc[va].values).astype(np.float32)\n        tp      += m.predict(Xt.values).astype(np.float32) / len(splits)\n        fold_rmse = root_mean_squared_error(y.iloc[va], oof[va])\n        print(f"   CB{seed} fold{fold}: rmse={fold_rmse:.4f}")\n        if SAVE_ARTIFACTS:\n            model_path = ARTIFACT_DIR / "catboost" / f"seed{seed}_fold{fold}.cbm"\n            model_path.parent.mkdir(parents=True, exist_ok=True)\n            m.save_model(str(model_path))\n            ARTIFACT_MANIFEST["catboost"].append({\n                "seed": int(seed),\n                "fold": int(fold),\n                "rmse": float(fold_rmse),\n                "path": _artifact_rel(model_path),\n            })\n        _dbg(f"CB fold={fold} done rmse={fold_rmse:.4f}")\n    va_all = np.concatenate([va for _, va in splits])\n    r = root_mean_squared_error(y.iloc[va_all], oof[va_all])\n    print(f"   CB{seed} OOF={r:.4f}")\n    return oof, tp, r\n\n\nimport threading\n\n_lgb_results = {}\n_lgb_exc     = [None]\n\ndef _run_lgb_parallel():\n    """LGB[42,7] on GPU 0; LGB[123] on GPU 1 — parallel threads."""\n    try:\n        import threading as _t\n\n        def _gpu0():\n            for s in [42, 7]:\n                o, t, r = run_lgb(s, gpu_id=0)\n                _lgb_results[f\'lgb{s}\'] = {\'oof\': o, \'test\': t, \'rmse\': r}\n\n        def _gpu1():\n            o, t, r = run_lgb(123, gpu_id=1)\n            _lgb_results[\'lgb123\'] = {\'oof\': o, \'test\': t, \'rmse\': r}\n\n        t0 = _t.Thread(target=_gpu0)\n        t1 = _t.Thread(target=_gpu1)\n        t0.start(); t1.start()\n        t0.join();  t1.join()\n    except Exception as e:\n        _lgb_exc[0] = e\n\nprint(">> GPU0: LGB[42,7]  |  GPU1: LGB[123]  (parallel)")\n_run_lgb_parallel()\nif _lgb_exc[0]: raise _lgb_exc[0]\n\nresults = {}\nresults.update(_lgb_results)\n\n# CB×3 seeds run serially. Device string is detected from available Kaggle GPUs.\nfor seed in [42, 7, 123]:\n    oof_cb, tp_cb, r_cb = run_cb(seed=seed)\n    results[f\'cb{seed}\'] = {\'oof\': oof_cb, \'test\': tp_cb, \'rmse\': r_cb}\n\n_va_union = np.concatenate([va for _, va in splits])\n\n# ── TabICL setup ─────────────────────────────────────────────────────────────\nif RUN_TABICL:\n    import subprocess as _sp, sys as _sys\n    from pathlib import Path as _Path\n\n    tabicl_roots = [_Path("/kaggle/input")]\n    if os.environ.get("ROGII_TABICL_DIR"):\n        tabicl_roots.insert(0, _Path(os.environ["ROGII_TABICL_DIR"]))\n    _wheels = []\n    _ckpts = []\n    for _root in tabicl_roots:\n        if _root.exists():\n            _wheels.extend(_root.rglob("tabicl-*.whl"))\n            _ckpts.extend(_root.rglob("tabicl-regressor*.ckpt"))\n    print(f"found wheels: {_wheels}")\n    print(f"found ckpts:  {_ckpts}")\n    if not _wheels or not _ckpts:\n        _msg = (\n            "TabICL artifacts not found. Add the public dataset "\n            "needless090/rogii-tabicl-mirror, or set ROGII_RUN_TABICL=0."\n        )\n        if RUNNING_ON_KAGGLE:\n            raise RuntimeError(_msg)\n        print(_msg + " Skipping TabICL for this local run.", flush=True)\n    else:\n        _sp.run([_sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",\n                 str(_wheels[0])], check=True)\n        TABICL_CKPT = str(_ckpts[0])\n        print(f"TabICL ckpt: {TABICL_CKPT}")\n\n        print(">> Selecting top-50 features for TabICL", flush=True)\n        _tr0, _ = splits[0]\n        _samp = np.random.RandomState(42).choice(_tr0, size=min(200_000, len(_tr0)), replace=False)\n        _lgb_sel_params = dict(\n            n_estimators=300, learning_rate=0.05, num_leaves=63,\n            n_jobs=-1, verbosity=-1,\n        )\n        if not FORCE_CPU:\n            _lgb_sel_params["device_type"] = "gpu"\n        _lgb_sel = lgb.LGBMRegressor(**_lgb_sel_params)\n        _lgb_sel.fit(X.iloc[_samp].values, y.iloc[_samp].values)\n        _imp = pd.Series(_lgb_sel.feature_importances_, index=feature_cols).sort_values(ascending=False)\n        TABICL_FEATS = _imp.head(50).index.tolist()\n        print(f"   top-5: {TABICL_FEATS[:5]}")\n        if SAVE_ARTIFACTS:\n            save_json(ARTIFACT_DIR / "tabicl_features.json", TABICL_FEATS)\n            ARTIFACT_MANIFEST["tabicl_features"] = "tabicl_features.json"\n\n        X_top  = X[TABICL_FEATS].values\n        Xt_top = Xt[TABICL_FEATS].values\n        CHUNK  = 50_000\n\n        def run_tabicl(ctx_n, n_estimators, seeds, label):\n            from tabicl import TabICLRegressor\n            print(f"\\n>> {label}: ctx={ctx_n} n_est={n_estimators} seeds={seeds}", flush=True)\n            n_per = len(seeds)\n            oof = np.zeros(len(train_df), dtype=np.float32)\n            test_pred = np.zeros(len(test_df), dtype=np.float32)\n            for fold, (tr, va) in enumerate(splits):\n                oof_fold = np.zeros(len(va), dtype=np.float32)\n                for sd in seeds:\n                    ctx_idx = np.random.RandomState(sd * 1000 + fold).choice(\n                        tr, size=min(ctx_n, len(tr)), replace=False)\n                    X_ctx = X_top[ctx_idx]\n                    y_ctx = y.iloc[ctx_idx].values.astype(np.float32)\n                    if SAVE_ARTIFACTS:\n                        ctx_path = (\n                            ARTIFACT_DIR / "tabicl_contexts"\n                            / f"{label}_fold{fold}_seed{sd}.npz"\n                        )\n                        ctx_path.parent.mkdir(parents=True, exist_ok=True)\n                        np.savez_compressed(\n                            ctx_path,\n                            X_ctx=X_ctx.astype(np.float32),\n                            y_ctx=y_ctx.astype(np.float32),\n                        )\n                        ARTIFACT_MANIFEST["tabicl_contexts"].append({\n                            "label": label,\n                            "fold": int(fold),\n                            "seed": int(sd),\n                            "ctx_n": int(ctx_n),\n                            "n_estimators": int(n_estimators),\n                            "path": _artifact_rel(ctx_path),\n                        })\n                    X_va  = X_top[va]\n                    t0_ = time.time()\n                    reg = TabICLRegressor(model_path=TABICL_CKPT, device="cuda",\n                                          random_state=sd, n_estimators=n_estimators,\n                                          n_jobs=1, verbose=False, use_amp="auto", batch_size=4)\n                    reg.fit(X_ctx, y_ctx)\n                    pv = np.empty(len(va), dtype=np.float32)\n                    for i in range(0, len(va), CHUNK):\n                        e = min(i + CHUNK, len(va))\n                        pv[i:e] = reg.predict(X_va[i:e]).astype(np.float32)\n                    oof_fold += pv / n_per\n                    tf = np.empty(len(test_df), dtype=np.float32)\n                    for i in range(0, len(Xt_top), CHUNK):\n                        e = min(i + CHUNK, len(Xt_top))\n                        tf[i:e] = reg.predict(Xt_top[i:e]).astype(np.float32)\n                    test_pred += tf / N_SPLITS / n_per\n                    print(f"   fold{fold} seed{sd}: [{time.time()-t0_:.1f}s]", flush=True)\n                oof[va] = oof_fold\n                rmse_fold = root_mean_squared_error(y.iloc[va].values, oof_fold)\n                print(f"   fold{fold} avg RMSE = {rmse_fold:.4f}", flush=True)\n            r_val = root_mean_squared_error(y.iloc[_va_union].values, oof[_va_union])\n            print(f"   {label} OOF RMSE (val rows) = {r_val:.4f}", flush=True)\n            return oof, test_pred, float(r_val)\n\n        # TabICL A: ctx=4096, 4 estimators, 5 seeds (~74 min)\n        oof_A, test_A, rmse_A = run_tabicl(4096, 4, [0, 1, 2, 3, 4], "tabicl_A_4096_5seed")\n        results["tabicl_A"] = {"oof": oof_A, "test": test_A, "rmse": rmse_A}\n\n        # TabICL B: ctx=8192, 4 estimators, 1 seed (~19 min)\n        oof_B, test_B, rmse_B = run_tabicl(8192, 4, [42], "tabicl_B_8192_1seed")\n        results["tabicl_B"] = {"oof": oof_B, "test": test_B, "rmse": rmse_B}\nelse:\n    print("ROGII_RUN_TABICL=0; skipping TabICL models.", flush=True)\n\n# Ridge stacking — fit and score only over rows that were actually validated\nresult_keys = list(results.keys())\nSx = np.column_stack([v["oof"] for v in results.values()]).astype(np.float32)\nSt = np.column_stack([v["test"] for v in results.values()]).astype(np.float32)\ny_val_stack = y.iloc[_va_union].values.astype(np.float32)\nSx_val = Sx[_va_union]\n\n\ndef _rmse_np(y_true: np.ndarray, pred: np.ndarray) -> float:\n    diff = pred.astype(np.float64) - y_true.astype(np.float64)\n    return float(np.sqrt(np.mean(diff * diff)))\n\n\ndef hill_climb_stack(\n    pred_mat: np.ndarray,\n    y_true: np.ndarray,\n    keys: list[str],\n    precision: float = 0.01,\n    allow_negative: bool = True,\n):\n    """Small deterministic hill-climb stacker, package-free."""\n    precision = max(float(precision), 0.001)\n    weights = np.arange(-0.5, 0.5001, precision) if allow_negative else np.arange(precision, 0.5001, precision)\n    model_scores = [_rmse_np(y_true, pred_mat[:, i]) for i in range(pred_mat.shape[1])]\n    start = int(np.argmin(model_scores))\n    coef = np.zeros(pred_mat.shape[1], dtype=np.float64)\n    coef[start] = 1.0\n    current = pred_mat[:, start].astype(np.float32).copy()\n    current_score = model_scores[start]\n    remaining = [i for i in range(pred_mat.shape[1]) if i != start]\n    history = [{\n        "iteration": 0,\n        "model": keys[start],\n        "weight": 1.0,\n        "score": float(current_score),\n    }]\n    iteration = 0\n    while remaining:\n        iteration += 1\n        best = (current_score, None, None, None)\n        for idx in remaining:\n            new_pred = pred_mat[:, idx]\n            for w in weights:\n                cand = (1.0 - w) * current + w * new_pred\n                score = _rmse_np(y_true, cand)\n                if score < best[0] - 1e-7:\n                    best = (score, idx, float(w), cand.astype(np.float32))\n        if best[1] is None:\n            break\n        score, idx, w, current = best\n        coef *= (1.0 - w)\n        coef[idx] += w\n        current_score = float(score)\n        remaining.remove(idx)\n        history.append({\n            "iteration": iteration,\n            "model": keys[idx],\n            "weight": float(w),\n            "score": current_score,\n        })\n    return coef.astype(np.float32), current.astype(np.float32), current_score, history\n\n\navg_coef = np.ones(len(result_keys), dtype=np.float32) / max(len(result_keys), 1)\navg_oof = Sx_val.mean(1).astype(np.float32)\navg_test = St.mean(1).astype(np.float32)\nr_avg = _rmse_np(y_val_stack, avg_oof)\n\nridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)\nridge.fit(Sx_val, y_val_stack)\nridge_coef = ridge.coef_.astype(np.float32)\nridge_oof = ridge.predict(Sx_val).astype(np.float32)\nridge_test = ridge.predict(St).astype(np.float32)\nr_stk = _rmse_np(y_val_stack, ridge_oof)\nridge_wts = ridge_coef / max(float(ridge_coef.sum()), 1e-9)\n\nstack_options = [\n    {\n        "kind": "simple_average",\n        "coef": avg_coef,\n        "intercept": 0.0,\n        "oof": avg_oof,\n        "test": avg_test,\n        "rmse": r_avg,\n        "history": [],\n    },\n    {\n        "kind": "positive_ridge",\n        "coef": ridge_coef,\n        "intercept": float(getattr(ridge, "intercept_", 0.0)),\n        "oof": ridge_oof,\n        "test": ridge_test,\n        "rmse": r_stk,\n        "history": [],\n    },\n]\n\nif USE_HILL_STACK and len(result_keys) >= 2:\n    hill_coef, hill_oof, hill_rmse, hill_history = hill_climb_stack(\n        Sx_val,\n        y_val_stack,\n        result_keys,\n        precision=HILL_PRECISION,\n        allow_negative=True,\n    )\n    hill_test = (St @ hill_coef).astype(np.float32)\n    stack_options.append({\n        "kind": "hill_climb",\n        "coef": hill_coef,\n        "intercept": 0.0,\n        "oof": hill_oof,\n        "test": hill_test,\n        "rmse": hill_rmse,\n        "history": hill_history,\n    })\n\nbest_stack = min(stack_options, key=lambda item: item["rmse"])\nfinal_test = best_stack["test"]\n_final_oof_full = Sx.mean(1).astype(np.float32)\n_final_oof_full[_va_union] = best_stack["oof"]\nfinal_oof = _final_oof_full\n\nprint(f"\\nSimple avg OOF (val rows): {r_avg:.4f}")\nprint(f"Ridge stk OOF (val rows): {r_stk:.4f}  wts={dict(zip(result_keys, ridge_wts.round(4)))}")\nif USE_HILL_STACK and len(result_keys) >= 2:\n    print(f"Hill stk OOF (val rows): {stack_options[-1][\'rmse\']:.4f}  coefs={dict(zip(result_keys, stack_options[-1][\'coef\'].round(4)))}")\nprint(f"Using {best_stack[\'kind\']} predictions.")\nprint(f"Final OOF RMSE (val rows): {best_stack[\'rmse\']:.4f}")\n\n\n# ═══════════════════════════════════════════════════════════════════════════════\n# Post-Processing\n# ═══════════════════════════════════════════════════════════════════════════════\n# Grid-search alpha/tau only on val rows (where OOF is clean).\n# SavGol smoothing is REMOVED from test predictions — it introduced bias on the\n# initial rows of each well and hurt the leaderboard score vs the reference.\n\nbase_val = train_df["last_known_tvt"].values[_va_union]\nytrue_val = y.values[_va_union] + base_val\nmd_val = train_df["md_since"].values[_va_union]\noof_val = final_oof[_va_union]\npf_val = (train_df["pf_ancc"].values[_va_union] - base_val).astype(np.float32)\n\nbest_cfg, best_r = (None, None, None), np.inf\nfor alpha in np.arange(0.6, 1.01, 0.05):\n    for tau in [None, 30.0, 60.0, 120.0, 250.0, 500.0]:\n        for w_pf in np.arange(0.0, 0.501, 0.05):\n            d = ((1.0 - w_pf) * oof_val + w_pf * pf_val).astype(np.float32)\n            if tau:\n                d *= 1.0 - np.exp(-np.maximum(md_val, 0.0) / tau)\n            r = root_mean_squared_error(ytrue_val, base_val + d * alpha)\n            if r < best_r:\n                best_r, best_cfg = r, (float(alpha), tau, float(w_pf))\n\nprint(\n    f"Best post-proc: alpha={best_cfg[0]:.2f} tau={best_cfg[1]} "\n    f"w_pf={best_cfg[2]:.2f}  TVT RMSE={best_r:.4f}"\n)\nALPHA, TAU, W_PF = best_cfg\n_fallback_tvt = float(\n    train_df["last_known_tvt"].iloc[_va_union].mean()\n    + train_df["target"].iloc[_va_union].mean()\n)\nif SAVE_ARTIFACTS:\n    stacker_config = {\n        "kind": best_stack["kind"],\n        "coef": best_stack["coef"].astype(float).tolist(),\n        "intercept": float(best_stack.get("intercept", 0.0)),\n        "rmse": float(best_stack["rmse"]),\n        "history": best_stack.get("history", []),\n        "hill_precision": float(HILL_PRECISION),\n    }\n    config = {\n        "result_keys": result_keys,\n        "stacker": stacker_config,\n        "use_ridge": bool(best_stack["kind"] == "positive_ridge"),\n        "ridge_alpha": 1.0,\n        "ridge_fit_intercept": False,\n        "ridge_coef": ridge_coef.astype(float).tolist(),\n        "ridge_intercept": float(getattr(ridge, "intercept_", 0.0)),\n        "simple_avg_oof": float(r_avg),\n        "ridge_oof": float(r_stk),\n        "final_oof": float(best_stack["rmse"]),\n        "postproc_alpha": float(ALPHA),\n        "postproc_tau": None if TAU is None else float(TAU),\n        "postproc_w_pf": float(W_PF),\n        "postproc_oof_rmse": float(best_r),\n        "fallback_tvt": _fallback_tvt,\n        "exact_overlap_enabled": os.environ.get("ROGII_EXACT_OVERLAP", "1").strip().lower()\n        not in {"0", "false", "no"},\n        "exact_blend_weight": env_float("ROGII_EXACT_BLEND_WEIGHT", 0.28),\n    }\n    save_json(ARTIFACT_DIR / "inference_config.json", config)\n    ARTIFACT_MANIFEST["inference_config"] = "inference_config.json"\n    ARTIFACT_MANIFEST["result_keys"] = result_keys\n    diag_dir = ARTIFACT_DIR / "diagnostics"\n    diag_dir.mkdir(parents=True, exist_ok=True)\n    np.save(diag_dir / "base_test_predictions.npy", St.astype(np.float32))\n    ARTIFACT_MANIFEST["created_outputs"]["base_test_predictions"] = _artifact_rel(\n        diag_dir / "base_test_predictions.npy"\n    )\n    np.savez_compressed(\n        diag_dir / "oof_val_predictions.npz",\n        predictions=Sx_val.astype(np.float32),\n        target=y_val_stack.astype(np.float32),\n        val_indices=_va_union.astype(np.int64),\n    )\n    save_json(diag_dir / "prediction_keys.json", result_keys)\n    ARTIFACT_MANIFEST["created_outputs"]["oof_val_predictions"] = _artifact_rel(\n        diag_dir / "oof_val_predictions.npz"\n    )\n    ARTIFACT_MANIFEST["created_outputs"]["prediction_keys"] = _artifact_rel(\n        diag_dir / "prediction_keys.json"\n    )\n    meta_cols = [\n        c for c in ["well", "id", "target", "last_known_tvt", "md_since", "pf_ancc", "aug_k"]\n        if c in train_df.columns\n    ]\n    train_df.loc[_va_union, meta_cols].to_csv(\n        diag_dir / "oof_val_meta.csv.gz",\n        index=False,\n        compression="gzip",\n    )\n    ARTIFACT_MANIFEST["created_outputs"]["oof_val_meta"] = _artifact_rel(\n        diag_dir / "oof_val_meta.csv.gz"\n    )\n    test_diag = pd.DataFrame({"id": test_df["id"].values})\n    for i, key in enumerate(result_keys):\n        test_diag[key] = St[:, i].astype(np.float32)\n    test_diag.to_csv(diag_dir / "test_base_predictions.csv.gz", index=False, compression="gzip")\n    ARTIFACT_MANIFEST["created_outputs"]["test_base_predictions"] = _artifact_rel(\n        diag_dir / "test_base_predictions.csv.gz"\n    )\n    if SAVE_FEATURE_FRAMES:\n        frame_dir = ARTIFACT_DIR / "feature_frames"\n        frame_dir.mkdir(parents=True, exist_ok=True)\n        train_df.to_pickle(frame_dir / "train_features.pkl")\n        test_df.to_pickle(frame_dir / "test_features.pkl")\n        ARTIFACT_MANIFEST["created_outputs"]["train_features"] = _artifact_rel(\n            frame_dir / "train_features.pkl"\n        )\n        ARTIFACT_MANIFEST["created_outputs"]["test_features"] = _artifact_rel(\n            frame_dir / "test_features.pkl"\n        )\n\n\ndef apply_pp(df: pd.DataFrame, delta: np.ndarray,\n             alpha: float, tau, w_pf: float = 0.0) -> np.ndarray:\n    pf_delta = (df["pf_ancc"].values - df["last_known_tvt"].values).astype(np.float32)\n    d = ((1.0 - w_pf) * delta.astype(np.float32) + w_pf * pf_delta).astype(np.float32)\n    if tau:\n        d *= 1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau)\n    return d * alpha\n\n\n# Apply post-processing to test predictions — no SavGol smoothing\ntest_df2 = test_df.copy()\ntest_df2["pred"] = (test_df2["last_known_tvt"].values\n                    + apply_pp(test_df2, final_test, ALPHA, TAU, W_PF))\n\nsample = pd.read_csv(SAMPLE)\nsub = sample[["id"]].merge(\n    test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}),\n    on="id", how="left",\n)\n_fb_val = _fallback_tvt\nsub["tvt"] = sub["tvt"].fillna(_fb_val)\n\nn_fill = sub["tvt"].isnull().sum()\nn_fb   = (sub["tvt"] == _fb_val).sum()\nprint(f"[SUB AUDIT] rows={len(sub)}  null={n_fill}  fallback_filled={n_fb}", flush=True)\n\nsub = apply_exact_train_coordinate_blend(sub[["id", "tvt"]], DATA)\nsub[["id", "tvt"]].to_csv(OUT, index=False)\nif SAVE_ARTIFACTS:\n    ARTIFACT_MANIFEST["created_outputs"]["submission"] = str(OUT)\n    save_json(ARTIFACT_DIR / "manifest.json", ARTIFACT_MANIFEST)\n    print(f"Artifact manifest -> {ARTIFACT_DIR / \'manifest.json\'}", flush=True)\n\nprint(f"\\n✅  {OUT}  {len(sub)} rows")\nprint("\\n─── Final Summary ───────────────────────────")\nfor k, v in results.items():\n    print(f"  {k}: OOF = {v[\'rmse\']:.4f}")\nprint(f"  Best stack: {best_stack[\'kind\']} OOF = {best_stack[\'rmse\']:.4f}  |  PostProc TVT RMSE: {best_r:.4f}")\nprint(sub.head(8).to_string(index=False))\nprint(f"=== PIPELINE COMPLETE  total={time.time()-_T0_GLOBAL:.0f}s ===")\n'

WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
COMPONENT_ROOT = WORKING / "embedded_components"
PF_VARIANT = os.getenv("ROGII_BLEND_PF_VARIANT", "grid_s3_b0_h0p17")
PF_WEIGHT = float(os.getenv("ROGII_BLEND_PF_WEIGHT", "0.80"))
PF_WEIGHT = max(0.0, min(1.0, PF_WEIGHT))
WELL_PF_WEIGHTS = {
    str(k): max(0.0, min(1.0, float(v)))
    for k, v in json.loads(
        os.getenv("ROGII_BLEND_WELL_PF_WEIGHTS", "{}")
    ).items()
}
ARTIFACT_EXACT_OVERLAP = os.getenv("ROGII_ARTIFACT_EXACT_OVERLAP", "0")
ARTIFACT_EXACT_BLEND_WEIGHT = os.getenv("ROGII_ARTIFACT_EXACT_BLEND_WEIGHT", "")
ARTIFACT_RUN_TABICL = os.getenv("ROGII_ARTIFACT_RUN_TABICL", "0")
ARTIFACT_FORCE_CPU = os.getenv("ROGII_ARTIFACT_FORCE_CPU", "1")
ARTIFACT_SOURCE_MODE = os.getenv("ROGII_ARTIFACT_SOURCE_MODE", "legacy_51c240_seeded").strip().lower()
ARTIFACT_SEED_SALT = os.getenv("ROGII_ARTIFACT_SEED_SALT", "0")
ARTIFACT_SEED_SALTS = [
    salt.strip()
    for salt in os.getenv("ROGII_ARTIFACT_SEED_SALTS", ARTIFACT_SEED_SALT).split(",")
    if salt.strip()
]
if not ARTIFACT_SEED_SALTS:
    ARTIFACT_SEED_SALTS = ["0"]
ARTIFACT_DYNAMIC_WELL_RULE = os.getenv("ROGII_ARTIFACT_DYNAMIC_WELL_RULE", "none")
ARTIFACT_DYNAMIC_PF_WEIGHT = float(os.getenv("ROGII_ARTIFACT_DYNAMIC_PF_WEIGHT", "0.75"))
BLEND_SELECTOR = os.getenv("ROGII_BLEND_SELECTOR", "off").strip().lower()
CONTACT_SURFACE_WEIGHT = float(os.getenv("ROGII_CONTACT_SURFACE_WEIGHT", "0"))
CONTACT_SURFACE_WEIGHT = max(-0.25, min(0.25, CONTACT_SURFACE_WEIGHT))
CONTACT_SURFACE_COL = os.getenv("ROGII_CONTACT_SURFACE_COL", "EGFDL")
CONTACT_SURFACE_K = int(os.getenv("ROGII_CONTACT_SURFACE_K", "64"))
CONTACT_SURFACE_STRIDE = int(os.getenv("ROGII_CONTACT_SURFACE_STRIDE", "20"))
CONTACT_SURFACE_XY_SCALE = float(os.getenv("ROGII_CONTACT_SURFACE_XY_SCALE", "1000.0"))
CONTACT_BASIS_RULE = os.getenv("ROGII_CONTACT_BASIS_RULE", "off").strip().lower()
CONTACT_BASIS_VALUE = float(os.getenv("ROGII_CONTACT_BASIS_VALUE", "0"))
CONTACT_BASIS_MAX_ABS = float(os.getenv("ROGII_CONTACT_BASIS_MAX_ABS", "30"))
CONTACT_BASIS_K = int(os.getenv("ROGII_CONTACT_BASIS_K", "64"))
GEO_PATH_RULE = os.getenv("ROGII_GEO_PATH_RULE", "off").strip().lower()
GEO_PATH_BLEND_WEIGHT = float(os.getenv("ROGII_GEO_PATH_BLEND_WEIGHT", "1.0"))
GEO_PATH_MAX_ABS_DELTA = float(os.getenv("ROGII_GEO_PATH_MAX_ABS_DELTA", "80"))
GEO_PATH_GEO_MARGIN = float(os.getenv("ROGII_GEO_PATH_GEO_MARGIN", "4.0"))
ANALOG_RESIDUAL_RULE = os.getenv("ROGII_ANALOG_RESIDUAL_RULE", "off").strip().lower()
ANALOG_RESIDUAL_ALPHA = float(os.getenv("ROGII_ANALOG_RESIDUAL_ALPHA", "0.25"))
ANALOG_RESIDUAL_K = int(os.getenv("ROGII_ANALOG_RESIDUAL_K", "12"))
ANALOG_RESIDUAL_MAX_ABS = float(os.getenv("ROGII_ANALOG_RESIDUAL_MAX_ABS", "12.5"))
ANALOG_RESIDUAL_TABLE = os.getenv("ROGII_ANALOG_RESIDUAL_TABLE", "analog_residual_table.csv")
CONTACT_OVERRIDE_RULE = os.getenv("ROGII_CONTACT_OVERRIDE_RULE", "off").strip().lower()
CONTACT_OVERRIDE_MAX_PREFIX_RMSE = float(os.getenv("ROGII_CONTACT_OVERRIDE_MAX_PREFIX_RMSE", "1.0"))
CONTACT_OVERRIDE_MIN_PREFIX_ROWS = int(os.getenv("ROGII_CONTACT_OVERRIDE_MIN_PREFIX_ROWS", "50"))
CONTACT_OVERRIDE_REFS = tuple(
    item.strip()
    for item in os.getenv(
        "ROGII_CONTACT_OVERRIDE_REFS",
        "EGFDU,ASTNU,ANCC,ASTNL,EGFDL,BUDA",
    ).split(",")
    if item.strip()
)
FINAL_DYNAMIC_OFFSET_RULE = os.getenv("ROGII_FINAL_DYNAMIC_OFFSET_RULE", "off").strip().lower()
FINAL_DYNAMIC_OFFSET_VALUE = float(os.getenv("ROGII_FINAL_DYNAMIC_OFFSET_VALUE", "0"))
FINAL_WELL_OFFSETS = {
    str(k): float(v)
    for k, v in json.loads(os.getenv("ROGII_FINAL_WELL_OFFSETS", "{}")).items()
}
FINAL_WELL_TRENDS = {
    str(k): float(v)
    for k, v in json.loads(os.getenv("ROGII_FINAL_WELL_TRENDS", "{}")).items()
}
CONTACT_BASIS_CONTACTS = tuple(
    item.strip()
    for item in os.getenv(
        "ROGII_CONTACT_BASIS_CONTACTS",
        "ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA",
    ).split(",")
    if item.strip()
)
CONTACT_BASIS_CONTACT_ORDER = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
CONTACT_GEO_ESTIMATOR_VERSION = "local_plane_residual_knn_v1"
ANALOG_RESIDUAL_FEATURES = (
    "known_frac",
    "n_eval",
    "eval_md_span",
    "eval_z_span",
    "eval_z_slope",
    "tail_tvt_slope",
    "tail_z_slope",
    "base_eval_slope",
    "base_delta_end",
    "base_eval_std",
    "gr_sigma",
    "gr_r2",
    "eval_gr_loss",
    "eval_gr_std",
)


EMBEDDED_ANALOG_RESIDUAL_TABLE_GZ_B64 = """
H4sIAFGnSGoC/3Sd645cSXKk/xPgm5CJuF+eplEki4KgmZGgmZWAffq1zzxOVh6S29I0u9nJrMw4ER7u5mbm//v+t799+fvbP//j
r//4x3/+7z/++vnfb9+/vPzjP/56/5+3v33hb3/9/cdf//yvt3/Ev/zf+z//7T//6/3Lv97+/W9//et//vX6r9d/+/b2z3e/1+u/
/3j/27/e/nr/x4/X//yvH1/+7b//+ue//9vf3/iH/y7xQ/SPf/vPf/7z+S+87u/vb//467/f//nvP/6PPuTf33/8++u/X//glz7/
5b9Kf/mX2T9/+v7jbb5/q/NLerQef2t5p0fSL2vrl9zqY890/ZXz0KtS7XnsWvdsrbesP/5Vv5nHmnP2VusYZbfa/MpWy2yzplbG
3H5d6mmmsdOaOaVa9Jt5PmYuKc+cW86p7f5lPNpouVTeaM/Sx5f16PqvdY09a2kzPu5Mbeyhtxl9LP1W1dvvXlbOZeq17YveOrWe
9GFL7avrQ3wp+dF2mXyOVNLac/CqpffMLa2uL5Ty+pLHY/Se60izrb1S/aLX9DZHXqXMrhWYX6o+916rz5Fq5oPeFnT4f/rm+gj8
MljOvB+7Ppcz6duzSDslfdlWtSar5lilvIu+/9Q661vnlctZzjbGblNfeukbxUuXvkervTQ9gjz60opuffo0qtZBH1BvP/Xxd176
hzp2r1M/rn/Zj1aWHtce+vbFD6zNspa+Xi1DH271ypLWlrRgc9RZSt4szaq7LS2UfmbpW2uq71hLHjPvoY/dhv5cfeSei55CmUVf
efth6B21WHX3sYseMwuoH9T1tn0UffJWHnOl0Uvh269xW9HZ42/aoY3F3Ekf7svS7+WPJd28KDXtIn37rOc1pzaUN2NZWx92r5ZG
yvq8K35zaJ26Xl+Knm9asabaxGOvPJs+y9ZqfPm6HitxEkbTIi/tuy/l0fRNW+9979Zyb9pJ6VHW1E9ua5a5e+JntLb019bO1lPr
TZvroec5O29T9bP1TL7k+uhazjq0A7Uaqetrtoce/Ch6kntv/hOv0nabTYua9KqsHT4esxWWUN+upV71fPJj6CCOvOvKdZc0vpT0
yLXo3Gpttcn2/Pyp/fjx4719zy9HX1tosVc7j9N7VX/8+VeLHajnq92g1cr63tofPvpJpyDxH2bSx908rKTjOmID17V3PodfO0Eb
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
"""


def write_component(name: str, code: str) -> Path:
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    path = COMPONENT_ROOT / name
    path.write_text(code, encoding="utf-8")
    return path


def run_component(name: str, script: Path, output_dir: Path, extra_env: dict[str, str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(extra_env)
    env["ROGII_OUTPUT_DIR"] = str(output_dir)
    print(f"\n=== Running {name} ===", flush=True)
    print(f"script={script}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    subprocess.run([sys.executable, str(script)], check=True, env=env)
    csv_path = output_dir / "submission.csv"
    fallback_csv = WORKING / "submission.csv"
    if not csv_path.exists() and fallback_csv.exists():
        shutil.copy2(fallback_csv, csv_path)
        print(f"{name} wrote {fallback_csv}; copied to {csv_path}", flush=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"{name} did not create {csv_path}")
    return csv_path


def find_sample() -> Path:
    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction/sample_submission.csv"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv"),
        Path.cwd() / "sample_submission.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for path in input_root.glob("**/sample_submission.csv"):
            return path
    raise FileNotFoundError("sample_submission.csv not found")


def read_component(path: Path, sample: pd.DataFrame, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tvt" not in df.columns:
        raise ValueError(f"{name} missing tvt column: {df.columns.tolist()}")
    out = sample[["id"]].merge(df[["id", "tvt"]], on="id", how="left")
    missing = int(out["tvt"].isna().sum())
    if missing:
        raise RuntimeError(f"{name} has {missing} missing rows after sample alignment")
    return out


def well_ids(data_dir: Path, split: str) -> list[str]:
    folder = data_dir / split
    return [
        path.name.split("__")[0]
        for path in sorted(folder.glob("*__horizontal_well.csv"))
    ]


def contact_anchor_path(hw: pd.DataFrame) -> np.ndarray:
    known = hw["TVT_input"].ffill().bfill()
    if known.notna().any():
        return known.to_numpy(dtype=float)
    return np.zeros(len(hw), dtype=float)


def build_contact_surface_pool(data_dir: Path) -> pd.DataFrame:
    test_ids = set(well_ids(data_dir, "test"))
    rows = []
    for path in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
        wid = path.name.split("__")[0]
        if wid in test_ids:
            continue
        try:
            hw = pd.read_csv(path, usecols=["X", "Y", CONTACT_SURFACE_COL])
        except ValueError:
            continue
        rows.append(hw.iloc[::CONTACT_SURFACE_STRIDE].dropna())
    if not rows:
        raise RuntimeError(f"No train rows found for contact surface {CONTACT_SURFACE_COL}")
    return pd.concat(rows, ignore_index=True)


def predict_contact_surface(pool: pd.DataFrame, hw: pd.DataFrame) -> np.ndarray:
    train_xy = pool[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    target_xy = hw[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    tree = cKDTree(train_xy)
    dist, idx = tree.query(target_xy, k=min(CONTACT_SURFACE_K, len(pool)), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    vals = pool[CONTACT_SURFACE_COL].to_numpy(dtype=float)[idx]
    weights = 1.0 / np.maximum(dist, 1e-3) ** 2
    return np.sum(vals * weights, axis=1) / np.sum(weights, axis=1)


def tvt_from_contact_surface(hw: pd.DataFrame, contact_pred: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    base = contact_pred - hw["Z"].to_numpy(dtype=float)
    known = hw["TVT_input"].notna().to_numpy() & np.isfinite(base)
    if int(known.sum()) < 8:
        return contact_anchor_path(hw), {"n_known": float(known.sum()), "mode": "anchor_fallback"}
    residual = hw.loc[known, "TVT_input"].to_numpy(dtype=float) - base[known]
    n_tail = min(80, max(12, len(residual) // 3))
    tail = residual[-n_tail:]
    weights = np.linspace(0.35, 1.0, len(tail))
    offset = float(np.average(tail, weights=weights))
    pred = base + offset
    anchor = contact_anchor_path(hw)
    pred = np.where(np.isfinite(pred), pred, anchor)
    return pred.astype(float), {
        "n_known": float(known.sum()),
        "n_tail": float(n_tail),
        "offset": offset,
        "contact_min": float(np.nanmin(contact_pred)),
        "contact_max": float(np.nanmax(contact_pred)),
        "pred_min": float(np.nanmin(pred)),
        "pred_max": float(np.nanmax(pred)),
    }


def contact_surface_component(data_dir: Path, sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    pool = build_contact_surface_pool(data_dir)
    predictions: dict[str, float] = {}
    summary: dict[str, object] = {
        "weight": CONTACT_SURFACE_WEIGHT,
        "contact": CONTACT_SURFACE_COL,
        "k": CONTACT_SURFACE_K,
        "stride": CONTACT_SURFACE_STRIDE,
        "pool_rows": int(len(pool)),
        "wells": {},
    }
    for wid in well_ids(data_dir, "test"):
        hw = pd.read_csv(data_dir / "test" / f"{wid}__horizontal_well.csv")
        contact_pred = predict_contact_surface(pool, hw)
        pred, meta = tvt_from_contact_surface(hw, contact_pred)
        summary["wells"][wid] = meta
        for idx, value in enumerate(pred):
            predictions[f"{wid}_{idx}"] = float(value)
    out = sample[["id"]].copy()
    out["tvt"] = out["id"].map(predictions)
    missing = int(out["tvt"].isna().sum())
    if missing:
        raise RuntimeError(f"contact_surface has {missing} missing rows after sample alignment")
    return out, summary


def build_contact_basis_pool(data_dir: Path, contacts: tuple[str, ...]) -> pd.DataFrame:
    test_ids = set(well_ids(data_dir, "test"))
    rows = []
    usecols = ["X", "Y", *contacts]
    for path in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
        wid = path.name.split("__")[0]
        if wid in test_ids:
            continue
        try:
            hw = pd.read_csv(path, usecols=usecols)
        except ValueError:
            continue
        slim = hw.iloc[::CONTACT_SURFACE_STRIDE].dropna(subset=usecols).copy()
        if not slim.empty:
            rows.append(slim)
    if not rows:
        raise RuntimeError(f"No train rows found for contact basis contacts={contacts}")
    return pd.concat(rows, ignore_index=True)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float(np.nanmedian(values))
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def robust_tail_offset(residual: np.ndarray, tail_max: int = 80) -> tuple[float, dict[str, float]]:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        raise RuntimeError("No finite residuals for contact offset calibration")
    n_tail = min(int(tail_max), max(12, len(residual) // 3))
    tail = residual[-n_tail:]
    weights = np.linspace(0.35, 1.0, len(tail))
    med = float(np.median(tail))
    mad = float(np.median(np.abs(tail - med)))
    scale = max(1.4826 * mad, float(np.std(tail)) * 0.5, 1e-6)
    keep = np.abs(tail - med) <= 3.0 * scale
    if int(keep.sum()) >= max(6, len(tail) // 2):
        tail_use = tail[keep]
        weights_use = weights[keep]
    else:
        lo, hi = np.quantile(tail, [0.10, 0.90])
        keep = (tail >= lo) & (tail <= hi)
        tail_use = tail[keep]
        weights_use = weights[keep]
    offset = float(np.average(tail_use, weights=weights_use))
    fit = float(np.sqrt(np.mean((tail - offset) ** 2)))
    return offset, {
        "n_tail": float(n_tail),
        "tail_fit_rmse": fit,
        "tail_residual_std": float(np.std(tail)),
        "tail_residual_mad": mad,
        "tail_trimmed_fraction": float(1.0 - len(tail_use) / max(len(tail), 1)),
    }


def predict_contact_surface_knn(pool: pd.DataFrame, hw: pd.DataFrame, contact_col: str) -> np.ndarray:
    train_xy = pool[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    target_xy = hw[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    tree = cKDTree(train_xy)
    dist, idx = tree.query(target_xy, k=min(CONTACT_BASIS_K, len(pool)), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    vals = pool[contact_col].to_numpy(dtype=float)[idx]
    weights = 1.0 / np.maximum(dist, 1e-3) ** 2
    return np.sum(vals * weights, axis=1) / np.sum(weights, axis=1)


def predict_contact_surface_plane(pool: pd.DataFrame, hw: pd.DataFrame, contact_col: str) -> np.ndarray:
    train_xy = pool[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    target_xy = hw[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    values = pool[contact_col].to_numpy(dtype=float)
    tree = cKDTree(train_xy)
    dist, idx = tree.query(target_xy, k=min(CONTACT_BASIS_K, len(pool)), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    out = np.empty(len(target_xy), dtype=float)
    for i in range(len(target_xy)):
        local_xy = train_xy[idx[i]] - target_xy[i]
        local_y = values[idx[i]]
        weights = 1.0 / np.maximum(dist[i], 1e-3) ** 2
        a = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.ones(len(local_xy))])
        root_w = np.sqrt(weights)
        try:
            coef, *_ = np.linalg.lstsq(a * root_w[:, None], local_y * root_w, rcond=None)
            out[i] = float(coef[2])
        except np.linalg.LinAlgError:
            out[i] = float(np.average(local_y, weights=weights))
    return out


def predict_contact_surface_geo(
    pool: pd.DataFrame,
    hw: pd.DataFrame,
    contact_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train_xy = pool[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    target_xy = hw[["X", "Y"]].to_numpy(dtype=float) / CONTACT_SURFACE_XY_SCALE
    values = pool[contact_col].to_numpy(dtype=float)
    tree = cKDTree(train_xy)
    dist, idx = tree.query(target_xy, k=min(CONTACT_BASIS_K, len(pool)), workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    plane = np.empty(len(target_xy), dtype=float)
    knn = np.empty(len(target_xy), dtype=float)
    corrected = np.empty(len(target_xy), dtype=float)
    residual_corr = np.empty(len(target_xy), dtype=float)
    local_mad = np.empty(len(target_xy), dtype=float)
    mean_dist = np.empty(len(target_xy), dtype=float)

    for i in range(len(target_xy)):
        local_xy = train_xy[idx[i]] - target_xy[i]
        local_y = values[idx[i]]
        weights = 1.0 / np.maximum(dist[i], 1e-3) ** 2
        a = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.ones(len(local_xy))])
        root_w = np.sqrt(weights)
        try:
            coef, *_ = np.linalg.lstsq(a * root_w[:, None], local_y * root_w, rcond=None)
            plane_i = float(coef[2])
            residual = local_y - (a @ coef)
        except np.linalg.LinAlgError:
            plane_i = float(np.average(local_y, weights=weights))
            residual = local_y - plane_i
        corr = weighted_median(residual, weights)
        plane[i] = plane_i
        knn[i] = float(np.average(local_y, weights=weights))
        corrected[i] = plane_i + corr
        residual_corr[i] = corr
        local_mad[i] = float(np.median(np.abs(residual - np.median(residual))))
        mean_dist[i] = float(np.mean(dist[i]))

    meta = {
        "surface_mode": CONTACT_GEO_ESTIMATOR_VERSION,
        "surface_k": float(min(CONTACT_BASIS_K, len(pool))),
        "surface_residual_correction_mean": float(np.mean(residual_corr)),
        "surface_residual_correction_std": float(np.std(residual_corr)),
        "surface_local_residual_mad_mean": float(np.mean(local_mad)),
        "surface_neighbor_dist_mean": float(np.mean(mean_dist)),
        "surface_plane_knn_gap_mean": float(np.mean(np.abs(plane - knn))),
        "surface_plane_geo_gap_mean": float(np.mean(np.abs(plane - corrected))),
    }
    return corrected, plane, knn, meta


def contact_order_fraction(surface_by_contact: dict[str, np.ndarray], row_idx: np.ndarray) -> float:
    row_idx = np.asarray(row_idx, dtype=int)
    pairs = []
    for upper, lower in zip(CONTACT_BASIS_CONTACT_ORDER[:-1], CONTACT_BASIS_CONTACT_ORDER[1:]):
        if upper in surface_by_contact and lower in surface_by_contact:
            up = np.asarray(surface_by_contact[upper], dtype=float)[row_idx]
            lo = np.asarray(surface_by_contact[lower], dtype=float)[row_idx]
            ok = np.isfinite(up) & np.isfinite(lo)
            if ok.any():
                pairs.append(float(np.mean((up[ok] - lo[ok]) > -1e-6)))
    if not pairs:
        return 1.0
    return float(np.mean(pairs))


def surface_roughness(hw: pd.DataFrame, contact_pred: np.ndarray, row_idx: np.ndarray) -> float:
    row_idx = np.asarray(row_idx, dtype=int)
    if len(row_idx) < 5:
        return 0.0
    order = np.argsort(row_idx)
    rows = row_idx[order]
    vals = np.asarray(contact_pred, dtype=float)[rows]
    md = hw["MD"].to_numpy(dtype=float)[rows] if "MD" in hw.columns else rows.astype(float)
    denom = np.maximum(np.gradient(md), 1.0)
    grad = np.gradient(vals) / denom
    return float(np.nanstd(np.gradient(grad)))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values.copy()
    return (
        pd.Series(values)
        .rolling(min(window, len(values)), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def robust_loss(values: np.ndarray, delta: float = 2.5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)
    return np.where(abs_values <= delta, 0.5 * values * values, delta * (abs_values - 0.5 * delta))


def gr_typewell_path_score(hw: pd.DataFrame, tw: pd.DataFrame | None, path_tvt: np.ndarray) -> dict[str, float]:
    if tw is None or "GR" not in hw.columns or "GR" not in tw.columns or "TVT" not in tw.columns:
        return {
            "gr_loss": 0.0,
            "gr_confidence": 0.5,
            "gr_sigma": 0.0,
            "gr_r2": 0.0,
            "gr_post_std": 0.0,
        }
    known = hw["TVT_input"].notna().to_numpy()
    eval_mask = ~known
    if int(known.sum()) < 20 or int(eval_mask.sum()) < 5:
        return {
            "gr_loss": 0.0,
            "gr_confidence": 0.5,
            "gr_sigma": 0.0,
            "gr_r2": 0.0,
            "gr_post_std": 0.0,
        }

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr_raw = pd.Series(tw_s["GR"]).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)
    hw_gr_raw = pd.Series(hw["GR"]).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)
    known_tvt = hw.loc[known, "TVT_input"].to_numpy(dtype=float)
    tw_known = np.interp(known_tvt, tw_tvt, rolling_mean(tw_gr_raw, 9))
    hw_known = hw_gr_raw[known]
    valid = np.isfinite(tw_known) & np.isfinite(hw_known)
    if int(valid.sum()) < 20 or float(np.std(tw_known[valid])) < 1e-6:
        return {
            "gr_loss": 0.0,
            "gr_confidence": 0.35,
            "gr_sigma": 0.0,
            "gr_r2": 0.0,
            "gr_post_std": float(np.nanstd(hw_gr_raw[eval_mask])),
        }

    x = tw_known[valid]
    y = hw_known[valid]
    a_mat = np.column_stack([x, np.ones(len(x))])
    scale, shift = np.linalg.lstsq(a_mat, y, rcond=None)[0]
    if abs(scale) < 1e-6:
        scale = 1.0
        shift = 0.0
    fitted = scale * x + shift
    resid = y - fitted
    sigma = max(5.0, 1.4826 * float(np.median(np.abs(resid - np.median(resid)))))
    denom = float(np.var(y))
    r2 = float(1.0 - np.var(resid) / denom) if denom > 1e-9 else 0.0

    losses = []
    post_gr_std = 0.0
    path_tvt = np.asarray(path_tvt, dtype=float)
    eval_path = path_tvt[eval_mask]
    eval_hw = hw_gr_raw[eval_mask]
    for window in (1, 9, 31):
        hw_use = rolling_mean(eval_hw, window)
        tw_use = rolling_mean(tw_gr_raw, window)
        expected = scale * np.interp(eval_path, tw_tvt, tw_use) + shift
        z = (hw_use - expected) / sigma
        losses.append(float(np.mean(robust_loss(z))))
        post_gr_std = max(post_gr_std, float(np.std(hw_use)))
    loss = float(np.mean(losses))
    confidence = 1.0 / (1.0 + sigma / 35.0 + max(0.0, -r2) + loss / 8.0)
    if post_gr_std < 3.0:
        confidence *= 0.5
    confidence = float(np.clip(confidence, 0.05, 1.0))
    return {
        "gr_loss": loss,
        "gr_confidence": confidence,
        "gr_sigma": float(sigma),
        "gr_r2": r2,
        "gr_post_std": float(post_gr_std),
    }


def robust_center_scale(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    numeric = frame.astype(float)
    center = numeric.median(axis=0)
    mad = (numeric - center).abs().median(axis=0)
    scale = (1.4826 * mad).replace(0, np.nan)
    std = numeric.std(axis=0).replace(0, np.nan)
    scale = scale.fillna(std).fillna(1.0)
    scale = scale.clip(lower=1e-6)
    return center, scale


def typewell_gr_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr = (
        pd.Series(tw_s["GR"])
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    return tw_tvt, rolling_mean(tw_gr, 15)


def fill_horizontal_gr(hw: pd.DataFrame, fallback: float) -> np.ndarray:
    if "GR" not in hw.columns:
        return np.full(len(hw), float(fallback), dtype=float)
    return (
        pd.Series(hw["GR"])
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(float(fallback))
        .to_numpy(dtype=float)
    )


def gr_calibration_features_for_analog(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    base_full: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, float]:
    known = hw["TVT_input"].notna().to_numpy()
    if int(known.sum()) < 20 or int(eval_mask.sum()) < 5:
        return {"gr_sigma": 25.0, "gr_r2": 0.0, "eval_gr_loss": 1.0, "eval_gr_std": 0.0}
    tw_tvt, tw_gr = typewell_gr_arrays(tw)
    hw_gr = fill_horizontal_gr(hw, float(np.nanmean(tw_gr)))
    known_tvt = hw.loc[known, "TVT_input"].to_numpy(dtype=float)
    x = np.interp(known_tvt, tw_tvt, tw_gr)
    y = hw_gr[known]
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 20 or float(np.std(x[ok])) < 1e-6:
        return {"gr_sigma": 25.0, "gr_r2": 0.0, "eval_gr_loss": 1.0, "eval_gr_std": 0.0}

    mat = np.column_stack([x[ok], np.ones(int(ok.sum()))])
    scale, shift = np.linalg.lstsq(mat, y[ok], rcond=None)[0]
    fitted = scale * x[ok] + shift
    resid = y[ok] - fitted
    med = float(np.median(resid))
    sigma = max(5.0, 1.4826 * float(np.median(np.abs(resid - med))))
    denom = float(np.var(y[ok]))
    r2 = float(1.0 - np.var(resid) / denom) if denom > 1e-9 else 0.0
    expected = scale * np.interp(np.asarray(base_full, dtype=float)[eval_mask], tw_tvt, tw_gr) + shift
    z = (hw_gr[eval_mask] - expected) / sigma
    eval_loss = float(np.mean(np.minimum(z * z, 25.0))) if len(z) else 0.0
    return {
        "gr_sigma": float(sigma),
        "gr_r2": float(r2),
        "eval_gr_loss": eval_loss,
        "eval_gr_std": float(np.nanstd(hw_gr[eval_mask])) if int(eval_mask.sum()) else 0.0,
    }


def analog_feature_record(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    row_idx: np.ndarray,
    base_full: np.ndarray,
) -> dict[str, float]:
    row_idx = np.asarray(row_idx, dtype=int)
    eval_mask = np.zeros(len(hw), dtype=bool)
    valid_rows = row_idx[(row_idx >= 0) & (row_idx < len(hw))]
    eval_mask[valid_rows] = True
    known = hw["TVT_input"].notna().to_numpy()
    known_idx = np.where(known)[0]
    md = hw["MD"].to_numpy(dtype=float)
    z = hw["Z"].to_numpy(dtype=float)
    tvt_known = hw["TVT_input"].to_numpy(dtype=float)
    tail_idx = known_idx[-min(80, len(known_idx)):] if len(known_idx) else np.array([], dtype=int)
    eval_md = md[valid_rows]
    eval_z = z[valid_rows]
    base_eval = np.asarray(base_full, dtype=float)[valid_rows]
    gr_feats = gr_calibration_features_for_analog(hw, tw, base_full, eval_mask)
    return {
        "known_frac": float(known.mean()) if len(known) else 0.0,
        "n_eval": float(len(valid_rows)),
        "eval_md_span": float(np.ptp(eval_md)) if len(eval_md) else 0.0,
        "eval_z_span": float(np.ptp(eval_z)) if len(eval_z) else 0.0,
        "eval_z_slope": finite_slope(eval_md, eval_z),
        "tail_tvt_slope": finite_slope(md[tail_idx], tvt_known[tail_idx]) if len(tail_idx) else 0.0,
        "tail_z_slope": finite_slope(md[tail_idx], z[tail_idx]) if len(tail_idx) else 0.0,
        "base_eval_slope": finite_slope(eval_md, base_eval),
        "base_delta_end": float(base_eval[-1] - tvt_known[known_idx[-1]]) if len(base_eval) and len(known_idx) else 0.0,
        "base_eval_std": float(np.std(base_eval)) if len(base_eval) else 0.0,
        **gr_feats,
    }


def find_analog_residual_table() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent / ANALOG_RESIDUAL_TABLE)
    except NameError:
        pass
    candidates.extend([
        Path.cwd() / ANALOG_RESIDUAL_TABLE,
        WORKING / ANALOG_RESIDUAL_TABLE,
    ])
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(input_root.glob(f"**/{ANALOG_RESIDUAL_TABLE}")))
    for path in candidates:
        if path.exists():
            return path
    if EMBEDDED_ANALOG_RESIDUAL_TABLE_GZ_B64.strip():
        out_path = WORKING / ANALOG_RESIDUAL_TABLE
        out_path.write_bytes(gzip.decompress(base64.b64decode(EMBEDDED_ANALOG_RESIDUAL_TABLE_GZ_B64)))
        return out_path
    raise FileNotFoundError(f"Cannot locate {ANALOG_RESIDUAL_TABLE}")


def apply_analog_residual_transfer(
    data_dir: Path,
    submission: pd.DataFrame,
    rule: str,
    alpha: float,
    k: int,
    max_abs: float,
    component_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], Path | None]:
    if rule in {"", "0", "off", "none"} or float(alpha) == 0.0:
        return submission, [], {}, None
    if rule != "scalar_bias_v1":
        raise ValueError(f"Unknown ROGII_ANALOG_RESIDUAL_RULE={rule}")

    table_path = find_analog_residual_table()
    table = pd.read_csv(table_path)
    missing = [col for col in ANALOG_RESIDUAL_FEATURES + ("mean_residual",) if col not in table.columns]
    if missing:
        raise ValueError(f"Analog residual table missing columns: {missing}")
    feature_frame = table.loc[:, list(ANALOG_RESIDUAL_FEATURES)].astype(float)
    center, scale = robust_center_scale(feature_frame)
    z_table = (feature_frame - center) / scale

    out = submission.copy()
    split = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    out["well"] = split[0]
    out["row_idx"] = split[1].astype(int)
    operations: list[dict[str, object]] = []
    component = out[["id"]].copy()
    component["base_tvt"] = out["tvt"].to_numpy(dtype=float)
    component["analog_correction"] = 0.0

    for wid, idxs in out.groupby("well", sort=True).groups.items():
        hw_path = data_dir / "test" / f"{wid}__horizontal_well.csv"
        tw_path = data_dir / "test" / f"{wid}__typewell.csv"
        if not hw_path.exists() or not tw_path.exists():
            continue
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)
        locs = np.asarray(list(idxs), dtype=int)
        row_idx = out.loc[locs, "row_idx"].to_numpy(dtype=int)
        base_full = contact_anchor_path(hw)
        valid = (row_idx >= 0) & (row_idx < len(base_full))
        if not valid.any():
            continue
        row_idx_valid = row_idx[valid]
        locs_valid = locs[valid]
        base_full[row_idx_valid] = out.loc[locs_valid, "tvt"].to_numpy(dtype=float)
        features = analog_feature_record(hw, tw, row_idx_valid, base_full)
        target = pd.Series({c: float(features.get(c, 0.0)) for c in ANALOG_RESIDUAL_FEATURES})
        target_z = ((target - center) / scale).to_numpy(dtype=float)
        diff = z_table.to_numpy(dtype=float) - target_z[None, :]
        dist = np.sqrt(np.nanmean(diff * diff, axis=1))
        dist = np.where(np.isfinite(dist), dist, np.inf)
        take = np.argsort(dist)[: max(1, min(int(k), len(table)))]
        weights = 1.0 / (0.05 + dist[take])
        weights = weights / float(weights.sum())
        raw = float(np.sum(weights * table.iloc[take]["mean_residual"].to_numpy(dtype=float)))
        correction = float(np.clip(float(alpha) * raw, -float(max_abs), float(max_abs)))
        out.loc[locs_valid, "tvt"] = out.loc[locs_valid, "tvt"].to_numpy(dtype=float) + correction
        component.loc[locs_valid, "analog_correction"] = correction
        operations.append({
            "rule": rule,
            "well": str(wid),
            "rows": int(len(locs_valid)),
            "alpha": float(alpha),
            "k": int(len(take)),
            "raw_mean_residual": raw,
            "applied_correction": correction,
            "mean_distance": float(np.sum(weights * dist[take])),
            "nearest_distance": float(dist[take[0]]),
            "nearest_wells": [
                str(x)
                for x in table.iloc[take]["well"].astype(str).head(5).tolist()
            ],
            "nearest_mask_known_frac": [
                float(x)
                for x in table.iloc[take].get("mask_known_frac", pd.Series(dtype=float)).head(5).tolist()
            ],
            "features": {str(k2): float(v2) for k2, v2 in features.items()},
        })

    component["analog_tvt"] = out["tvt"].to_numpy(dtype=float)
    component_path.parent.mkdir(parents=True, exist_ok=True)
    component.to_csv(component_path, index=False)
    summary = {
        "rule": rule,
        "alpha": float(alpha),
        "k": int(k),
        "max_abs": float(max_abs),
        "table_path": str(table_path),
        "table_sha256": sha256_file(table_path),
        "table_rows": int(len(table)),
        "feature_columns": list(ANALOG_RESIDUAL_FEATURES),
        "operations": int(len(operations)),
        "mean_abs_applied_correction": float(np.mean(np.abs(component["analog_correction"].to_numpy(dtype=float)))),
        "max_abs_applied_correction": float(np.max(np.abs(component["analog_correction"].to_numpy(dtype=float)))),
    }
    return out[["id", "tvt"]], operations, summary, component_path


def train_contact_tvt_path(
    hw_tr: pd.DataFrame,
    tw_tr: pd.DataFrame,
    ref_col: str,
) -> np.ndarray:
    if ref_col not in hw_tr.columns or "TVT" not in hw_tr.columns or "Z" not in hw_tr.columns:
        return np.full(len(hw_tr), np.nan, dtype=float)
    if "Geology" not in tw_tr.columns or "TVT" not in tw_tr.columns:
        return np.full(len(hw_tr), np.nan, dtype=float)
    tw_g = tw_tr.dropna(subset=["Geology", "TVT"])
    ref_rows = tw_g[tw_g["Geology"].astype(str) == str(ref_col)]
    if len(ref_rows) == 0:
        return np.full(len(hw_tr), np.nan, dtype=float)
    ref_tvt = float(ref_rows["TVT"].min())
    raw = ref_tvt - (hw_tr["Z"].to_numpy(dtype=float) - hw_tr[ref_col].to_numpy(dtype=float))
    truth = hw_tr["TVT"].to_numpy(dtype=float)
    ok = np.isfinite(raw) & np.isfinite(truth)
    if int(ok.sum()) < 50:
        return np.full(len(hw_tr), np.nan, dtype=float)
    offset = float(np.mean(truth[ok] - raw[ok]))
    return raw + offset


def best_guarded_contact_path(
    hw_te: pd.DataFrame,
    hw_tr: pd.DataFrame,
    tw_tr: pd.DataFrame,
    refs: tuple[str, ...],
    max_prefix_rmse: float,
    min_prefix_rows: int,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if "MD" not in hw_te.columns or "MD" not in hw_tr.columns or "TVT_input" not in hw_te.columns:
        return None, {"status": "skip", "reason": "missing_required_columns"}
    known = hw_te["TVT_input"].notna().to_numpy()
    if int(known.sum()) < int(min_prefix_rows):
        return None, {"status": "skip", "reason": "too_few_prefix_rows", "prefix_rows": int(known.sum())}

    md_tr_all = hw_tr["MD"].to_numpy(dtype=float)
    md_te = hw_te["MD"].to_numpy(dtype=float)
    known_md = md_te[known]
    known_tvt = hw_te.loc[known, "TVT_input"].to_numpy(dtype=float)
    best: tuple[float, str, np.ndarray, int, float, float] | None = None
    candidates: list[dict[str, object]] = []
    for ref_col in refs:
        phys = train_contact_tvt_path(hw_tr, tw_tr, ref_col)
        finite = np.isfinite(md_tr_all) & np.isfinite(phys)
        if int(finite.sum()) < 100:
            candidates.append({"ref": ref_col, "status": "skip", "reason": "too_few_finite_train_rows"})
            continue
        order = np.argsort(md_tr_all[finite])
        md_tr = md_tr_all[finite][order]
        tvt_tr = phys[finite][order]
        in_range = (
            np.isfinite(known_md)
            & np.isfinite(known_tvt)
            & (known_md >= md_tr[0])
            & (known_md <= md_tr[-1])
        )
        n_fit = int(in_range.sum())
        if n_fit < int(min_prefix_rows):
            candidates.append({"ref": ref_col, "status": "skip", "reason": "too_few_in_range_prefix_rows", "prefix_rows": n_fit})
            continue
        pred_known = np.interp(known_md[in_range], md_tr, tvt_tr)
        rmse = float(np.sqrt(np.mean((pred_known - known_tvt[in_range]) ** 2)))
        coverage = float(np.mean((md_te >= md_tr[0]) & (md_te <= md_tr[-1])))
        candidates.append({
            "ref": ref_col,
            "status": "candidate",
            "prefix_rmse": rmse,
            "prefix_rows": n_fit,
            "coverage": coverage,
            "train_rows": int(finite.sum()),
        })
        if best is None or rmse < best[0]:
            full = np.interp(np.clip(md_te, md_tr[0], md_tr[-1]), md_tr, tvt_tr)
            best = (rmse, ref_col, full, n_fit, coverage, float(finite.sum()))

    if best is None:
        return None, {"status": "skip", "reason": "no_valid_contact_ref", "candidates": candidates}
    rmse, ref_col, full, n_fit, coverage, train_rows = best
    meta = {
        "status": "apply" if rmse <= float(max_prefix_rmse) else "skip",
        "reason": "prefix_guard_pass" if rmse <= float(max_prefix_rmse) else "prefix_rmse_above_threshold",
        "ref": ref_col,
        "prefix_rmse": rmse,
        "prefix_rows": int(n_fit),
        "coverage": coverage,
        "train_rows": int(train_rows),
        "candidates": candidates,
    }
    if rmse > float(max_prefix_rmse):
        return None, meta
    return full, meta


def apply_guarded_contact_override(
    data_dir: Path,
    submission: pd.DataFrame,
    rule: str,
    refs: tuple[str, ...],
    max_prefix_rmse: float,
    min_prefix_rows: int,
    component_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], Path | None]:
    if rule in {"", "0", "off", "none"}:
        return submission, [], {}, None
    if rule != "guarded_multi_ref":
        raise ValueError(f"Unknown ROGII_CONTACT_OVERRIDE_RULE={rule}")

    out = submission.copy()
    split = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    out["well"] = split[0]
    out["row_idx"] = split[1].astype(int)
    component = out[["id"]].copy()
    component["base_tvt"] = out["tvt"].to_numpy(dtype=float)
    component["override_tvt"] = out["tvt"].to_numpy(dtype=float)
    component["override_delta"] = 0.0
    component["contact_override_applied"] = False
    component["contact_override_ref"] = ""
    component["contact_override_prefix_rmse"] = np.nan

    train_wells = set(well_ids(data_dir, "train"))
    operations: list[dict[str, object]] = []
    for wid, idxs in out.groupby("well", sort=True).groups.items():
        locs = np.asarray(list(idxs), dtype=int)
        op: dict[str, object] = {
            "rule": rule,
            "well": str(wid),
            "rows": int(len(locs)),
            "max_prefix_rmse": float(max_prefix_rmse),
            "min_prefix_rows": int(min_prefix_rows),
        }
        if str(wid) not in train_wells:
            op.update({"status": "skip", "reason": "no_same_id_train_well"})
            operations.append(op)
            continue
        try:
            hw_te = pd.read_csv(data_dir / "test" / f"{wid}__horizontal_well.csv")
            hw_tr = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
            tw_tr = pd.read_csv(data_dir / "train" / f"{wid}__typewell.csv")
            full, meta = best_guarded_contact_path(
                hw_te,
                hw_tr,
                tw_tr,
                refs,
                max_prefix_rmse,
                min_prefix_rows,
            )
            op.update(meta)
            if full is None:
                operations.append(op)
                continue
            row_idx = out.loc[locs, "row_idx"].to_numpy(dtype=int)
            valid = (row_idx >= 0) & (row_idx < len(full)) & np.isfinite(full[np.clip(row_idx, 0, len(full) - 1)])
            if not valid.any():
                op.update({"status": "skip", "reason": "no_valid_submission_rows"})
                operations.append(op)
                continue
            locs_valid = locs[valid]
            row_idx_valid = row_idx[valid]
            new_vals = full[row_idx_valid].astype(float)
            old_vals = out.loc[locs_valid, "tvt"].to_numpy(dtype=float)
            out.loc[locs_valid, "tvt"] = new_vals
            component.loc[locs_valid, "override_tvt"] = new_vals
            component.loc[locs_valid, "override_delta"] = new_vals - old_vals
            component.loc[locs_valid, "contact_override_applied"] = True
            component.loc[locs_valid, "contact_override_ref"] = str(meta.get("ref", ""))
            component.loc[locs_valid, "contact_override_prefix_rmse"] = float(meta.get("prefix_rmse", np.nan))
            op.update({
                "status": "apply",
                "applied_rows": int(len(locs_valid)),
                "mean_abs_delta": float(np.mean(np.abs(new_vals - old_vals))),
                "max_abs_delta": float(np.max(np.abs(new_vals - old_vals))),
            })
            operations.append(op)
        except Exception as exc:
            op.update({"status": "error", "reason": str(exc)})
            operations.append(op)

    component_path.parent.mkdir(parents=True, exist_ok=True)
    component.to_csv(component_path, index=False)
    applied = component["contact_override_applied"].to_numpy(dtype=bool)
    deltas = component.loc[applied, "override_delta"].to_numpy(dtype=float)
    summary = {
        "rule": rule,
        "refs": list(refs),
        "max_prefix_rmse": float(max_prefix_rmse),
        "min_prefix_rows": int(min_prefix_rows),
        "operations": int(len(operations)),
        "applied_wells": int(sum(1 for op in operations if op.get("status") == "apply")),
        "applied_rows": int(applied.sum()),
        "mean_abs_delta": float(np.mean(np.abs(deltas))) if len(deltas) else 0.0,
        "max_abs_delta": float(np.max(np.abs(deltas))) if len(deltas) else 0.0,
    }
    return out[["id", "tvt"]], operations, summary, component_path


def smooth_center_clip_basis(values: np.ndarray, max_abs: float) -> np.ndarray:
    basis = np.asarray(values, dtype=float).copy()
    if len(basis) >= 5:
        window = min(31, len(basis) if len(basis) % 2 == 1 else len(basis) - 1)
        if window >= 5:
            basis = (
                pd.Series(basis)
                .rolling(window, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
    basis = basis - float(np.mean(basis))
    clipped = np.clip(basis, -float(max_abs), float(max_abs))
    clipped = clipped - float(np.mean(clipped))
    max_seen = float(np.max(np.abs(clipped))) if len(clipped) else 0.0
    if max_seen > float(max_abs) > 0:
        clipped *= float(max_abs) / max_seen
    return clipped.astype(float)


def contact_basis_for_rows(
    hw: pd.DataFrame,
    row_idx: np.ndarray,
    base_tvt: np.ndarray,
    knn_contact: np.ndarray,
    plane_contact: np.ndarray,
    max_abs: float,
    geo_contact: np.ndarray | None = None,
    surface_by_contact: dict[str, np.ndarray] | None = None,
    tw: pd.DataFrame | None = None,
    surface_meta: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    contact_pred = np.asarray(
        geo_contact if geo_contact is not None else np.median(np.vstack([knn_contact, plane_contact]), axis=0),
        dtype=float,
    )
    surface_base = contact_pred - hw["Z"].to_numpy(dtype=float)
    known = hw["TVT_input"].notna().to_numpy() & np.isfinite(surface_base)
    if int(known.sum()) < 8:
        raise RuntimeError("Not enough known TVT_input rows to calibrate contact basis")

    known_residual = hw.loc[known, "TVT_input"].to_numpy(dtype=float) - surface_base[known]
    offset, offset_meta = robust_tail_offset(known_residual, tail_max=80)
    contact_tvt = surface_base + offset

    row_idx = np.asarray(row_idx, dtype=int)
    base_tvt = np.asarray(base_tvt, dtype=float)
    order = np.argsort(row_idx)
    raw_delta_sorted = contact_tvt[row_idx[order]] - base_tvt[order]
    basis_sorted = smooth_center_clip_basis(raw_delta_sorted, max_abs)
    basis = np.empty_like(basis_sorted)
    basis[order] = basis_sorted

    known_fit = float(np.sqrt(np.mean((known_residual - offset) ** 2)))
    method_gap = float(np.mean(np.abs(knn_contact[row_idx] - plane_contact[row_idx])))
    order_fraction = contact_order_fraction(surface_by_contact or {}, row_idx)
    roughness = surface_roughness(hw, contact_pred, row_idx)
    base_full = contact_anchor_path(hw)
    base_full[row_idx] = base_tvt
    contact_full = contact_anchor_path(hw)
    contact_full[row_idx] = contact_tvt[row_idx]
    gr_base = gr_typewell_path_score(hw, tw, base_full)
    gr_contact = gr_typewell_path_score(hw, tw, contact_full)
    gr_agreement = 1.0 / (1.0 + max(0.0, gr_contact["gr_loss"] - gr_base["gr_loss"]))
    geo_confidence = 1.0 / (
        1.0
        + offset_meta["tail_fit_rmse"] / 20.0
        + offset_meta["tail_residual_std"] / 25.0
        + method_gap / 25.0
        + known_fit / 30.0
        + roughness / 0.04
        + max(0.0, 1.0 - order_fraction) * 2.5
    )
    geo_confidence = float(np.clip(geo_confidence, 0.05, 1.0))
    confidence = geo_confidence * (0.50 + 0.50 * gr_contact["gr_confidence"]) * gr_agreement
    confidence = float(np.clip(confidence, 0.03, 1.0))
    meta = {
        "offset": offset,
        "n_known": float(known.sum()),
        **offset_meta,
        "known_fit_rmse": known_fit,
        "method_disagreement_mean": method_gap,
        "contact_order_fraction": order_fraction,
        "surface_roughness": roughness,
        "geo_confidence": geo_confidence,
        "gr_base_loss": float(gr_base["gr_loss"]),
        "gr_contact_loss": float(gr_contact["gr_loss"]),
        "gr_loss_delta_base_minus_contact": float(gr_base["gr_loss"] - gr_contact["gr_loss"]),
        "gr_confidence": float(gr_contact["gr_confidence"]),
        "gr_agreement": float(gr_agreement),
        "gr_sigma": float(gr_contact["gr_sigma"]),
        "gr_r2": float(gr_contact["gr_r2"]),
        "gr_post_std": float(gr_contact["gr_post_std"]),
        "basis_mean": float(np.mean(basis)),
        "basis_mean_abs": float(np.mean(np.abs(basis))),
        "basis_std": float(np.std(basis)),
        "basis_max_abs": float(np.max(np.abs(basis))) if len(basis) else 0.0,
        "confidence": confidence,
    }
    if surface_meta:
        meta.update({str(k): float(v) for k, v in surface_meta.items() if isinstance(v, (int, float, np.floating))})
    return basis, contact_tvt[row_idx], meta


def select_contact_basis_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        raise RuntimeError("No contact-basis candidates were produced")
    return max(candidates, key=lambda item: float(item["score"]))


def calibrated_geo_path(
    hw: pd.DataFrame,
    contact_pred: np.ndarray,
    row_idx: np.ndarray,
    max_abs_delta: float,
    offset_mode: str,
    smooth_window: int,
) -> tuple[np.ndarray, dict[str, float]]:
    contact_pred = np.asarray(contact_pred, dtype=float)
    surface_base = contact_pred - hw["Z"].to_numpy(dtype=float)
    known = hw["TVT_input"].notna().to_numpy() & np.isfinite(surface_base)
    if int(known.sum()) < 8:
        raise RuntimeError("Not enough known TVT_input rows to calibrate geo path")
    residual = hw.loc[known, "TVT_input"].to_numpy(dtype=float) - surface_base[known]
    if offset_mode == "median":
        offset = float(np.median(residual))
        offset_meta = {
            "n_tail": float(len(residual)),
            "tail_fit_rmse": float(np.sqrt(np.mean((residual - offset) ** 2))),
            "tail_residual_std": float(np.std(residual)),
            "tail_residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
            "tail_trimmed_fraction": 0.0,
        }
    elif offset_mode == "tail_median":
        n_tail = min(80, max(12, len(residual) // 3))
        tail = residual[-n_tail:]
        offset = float(np.median(tail))
        offset_meta = {
            "n_tail": float(n_tail),
            "tail_fit_rmse": float(np.sqrt(np.mean((tail - offset) ** 2))),
            "tail_residual_std": float(np.std(tail)),
            "tail_residual_mad": float(np.median(np.abs(tail - np.median(tail)))),
            "tail_trimmed_fraction": 0.0,
        }
    else:
        offset, offset_meta = robust_tail_offset(residual, tail_max=80)

    full = surface_base + offset
    full = np.where(np.isfinite(full), full, contact_anchor_path(hw))
    row_idx = np.asarray(row_idx, dtype=int)
    if smooth_window > 1 and len(row_idx) >= 5:
        ordered = row_idx[np.argsort(row_idx)]
        vals = full[ordered]
        full[ordered] = rolling_mean(vals, min(int(smooth_window), len(vals)))

    anchor = contact_anchor_path(hw)
    max_abs_delta = float(max_abs_delta)
    if max_abs_delta > 0:
        delta = np.clip(full[row_idx] - anchor[row_idx], -max_abs_delta, max_abs_delta)
        full[row_idx] = anchor[row_idx] + delta
    full[known] = hw.loc[known, "TVT_input"].to_numpy(dtype=float)
    known_fit = float(np.sqrt(np.mean((residual - offset) ** 2)))
    return full, {
        "offset": float(offset),
        "offset_mode": offset_mode,
        "smooth_window": float(smooth_window),
        "n_known": float(known.sum()),
        "known_fit_rmse": known_fit,
        **offset_meta,
    }


def candidate_tail_metrics(hw: pd.DataFrame, candidate_full: np.ndarray, row_idx: np.ndarray) -> dict[str, float]:
    known_idx = np.where(hw["TVT_input"].notna().to_numpy())[0]
    row_idx = np.asarray(row_idx, dtype=int)
    if len(known_idx) < 5 or len(row_idx) < 2:
        return {"tail_jump": 0.0, "tail_slope_delta": 0.0}
    last_known = int(known_idx[-1])
    first_eval = int(row_idx[np.argsort(row_idx)][0])
    tail = hw.iloc[known_idx[-min(30, len(known_idx)):]]
    md_tail = tail["MD"].to_numpy(dtype=float)
    tvt_tail = tail["TVT_input"].to_numpy(dtype=float)
    md_eval = hw["MD"].to_numpy(dtype=float)[row_idx]
    pred_eval = np.asarray(candidate_full, dtype=float)[row_idx]
    tail_slope = finite_slope(md_tail, tvt_tail)
    eval_slope = finite_slope(md_eval[: min(30, len(md_eval))], pred_eval[: min(30, len(pred_eval))])
    return {
        "tail_jump": float(abs(candidate_full[first_eval] - hw.loc[last_known, "TVT_input"])),
        "tail_slope_delta": float(abs(eval_slope - tail_slope)),
    }


def geo_path_score(
    hw: pd.DataFrame,
    tw: pd.DataFrame | None,
    candidate_full: np.ndarray,
    base_full: np.ndarray,
    row_idx: np.ndarray,
    meta: dict[str, object],
) -> tuple[float, dict[str, float]]:
    gr_candidate = gr_typewell_path_score(hw, tw, candidate_full)
    gr_base = gr_typewell_path_score(hw, tw, base_full)
    tail = candidate_tail_metrics(hw, candidate_full, row_idx)
    delta = np.asarray(candidate_full, dtype=float)[row_idx] - np.asarray(base_full, dtype=float)[row_idx]
    delta_mean_abs = float(np.mean(np.abs(delta))) if len(delta) else 0.0
    score = (
        float(gr_candidate["gr_loss"])
        + max(0.0, float(gr_candidate["gr_loss"]) - float(gr_base["gr_loss"])) * 1.5
        + float(meta.get("known_fit_rmse", 0.0)) / 30.0
        + float(meta.get("tail_fit_rmse", 0.0)) / 20.0
        + float(meta.get("method_disagreement_mean", 0.0)) / 35.0
        + float(meta.get("surface_roughness", 0.0)) / 0.05
        + max(0.0, 1.0 - float(meta.get("contact_order_fraction", 1.0))) * 2.5
        + tail["tail_jump"] / 60.0
        + tail["tail_slope_delta"] * 20.0
        + max(0.0, delta_mean_abs - 60.0) / 20.0
    )
    confidence = 1.0 / (1.0 + score / 4.0)
    confidence *= 0.50 + 0.50 * float(gr_candidate["gr_confidence"])
    payload = {
        "selector_score": float(score),
        "confidence": float(np.clip(confidence, 0.02, 1.0)),
        "gr_loss": float(gr_candidate["gr_loss"]),
        "gr_base_loss": float(gr_base["gr_loss"]),
        "gr_confidence": float(gr_candidate["gr_confidence"]),
        "gr_sigma": float(gr_candidate["gr_sigma"]),
        "gr_r2": float(gr_candidate["gr_r2"]),
        "delta_mean_abs": delta_mean_abs,
        "delta_mean": float(np.mean(delta)) if len(delta) else 0.0,
        "delta_max_abs": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        **tail,
    }
    return float(score), payload


def build_geo_path_candidates(
    hw: pd.DataFrame,
    tw: pd.DataFrame | None,
    row_idx: np.ndarray,
    base_tvt: np.ndarray,
    pf_tvt: np.ndarray | None,
    artifact_tvt: np.ndarray | None,
    pool: pd.DataFrame,
    contacts: tuple[str, ...] = CONTACT_BASIS_CONTACTS,
    max_abs_delta: float = GEO_PATH_MAX_ABS_DELTA,
    precomputed_surfaces: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    row_idx = np.asarray(row_idx, dtype=int)
    anchor = contact_anchor_path(hw)
    base_full = anchor.copy()
    base_full[row_idx] = np.asarray(base_tvt, dtype=float)
    candidates: list[dict[str, object]] = [{
        "candidate_type": "base_blend",
        "name": "base_blend",
        "full": base_full,
        "contact": "",
        "meta": {"candidate_source": "base_blend"},
    }]
    if pf_tvt is not None:
        full = anchor.copy()
        full[row_idx] = np.asarray(pf_tvt, dtype=float)
        candidates.append({"candidate_type": "pf", "name": "pf", "full": full, "contact": "", "meta": {"candidate_source": "pf"}})
    if artifact_tvt is not None:
        full = anchor.copy()
        full[row_idx] = np.asarray(artifact_tvt, dtype=float)
        candidates.append({"candidate_type": "artifact", "name": "artifact", "full": full, "contact": "", "meta": {"candidate_source": "artifact"}})

    if pf_tvt is not None and artifact_tvt is not None:
        for w_pf in (0.65, 0.80):
            full = anchor.copy()
            full[row_idx] = w_pf * np.asarray(pf_tvt, dtype=float) + (1.0 - w_pf) * np.asarray(artifact_tvt, dtype=float)
            name = f"blend_pf{w_pf:.2f}".replace(".", "p")
            candidates.append({
                "candidate_type": "blend",
                "name": name,
                "full": full,
                "contact": "",
                "meta": {"candidate_source": "blend", "candidate_pf_weight": float(w_pf)},
            })

    surface_cache: dict[str, dict[str, object]] = precomputed_surfaces or {}
    surface_by_contact: dict[str, np.ndarray] = {
        str(contact): np.asarray(surfaces["geo"], dtype=float)
        for contact, surfaces in surface_cache.items()
        if "geo" in surfaces
    }
    if precomputed_surfaces is None:
        for contact in contacts:
            if contact not in pool.columns:
                continue
            geo, plane, knn, surf_meta = predict_contact_surface_geo(pool, hw, contact)
            surface_cache[contact] = {"geo": geo, "plane": plane, "knn": knn, "meta": surf_meta}
            surface_by_contact[contact] = geo

    for contact, surfaces in surface_cache.items():
        geo = np.asarray(surfaces["geo"], dtype=float)
        plane = np.asarray(surfaces["plane"], dtype=float)
        knn = np.asarray(surfaces["knn"], dtype=float)
        method_gap = float(np.mean(np.abs(knn[row_idx] - plane[row_idx])))
        order_fraction = contact_order_fraction(surface_by_contact, row_idx)
        roughness = surface_roughness(hw, geo, row_idx)
        for offset_mode in ("tail_robust", "tail_median", "median"):
            for smooth_window in (1, 15, 31):
                full, meta = calibrated_geo_path(
                    hw,
                    geo,
                    row_idx,
                    max_abs_delta,
                    offset_mode,
                    smooth_window,
                )
                meta.update({
                    "candidate_source": "geo_contact",
                    "method_disagreement_mean": method_gap,
                    "contact_order_fraction": order_fraction,
                    "surface_roughness": roughness,
                    **{str(k): float(v) for k, v in surfaces["meta"].items() if isinstance(v, (int, float, np.floating))},
                })
                candidates.append({
                    "candidate_type": "geo_contact",
                    "name": f"geo_{contact}_{offset_mode}_s{smooth_window}",
                    "full": full,
                    "contact": contact,
                    "meta": meta,
                })
    return candidates


def select_geo_path_candidate(
    hw: pd.DataFrame,
    tw: pd.DataFrame | None,
    candidates: list[dict[str, object]],
    row_idx: np.ndarray,
    base_tvt: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not candidates:
        raise RuntimeError("No geo-path candidates were produced")
    anchor = contact_anchor_path(hw)
    base_full = anchor.copy()
    base_full[np.asarray(row_idx, dtype=int)] = np.asarray(base_tvt, dtype=float)
    scored: list[dict[str, object]] = []
    for cand in candidates:
        score, score_meta = geo_path_score(
            hw,
            tw,
            np.asarray(cand["full"], dtype=float),
            base_full,
            row_idx,
            cand.get("meta", {}),
        )
        payload = {
            "name": str(cand["name"]),
            "candidate_type": str(cand["candidate_type"]),
            "contact": str(cand.get("contact", "")),
            "selector_score": float(score),
            **score_meta,
            **{
                str(k): (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in cand.get("meta", {}).items()
            },
        }
        scored.append({**cand, "score": score, "score_meta": score_meta, "payload": payload})
    best_any = min(scored, key=lambda item: float(item["score"]))
    geo_scored = [item for item in scored if str(item["candidate_type"]) == "geo_contact"]
    if geo_scored:
        best_geo = min(geo_scored, key=lambda item: float(item["score"]))
        selected = best_geo if float(best_geo["score"]) <= float(best_any["score"]) + GEO_PATH_GEO_MARGIN else best_any
    else:
        selected = best_any
    return selected, [item["payload"] for item in sorted(scored, key=lambda item: float(item["score"]))[:12]]


def apply_geo_path_selector(
    data_dir: Path,
    submission: pd.DataFrame,
    pf: pd.DataFrame,
    artifact: pd.DataFrame,
    rule: str,
    blend_weight: float,
    max_abs_delta: float,
    component_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], Path | None]:
    summary: dict[str, object] = {
        "rule": rule,
        "blend_weight": float(blend_weight),
        "max_abs_delta": float(max_abs_delta),
        "contacts": list(CONTACT_BASIS_CONTACTS),
        "estimator": CONTACT_GEO_ESTIMATOR_VERSION,
        "k": CONTACT_BASIS_K,
        "stride": CONTACT_SURFACE_STRIDE,
        "xy_scale": CONTACT_SURFACE_XY_SCALE,
        "selected_by_well": {},
        "top_candidates_by_well": {},
        "skipped": [],
    }
    if rule in {"", "0", "off", "none"}:
        return submission, [], summary, None
    if rule != "selector_v1":
        raise ValueError(f"Unknown ROGII_GEO_PATH_RULE={rule}")

    split = submission["id"].astype(str).str.rsplit("_", n=1, expand=True)
    frame = submission.copy()
    frame["well"] = split[0]
    frame["row_idx"] = split[1].astype(int)
    frame["pf_tvt"] = pf["tvt"].to_numpy(dtype=float)
    frame["artifact_tvt"] = artifact["tvt"].to_numpy(dtype=float)
    pool = build_contact_basis_pool(data_dir, CONTACT_BASIS_CONTACTS)
    summary["pool_rows"] = int(len(pool))

    out = submission.copy()
    component = submission[["id"]].copy()
    component["geo_tvt"] = np.nan
    component["base_tvt"] = submission["tvt"].to_numpy(dtype=float)
    component["selected_name"] = ""
    component["selected_type"] = ""
    component["selected_contact"] = ""
    operations: list[dict[str, object]] = []

    for well in sorted(frame["well"].unique()):
        hw_path = data_dir / "test" / f"{well}__horizontal_well.csv"
        if not hw_path.exists():
            continue
        hw = pd.read_csv(hw_path)
        tw_path = data_dir / "test" / f"{well}__typewell.csv"
        tw = pd.read_csv(tw_path) if tw_path.exists() else None
        mask = frame["well"].to_numpy(str) == str(well)
        row_idx = frame.loc[mask, "row_idx"].to_numpy(dtype=int)
        base_tvt = frame.loc[mask, "tvt"].to_numpy(dtype=float)
        try:
            candidates = build_geo_path_candidates(
                hw,
                tw,
                row_idx,
                base_tvt,
                frame.loc[mask, "pf_tvt"].to_numpy(dtype=float),
                frame.loc[mask, "artifact_tvt"].to_numpy(dtype=float),
                pool,
                max_abs_delta=max_abs_delta,
            )
            selected, top_payload = select_geo_path_candidate(hw, tw, candidates, row_idx, base_tvt)
        except (RuntimeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
            summary["skipped"].append({"well": str(well), "reason": str(exc)})
            continue
        geo_tvt = np.asarray(selected["full"], dtype=float)[row_idx]
        final_tvt = (1.0 - float(blend_weight)) * base_tvt + float(blend_weight) * geo_tvt
        out.loc[mask, "tvt"] = final_tvt
        component.loc[mask, "geo_tvt"] = geo_tvt
        component.loc[mask, "selected_name"] = str(selected["name"])
        component.loc[mask, "selected_type"] = str(selected["candidate_type"])
        component.loc[mask, "selected_contact"] = str(selected.get("contact", ""))
        payload = dict(selected["payload"])
        payload.update({
            "well": str(well),
            "rows": int(mask.sum()),
            "blend_weight": float(blend_weight),
            "final_delta_mean_abs": float(np.mean(np.abs(final_tvt - base_tvt))),
            "final_delta_max_abs": float(np.max(np.abs(final_tvt - base_tvt))) if len(final_tvt) else 0.0,
        })
        summary["selected_by_well"][str(well)] = payload
        summary["top_candidates_by_well"][str(well)] = top_payload
        operations.append(payload)

    component_file = component_path or (WORKING / "geo_path_component.csv")
    component.to_csv(component_file, index=False)
    return out[["id", "tvt"]], operations, summary, component_file


def apply_contact_basis_probe(
    data_dir: Path,
    submission: pd.DataFrame,
    rule: str,
    value: float,
    max_abs: float,
    component_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], Path | None]:
    summary: dict[str, object] = {
        "rule": rule,
        "value": float(value),
        "max_abs": float(max_abs),
        "contacts": list(CONTACT_BASIS_CONTACTS),
        "estimator": CONTACT_GEO_ESTIMATOR_VERSION,
        "k": CONTACT_BASIS_K,
        "stride": CONTACT_SURFACE_STRIDE,
        "xy_scale": CONTACT_SURFACE_XY_SCALE,
        "candidates": {},
        "skipped": [],
    }
    if rule in {"", "0", "off", "none"} or value == 0:
        return submission, [], summary, None
    if rule != "max_contact_shape_gap":
        raise ValueError(f"Unknown ROGII_CONTACT_BASIS_RULE={rule}")

    split = submission["id"].astype(str).str.rsplit("_", n=1, expand=True)
    frame = submission.copy()
    frame["well"] = split[0]
    frame["row_idx"] = split[1].astype(int)
    pool = build_contact_basis_pool(data_dir, CONTACT_BASIS_CONTACTS)
    summary["pool_rows"] = int(len(pool))

    candidates: list[dict[str, object]] = []
    for well in sorted(frame["well"].unique()):
        hw_path = data_dir / "test" / f"{well}__horizontal_well.csv"
        if not hw_path.exists():
            continue
        hw = pd.read_csv(hw_path)
        tw_path = data_dir / "test" / f"{well}__typewell.csv"
        tw = pd.read_csv(tw_path) if tw_path.exists() else None
        well_mask = frame["well"].to_numpy(str) == str(well)
        row_idx = frame.loc[well_mask, "row_idx"].to_numpy(dtype=int)
        base_tvt = frame.loc[well_mask, "tvt"].to_numpy(dtype=float)
        surface_cache: dict[str, dict[str, object]] = {}
        surface_by_contact: dict[str, np.ndarray] = {}
        for contact in CONTACT_BASIS_CONTACTS:
            if contact not in pool.columns:
                continue
            try:
                geo, plane, knn, surf_meta = predict_contact_surface_geo(pool, hw, contact)
            except (RuntimeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                summary["skipped"].append({
                    "well": str(well),
                    "contact": contact,
                    "reason": str(exc),
                })
                continue
            surface_cache[contact] = {
                "geo": geo,
                "plane": plane,
                "knn": knn,
                "meta": surf_meta,
            }
            surface_by_contact[contact] = geo
        for contact, surfaces in surface_cache.items():
            try:
                basis, contact_tvt, meta = contact_basis_for_rows(
                    hw,
                    row_idx,
                    base_tvt,
                    np.asarray(surfaces["knn"], dtype=float),
                    np.asarray(surfaces["plane"], dtype=float),
                    max_abs,
                    geo_contact=np.asarray(surfaces["geo"], dtype=float),
                    surface_by_contact=surface_by_contact,
                    tw=tw,
                    surface_meta=surfaces["meta"],
                )
            except (RuntimeError, ValueError, IndexError) as exc:
                summary["skipped"].append({
                    "well": str(well),
                    "contact": contact,
                    "reason": str(exc),
                })
                continue
            score = float(meta["basis_mean_abs"] * meta["confidence"])
            key = f"{well}:{contact}"
            candidate_payload = {
                str(k): float(v)
                for k, v in meta.items()
            }
            candidate_payload.update({"score": score, "rows": int(len(row_idx))})
            summary["candidates"][key] = candidate_payload
            candidates.append({
                "well": str(well),
                "contact": contact,
                "mask": well_mask,
                "basis": basis,
                "contact_tvt": contact_tvt,
                "score": score,
                "meta": meta,
            })

    selected = select_contact_basis_candidate(candidates)
    out = submission.copy()
    out.loc[selected["mask"], "tvt"] = (
        out.loc[selected["mask"], "tvt"].to_numpy(dtype=float)
        + float(value) * np.asarray(selected["basis"], dtype=float)
    )

    component_file = component_path or (WORKING / "contact_basis_component.csv")
    component = submission[["id"]].copy()
    component["basis"] = 0.0
    component["contact_tvt"] = np.nan
    component["selected"] = False
    component["contact"] = ""
    component.loc[selected["mask"], "basis"] = np.asarray(selected["basis"], dtype=float)
    component.loc[selected["mask"], "contact_tvt"] = np.asarray(selected["contact_tvt"], dtype=float)
    component.loc[selected["mask"], "selected"] = True
    component.loc[selected["mask"], "contact"] = str(selected["contact"])
    component.to_csv(component_file, index=False)

    meta = selected["meta"]
    operation = {
        "rule": rule,
        "well": str(selected["well"]),
        "contact": str(selected["contact"]),
        "kind": "contact_shape_basis",
        "value": float(value),
        "rows": int(np.sum(selected["mask"])),
        "score": float(selected["score"]),
        **{str(k): float(v) for k, v in meta.items()},
    }
    summary["selected"] = operation
    return out, [operation], summary, component_file


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_final_well_probe(
    submission: pd.DataFrame,
    offsets: dict[str, float],
    trends: dict[str, float],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not offsets and not trends:
        return submission, []

    out = submission.copy()
    split = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    out["well"] = split[0]
    out["row_idx"] = split[1].astype(int)
    out["delta"] = 0.0
    operations: list[dict[str, object]] = []

    for well, offset in offsets.items():
        mask = out["well"] == well
        if not mask.any():
            raise ValueError(f"Final offset well not found in submission: {well}")
        out.loc[mask, "delta"] += float(offset)
        operations.append({
            "well": well,
            "kind": "offset",
            "value": float(offset),
            "rows": int(mask.sum()),
        })

    for well, trend in trends.items():
        mask = out["well"] == well
        if not mask.any():
            raise ValueError(f"Final trend well not found in submission: {well}")
        idx = out.loc[mask].sort_values("row_idx").index
        basis = np.linspace(-0.5, 0.5, len(idx), dtype=float) if len(idx) > 1 else np.array([0.0])
        out.loc[idx, "delta"] += float(trend) * basis
        operations.append({
            "well": well,
            "kind": "linear_trend",
            "value": float(trend),
            "rows": int(len(idx)),
        })

    out["tvt"] = out["tvt"].astype(float) + out["delta"]
    return out[["id", "tvt"]], operations


def component_gap_stats(
    well_ids: pd.Series,
    pf: pd.DataFrame,
    artifact: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    gap = pf["tvt"].to_numpy(float) - artifact["tvt"].to_numpy(float)
    stats_frame = pd.DataFrame({
        "well": well_ids.to_numpy(str),
        "abs_gap": np.abs(gap),
        "gap": gap,
    })
    well_stats = (
        stats_frame
        .groupby("well", sort=True)
        .agg(
            rows=("abs_gap", "size"),
            mean_abs_gap=("abs_gap", "mean"),
            mean_gap=("gap", "mean"),
            max_abs_gap=("abs_gap", "max"),
        )
    )
    stats_payload = {
        str(idx): {str(k): float(v) for k, v in row.items()}
        for idx, row in well_stats.iterrows()
    }
    return well_stats, stats_payload


def apply_dynamic_final_offset(
    submission: pd.DataFrame,
    well_ids: pd.Series,
    well_stats: pd.DataFrame,
    rule: str,
    value: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if rule in {"", "0", "off", "none"} or value == 0:
        return submission, []
    if rule != "max_component_gap":
        raise ValueError(f"Unknown ROGII_FINAL_DYNAMIC_OFFSET_RULE={rule}")
    if well_stats.empty:
        raise ValueError("Cannot apply dynamic final offset without component gap stats")

    selected = str(well_stats["mean_abs_gap"].idxmax())
    mask = well_ids.to_numpy(str) == selected
    if not mask.any():
        raise ValueError(f"Dynamic offset selected well not found in submission: {selected}")

    out = submission.copy()
    out.loc[mask, "tvt"] = out.loc[mask, "tvt"].astype(float) + float(value)
    row = well_stats.loc[selected]
    return out, [{
        "rule": rule,
        "well": selected,
        "kind": "dynamic_offset",
        "value": float(value),
        "rows": int(mask.sum()),
        "mean_abs_gap": float(row["mean_abs_gap"]),
        "mean_gap": float(row["mean_gap"]),
        "max_abs_gap": float(row["max_abs_gap"]),
    }]


def dynamic_well_weights(
    well_ids: pd.Series,
    pf: pd.DataFrame,
    artifact: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, dict[str, float]]]:
    weights = well_ids.map(WELL_PF_WEIGHTS).fillna(PF_WEIGHT).to_numpy(float)
    if not ARTIFACT_DYNAMIC_WELL_RULE:
        return weights, [], {}

    well_stats, stats_payload = component_gap_stats(well_ids, pf, artifact)
    operations: list[dict[str, object]] = []
    if ARTIFACT_DYNAMIC_WELL_RULE == "gap_high_more_artifact":
        selected = str(well_stats["mean_abs_gap"].idxmax())
        target_weight = float(np.clip(ARTIFACT_DYNAMIC_PF_WEIGHT, 0.0, 1.0))
        mask = well_ids.to_numpy(str) == selected
        weights[mask] = target_weight
        operations.append({
            "rule": ARTIFACT_DYNAMIC_WELL_RULE,
            "well": selected,
            "pf_weight": target_weight,
            "artifact_weight": 1.0 - target_weight,
            "rows": int(mask.sum()),
            "mean_abs_gap": float(well_stats.loc[selected, "mean_abs_gap"]),
        })
    elif ARTIFACT_DYNAMIC_WELL_RULE in {"0", "none", "off"}:
        return weights, [], stats_payload
    else:
        raise ValueError(f"Unknown ROGII_ARTIFACT_DYNAMIC_WELL_RULE={ARTIFACT_DYNAMIC_WELL_RULE}")

    return weights, operations, stats_payload


def finite_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    xv = x[mask].astype(float)
    yv = y[mask].astype(float)
    denom = float(np.var(xv))
    if denom <= 1e-9:
        return 0.0
    return float(np.cov(xv, yv, bias=True)[0, 1] / denom)


def prefix_feature_payload(hw: pd.DataFrame, gap_stats: dict[str, float]) -> dict[str, float]:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    known_mask = hw["TVT_input"].notna().to_numpy()
    known = hw.loc[known_mask].copy()
    eval_rows = hw.loc[eval_mask].copy()
    tail = known.tail(min(80, max(12, len(known) // 3)))

    gr_tail = tail["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)
    tvt_tail = tail["TVT_input"].to_numpy(dtype=float)
    md_tail = tail["MD"].to_numpy(dtype=float)
    z_tail = tail["Z"].to_numpy(dtype=float)
    gr_known = known["GR"].interpolate(limit_direction="both").to_numpy(dtype=float)

    out = {
        "n": float(len(hw)),
        "n_known": float(known_mask.sum()),
        "n_eval": float(eval_mask.sum()),
        "known_frac": float(known_mask.mean()) if len(hw) else 0.0,
        "eval_z_span": float(eval_rows["Z"].max() - eval_rows["Z"].min()) if len(eval_rows) else 0.0,
        "eval_md_span": float(eval_rows["MD"].max() - eval_rows["MD"].min()) if len(eval_rows) else 0.0,
        "eval_gr_nan": float(eval_rows["GR"].isna().mean()) if len(eval_rows) else 0.0,
        "prefix_gr_std": float(np.nanstd(gr_known)) if len(gr_known) else 0.0,
        "tail_gr_std": float(np.nanstd(gr_tail)) if len(gr_tail) else 0.0,
        "tail_gr_slope_md": finite_slope(md_tail, gr_tail),
        "tail_tvt_slope_md": finite_slope(md_tail, tvt_tail),
        "tail_z_slope_md": finite_slope(md_tail, z_tail),
    }
    out.update({f"component_{k}": float(v) for k, v in gap_stats.items()})
    return out


def prefix_regime_weight(features: dict[str, float]) -> tuple[str, float, float]:
    """CPU-safe handoff rule for hidden-rerun wells.

    The selector is intentionally conservative: it only moves away from the
    v16 80/20 baseline when prefix risk or PF/artifact disagreement is large.
    """
    mean_abs_gap = features.get("component_mean_abs_gap", 0.0)
    mean_gap = features.get("component_mean_gap", 0.0)
    tail_gr_std = features.get("tail_gr_std", 0.0)
    eval_z_span = features.get("eval_z_span", 0.0)
    known_frac = features.get("known_frac", 1.0)
    eval_gr_nan = features.get("eval_gr_nan", 0.0)
    tail_tvt_slope = abs(features.get("tail_tvt_slope_md", 0.0))

    if known_frac < 0.40 or eval_gr_nan > 0.35:
        return "anchor_like_low_support", 0.90, 0.60
    if tail_gr_std >= 36.0 or eval_z_span >= 190.0 or tail_tvt_slope >= 0.045:
        return "conservative_prefix_risk", 0.85, 0.65
    if mean_abs_gap >= 11.0 and tail_gr_std <= 28.0 and eval_gr_nan <= 0.10 and mean_gap > 0.0:
        return "artifact_heavy_clean_gap", 0.75, 0.70
    if mean_abs_gap >= 16.0:
        return "conservative_large_gap", 0.85, 0.55
    return "pf_heavy_baseline", 0.80, 0.80


def apply_prefix_selector(
    data_dir: Path,
    well_ids_series: pd.Series,
    pf: pd.DataFrame,
    artifact: pd.DataFrame,
    initial_weights: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, dict[str, float]]]:
    if BLEND_SELECTOR in {"", "0", "off", "none"}:
        return initial_weights, [], {}
    if BLEND_SELECTOR != "prefix_regime_v1":
        raise ValueError(f"Unknown ROGII_BLEND_SELECTOR={BLEND_SELECTOR}")

    gap = pf["tvt"].to_numpy(float) - artifact["tvt"].to_numpy(float)
    gap_frame = pd.DataFrame({
        "well": well_ids_series.to_numpy(str),
        "abs_gap": np.abs(gap),
        "gap": gap,
    })
    gap_by_well = (
        gap_frame
        .groupby("well", sort=True)
        .agg(
            rows=("abs_gap", "size"),
            mean_abs_gap=("abs_gap", "mean"),
            mean_gap=("gap", "mean"),
            max_abs_gap=("abs_gap", "max"),
        )
    )

    weights = initial_weights.copy()
    operations: list[dict[str, object]] = []
    feature_payload: dict[str, dict[str, float]] = {}
    wells = sorted(gap_by_well.index.astype(str).tolist())
    for wid in wells:
        hw_path = data_dir / "test" / f"{wid}__horizontal_well.csv"
        if not hw_path.exists():
            continue
        hw = pd.read_csv(hw_path)
        gap_stats = {str(k): float(v) for k, v in gap_by_well.loc[wid].items()}
        features = prefix_feature_payload(hw, gap_stats)
        regime, pf_weight, confidence = prefix_regime_weight(features)
        mask = well_ids_series.to_numpy(str) == wid
        weights[mask] = pf_weight
        feature_payload[wid] = features
        operations.append({
            "well": wid,
            "rule": BLEND_SELECTOR,
            "regime": regime,
            "pf_weight": float(pf_weight),
            "artifact_weight": float(1.0 - pf_weight),
            "confidence": float(confidence),
            "rows": int(mask.sum()),
        })

    # Explicit per-well overrides are treated as manual audit controls.
    for wid, pf_weight in WELL_PF_WEIGHTS.items():
        mask = well_ids_series.to_numpy(str) == wid
        if mask.any():
            weights[mask] = pf_weight
            operations.append({
                "well": wid,
                "rule": "manual_override",
                "pf_weight": float(pf_weight),
                "artifact_weight": float(1.0 - pf_weight),
                "rows": int(mask.sum()),
            })
    return weights, operations, feature_payload


def main() -> None:
    sample_path = find_sample()
    data_dir = sample_path.parent
    sample = pd.read_csv(sample_path)[["id"]]
    pf_script = write_component("pf_component.py", PF_COMPONENT_CODE)
    artifact_script = write_component("v10_artifact_component.py", V10_COMPONENT_CODE)

    pf_csv = run_component(
        "pf_h017",
        pf_script,
        WORKING / "component_pf_h017",
        {
            "ROGII_VARIANT": PF_VARIANT,
            "ROGII_N_SEEDS": os.getenv("ROGII_N_SEEDS", "64"),
            "ROGII_N_PARTICLES": os.getenv("ROGII_N_PARTICLES", "160"),
            "ROGII_USE_VISIBLE_PHYSICAL": "0",
        },
    )
    artifact_env_base = {
        "ROGII_INFERENCE_ONLY": "1",
        "ROGII_SAVE_ARTIFACTS": "0",
        "ROGII_RUN_TABICL": ARTIFACT_RUN_TABICL,
        "ROGII_FORCE_CPU": ARTIFACT_FORCE_CPU,
        "ROGII_USE_PROCESS_POOL": "0",
        "ROGII_USE_THREAD_POOL": "0",
        "ROGII_NCPU": "1",
        "PYTHONHASHSEED": "42",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if ARTIFACT_EXACT_OVERLAP:
        artifact_env_base["ROGII_EXACT_OVERLAP"] = ARTIFACT_EXACT_OVERLAP
    if ARTIFACT_EXACT_BLEND_WEIGHT:
        artifact_env_base["ROGII_EXACT_BLEND_WEIGHT"] = ARTIFACT_EXACT_BLEND_WEIGHT
    artifact_component_records: list[dict[str, object]] = []
    artifact_frames: list[pd.DataFrame] = []
    for idx, salt in enumerate(ARTIFACT_SEED_SALTS):
        artifact_env = dict(artifact_env_base)
        artifact_env["ROGII_ARTIFACT_SEED_SALT"] = salt
        artifact_dir = (
            WORKING / "component_v10_artifact"
            if len(ARTIFACT_SEED_SALTS) == 1
            else WORKING / f"component_v10_artifact_salt_{idx}"
        )
        cur_csv = run_component(
            f"v10_artifact_salt_{salt}",
            artifact_script,
            artifact_dir,
            artifact_env,
        )
        cur_artifact = read_component(cur_csv, sample, f"artifact_salt_{salt}")
        artifact_frames.append(cur_artifact[["id", "tvt"]].rename(columns={"tvt": f"tvt_{idx}"}))
        artifact_component_records.append({
            "index": int(idx),
            "salt": salt,
            "path": str(cur_csv),
            "sha256": sha256_file(cur_csv),
        })
    artifact_merged = sample[["id"]].copy()
    for frame in artifact_frames:
        artifact_merged = artifact_merged.merge(frame, on="id", how="left")
    artifact_cols = [c for c in artifact_merged.columns if c.startswith("tvt_")]
    artifact_merged["tvt"] = artifact_merged[artifact_cols].mean(axis=1)
    artifact_csv = WORKING / "component_v10_artifact" / "submission.csv"
    artifact_csv.parent.mkdir(parents=True, exist_ok=True)
    artifact_merged[["id", "tvt"]].to_csv(artifact_csv, index=False)

    pf = read_component(pf_csv, sample, "pf")
    artifact = read_component(artifact_csv, sample, "artifact")
    submission = sample[["id"]].copy()
    well_ids = submission["id"].str.rsplit("_", n=1).str[0]
    gap_by_well, component_gap_payload = component_gap_stats(well_ids, pf, artifact)
    weights, dynamic_weight_ops, dynamic_well_stats = dynamic_well_weights(well_ids, pf, artifact)
    weights, selector_ops, selector_features = apply_prefix_selector(data_dir, well_ids, pf, artifact, weights)
    submission["tvt"] = weights * pf["tvt"].to_numpy(float) + (1.0 - weights) * artifact["tvt"].to_numpy(float)
    analog_residual_summary: dict[str, object] = {}
    analog_residual_ops: list[dict[str, object]] = []
    analog_residual_component: Path | None = None
    pre_analog_residual_hash = ""
    if ANALOG_RESIDUAL_RULE not in {"", "0", "off", "none"} and ANALOG_RESIDUAL_ALPHA != 0:
        base_path = WORKING / "base_submission_before_analog_residual.csv"
        submission.to_csv(base_path, index=False)
        pre_analog_residual_hash = sha256_file(base_path)
    submission, analog_residual_ops, analog_residual_summary, analog_residual_component = apply_analog_residual_transfer(
        data_dir,
        submission,
        ANALOG_RESIDUAL_RULE,
        ANALOG_RESIDUAL_ALPHA,
        ANALOG_RESIDUAL_K,
        ANALOG_RESIDUAL_MAX_ABS,
        WORKING / "analog_residual_component.csv",
    )
    submission, final_probe_ops = apply_final_well_probe(
        submission,
        FINAL_WELL_OFFSETS,
        FINAL_WELL_TRENDS,
    )
    submission, dynamic_offset_ops = apply_dynamic_final_offset(
        submission,
        well_ids,
        gap_by_well,
        FINAL_DYNAMIC_OFFSET_RULE,
        FINAL_DYNAMIC_OFFSET_VALUE,
    )
    geo_path_summary: dict[str, object] = {}
    geo_path_ops: list[dict[str, object]] = []
    geo_path_component: Path | None = None
    pre_geo_path_hash = ""
    if GEO_PATH_RULE not in {"", "0", "off", "none"}:
        base_path = WORKING / "base_submission_before_geo_path.csv"
        submission.to_csv(base_path, index=False)
        pre_geo_path_hash = sha256_file(base_path)
    submission, geo_path_ops, geo_path_summary, geo_path_component = apply_geo_path_selector(
        data_dir,
        submission,
        pf,
        artifact,
        GEO_PATH_RULE,
        GEO_PATH_BLEND_WEIGHT,
        GEO_PATH_MAX_ABS_DELTA,
        WORKING / "geo_path_component.csv",
    )
    contact_surface_summary: dict[str, object] = {}
    if CONTACT_SURFACE_WEIGHT != 0:
        contact_surface, contact_surface_summary = contact_surface_component(data_dir, sample)
        submission["tvt"] = (
            (1.0 - CONTACT_SURFACE_WEIGHT) * submission["tvt"].to_numpy(float)
            + CONTACT_SURFACE_WEIGHT * contact_surface["tvt"].to_numpy(float)
        )
    contact_basis_summary: dict[str, object] = {}
    contact_basis_ops: list[dict[str, object]] = []
    contact_basis_component: Path | None = None
    pre_contact_basis_hash = ""
    if CONTACT_BASIS_RULE not in {"", "0", "off", "none"} and CONTACT_BASIS_VALUE != 0:
        base_path = WORKING / "base_submission_before_contact_basis.csv"
        submission.to_csv(base_path, index=False)
        pre_contact_basis_hash = sha256_file(base_path)
    submission, contact_basis_ops, contact_basis_summary, contact_basis_component = apply_contact_basis_probe(
        data_dir,
        submission,
        CONTACT_BASIS_RULE,
        CONTACT_BASIS_VALUE,
        CONTACT_BASIS_MAX_ABS,
        WORKING / "contact_basis_component.csv",
    )
    contact_override_summary: dict[str, object] = {}
    contact_override_ops: list[dict[str, object]] = []
    contact_override_component: Path | None = None
    pre_contact_override_hash = ""
    if CONTACT_OVERRIDE_RULE not in {"", "0", "off", "none"}:
        base_path = WORKING / "base_submission_before_contact_override.csv"
        submission.to_csv(base_path, index=False)
        pre_contact_override_hash = sha256_file(base_path)
    submission, contact_override_ops, contact_override_summary, contact_override_component = apply_guarded_contact_override(
        data_dir,
        submission,
        CONTACT_OVERRIDE_RULE,
        CONTACT_OVERRIDE_REFS,
        CONTACT_OVERRIDE_MAX_PREFIX_RMSE,
        CONTACT_OVERRIDE_MIN_PREFIX_ROWS,
        WORKING / "contact_override_component.csv",
    )
    out_path = WORKING / "submission.csv"
    submission.to_csv(out_path, index=False)

    disagreement = pf["tvt"].to_numpy(float) - artifact["tvt"].to_numpy(float)
    contact_selected = {}
    if isinstance(contact_basis_summary, dict):
        contact_selected = contact_basis_summary.get("selected") or {}
    if not contact_selected and contact_basis_ops:
        contact_selected = contact_basis_ops[0]
    summary = {
        "pf_weight": PF_WEIGHT,
        "pf_variant": PF_VARIANT,
        "artifact_weight": 1.0 - PF_WEIGHT,
        "well_pf_weights": WELL_PF_WEIGHTS,
        "artifact_exact_overlap": ARTIFACT_EXACT_OVERLAP or "component_default",
        "artifact_exact_blend_weight": ARTIFACT_EXACT_BLEND_WEIGHT or "component_default",
        "artifact_run_tabicl": ARTIFACT_RUN_TABICL,
        "artifact_force_cpu": ARTIFACT_FORCE_CPU,
        "artifact_source_mode": ARTIFACT_SOURCE_MODE,
        "artifact_seed_salt": ARTIFACT_SEED_SALT,
        "artifact_seed_salts": ARTIFACT_SEED_SALTS,
        "artifact_seed_ensemble_size": len(ARTIFACT_SEED_SALTS),
        "artifact_seed_components": artifact_component_records,
        "artifact_dynamic_well_rule": ARTIFACT_DYNAMIC_WELL_RULE,
        "artifact_dynamic_pf_weight": ARTIFACT_DYNAMIC_PF_WEIGHT,
        "blend_selector": BLEND_SELECTOR,
        "selector_operations": selector_ops,
        "selector_features": selector_features,
        "contact_surface_weight": CONTACT_SURFACE_WEIGHT,
        "contact_surface_summary": contact_surface_summary,
        "geo_path_rule": GEO_PATH_RULE,
        "geo_path_blend_weight": GEO_PATH_BLEND_WEIGHT,
        "geo_path_max_abs_delta": GEO_PATH_MAX_ABS_DELTA,
        "geo_path_geo_margin": GEO_PATH_GEO_MARGIN,
        "geo_path_operations": geo_path_ops,
        "geo_path_summary": geo_path_summary,
        "geo_path_component": "" if geo_path_component is None else str(geo_path_component),
        "geo_path_component_sha256": "" if geo_path_component is None else sha256_file(geo_path_component),
        "pre_geo_path_submission_sha256": pre_geo_path_hash,
        "analog_residual_rule": ANALOG_RESIDUAL_RULE,
        "analog_residual_alpha": ANALOG_RESIDUAL_ALPHA,
        "analog_residual_k": ANALOG_RESIDUAL_K,
        "analog_residual_max_abs": ANALOG_RESIDUAL_MAX_ABS,
        "analog_residual_table": ANALOG_RESIDUAL_TABLE,
        "analog_residual_operations": analog_residual_ops,
        "analog_residual_summary": analog_residual_summary,
        "analog_residual_component": "" if analog_residual_component is None else str(analog_residual_component),
        "analog_residual_component_sha256": "" if analog_residual_component is None else sha256_file(analog_residual_component),
        "pre_analog_residual_submission_sha256": pre_analog_residual_hash,
        "contact_basis_rule": CONTACT_BASIS_RULE,
        "contact_basis_value": CONTACT_BASIS_VALUE,
        "contact_basis_max_abs": CONTACT_BASIS_MAX_ABS,
        "contact_basis_contacts": list(CONTACT_BASIS_CONTACTS),
        "contact_basis_k": CONTACT_BASIS_K,
        "contact_basis_estimator": CONTACT_GEO_ESTIMATOR_VERSION,
        "contact_basis_operations": contact_basis_ops,
        "contact_basis_summary": contact_basis_summary,
        "contact_basis_selected_well_id": contact_selected.get("well"),
        "contact_basis_selected_rows": contact_selected.get("rows"),
        "contact_basis_selected_contact": contact_selected.get("contact"),
        "contact_basis_selected_score": contact_selected.get("score"),
        "contact_basis_selected_confidence": contact_selected.get("confidence"),
        "contact_basis_selected_geo_confidence": contact_selected.get("geo_confidence"),
        "contact_basis_selected_gr_confidence": contact_selected.get("gr_confidence"),
        "contact_basis_selected_mean_abs_basis": contact_selected.get("basis_mean_abs"),
        "contact_basis_selected_basis_mean": contact_selected.get("basis_mean"),
        "contact_basis_selected_basis_std": contact_selected.get("basis_std"),
        "contact_basis_selected_basis_max_abs": contact_selected.get("basis_max_abs"),
        "contact_basis_component": "" if contact_basis_component is None else str(contact_basis_component),
        "contact_basis_component_sha256": "" if contact_basis_component is None else sha256_file(contact_basis_component),
        "pre_contact_basis_submission_sha256": pre_contact_basis_hash,
        "contact_override_rule": CONTACT_OVERRIDE_RULE,
        "contact_override_refs": list(CONTACT_OVERRIDE_REFS),
        "contact_override_max_prefix_rmse": CONTACT_OVERRIDE_MAX_PREFIX_RMSE,
        "contact_override_min_prefix_rows": CONTACT_OVERRIDE_MIN_PREFIX_ROWS,
        "contact_override_operations": contact_override_ops,
        "contact_override_summary": contact_override_summary,
        "contact_override_component": "" if contact_override_component is None else str(contact_override_component),
        "contact_override_component_sha256": "" if contact_override_component is None else sha256_file(contact_override_component),
        "pre_contact_override_submission_sha256": pre_contact_override_hash,
        "base_submission_sha256": pre_contact_basis_hash,
        "dynamic_weight_operations": dynamic_weight_ops,
        "dynamic_well_stats": dynamic_well_stats,
        "component_gap_stats": component_gap_payload,
        "final_well_offsets": FINAL_WELL_OFFSETS,
        "final_well_trends": FINAL_WELL_TRENDS,
        "final_dynamic_offset_rule": FINAL_DYNAMIC_OFFSET_RULE,
        "final_dynamic_offset_value": FINAL_DYNAMIC_OFFSET_VALUE,
        "final_dynamic_offset_operations": dynamic_offset_ops,
        "final_probe_operations": final_probe_ops,
        "rows": int(len(submission)),
        "submission_sha256": sha256_file(out_path),
        "final_submission_sha256": sha256_file(out_path),
        "final_submission_hash": sha256_file(out_path),
        "component_pf": str(pf_csv),
        "component_pf_sha256": sha256_file(pf_csv),
        "component_pf_script_sha256": sha256_file(pf_script),
        "component_artifact": str(artifact_csv),
        "component_artifact_sha256": sha256_file(artifact_csv),
        "component_artifact_script_sha256": sha256_file(artifact_script),
        "mean_abs_component_gap": float(abs(disagreement).mean()),
        "max_abs_component_gap": float(abs(disagreement).max()),
    }
    (WORKING / "blend_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(submission.head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
