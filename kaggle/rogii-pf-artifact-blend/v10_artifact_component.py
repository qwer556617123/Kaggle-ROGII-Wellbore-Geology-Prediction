# %% cell 1
from __future__ import annotations

import gc
import json
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Kaggle inference notebook defaults: load the artifact dataset we created locally.
os.environ.setdefault("ROGII_INFERENCE_ONLY", "1")
os.environ.setdefault("ROGII_SAVE_ARTIFACTS", "0")
os.environ.setdefault("ROGII_RUN_TABICL", "1")

# ── optional numba JIT ──────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    def _njit(*a, **kw):
        def _wrap(f): return f
        return _wrap
    _NUMBA = False

print("NUMBA:", _NUMBA)

SEED = 42
np.random.seed(SEED)
# Use ALL available cores — Kaggle typically gives 4 (sometimes 2×4 on multi-GPU)
def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return int(default)
    return int(value)


RUNNING_ON_KAGGLE = (
    Path("/kaggle/input").exists()
    or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    or bool(os.environ.get("KAGGLE_URL_BASE"))
)

NCPU = max(1, min(multiprocessing.cpu_count(), env_int("ROGII_NCPU", multiprocessing.cpu_count())))
print(f"NCPU={NCPU}")

# ══════════════════════════════════════════════════════════════════════════════
# DEBUG CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DBG_VERBOSE        = env_flag("ROGII_DEBUG_VERBOSE", False)
DBG_SINGLE_WELL    = env_flag("ROGII_SMOKE_WELL", False)
DBG_SINGLE_TEST    = env_flag("ROGII_SMOKE_TEST", False)
DBG_PARALLEL_STATS = env_flag("ROGII_PARALLEL_STATS", True)
DBG_NAN_AUDIT      = env_flag("ROGII_NAN_AUDIT", False)
DBG_FEATURE_AUDIT  = env_flag("ROGII_FEATURE_AUDIT", False)
DBG_MAX_ERRORS     = 20
_dbg_error_count   = 0

_T0_GLOBAL = time.time()


def _dbg(msg: str, arr: np.ndarray = None, level: str = "INFO") -> None:
    if not DBG_VERBOSE:
        return
    elapsed = time.time() - _T0_GLOBAL
    prefix  = f"[{elapsed:8.1f}s][{level}] "
    print(prefix + msg, file=sys.stderr, flush=True)
    if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
        try:
            nans = int(np.isnan(arr.astype(float)).sum())
            infs = int(np.isinf(arr.astype(float)).sum())
            print(
                f"{prefix}  shape={arr.shape} dtype={arr.dtype} "
                f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g} "
                f"nan={nans} inf={infs}",
                file=sys.stderr, flush=True,
            )
        except Exception:
            print(f"{prefix}  shape={arr.shape} dtype={arr.dtype} (stats failed)",
                  file=sys.stderr, flush=True)


def _dbg_df(tag: str, df: pd.DataFrame) -> None:
    if not DBG_VERBOSE:
        return
    elapsed = time.time() - _T0_GLOBAL
    nan_cols = df.isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]
    print(
        f"[{elapsed:8.1f}s][DF] {tag}: shape={df.shape}  "
        f"nan_cols={len(nan_cols)}/{len(df.columns)}",
        file=sys.stderr, flush=True,
    )
    if len(nan_cols) > 0 and DBG_VERBOSE:
        top = nan_cols.nlargest(10)
        print(f"  top nan cols: {dict(top)}", file=sys.stderr, flush=True)


def _log_error(ctx: str, exc: Exception) -> None:
    global _dbg_error_count
    _dbg_error_count += 1
    if _dbg_error_count > DBG_MAX_ERRORS:
        if _dbg_error_count == DBG_MAX_ERRORS + 1:
            print(f"[ERROR] Max error log limit ({DBG_MAX_ERRORS}) reached.",
                  file=sys.stderr, flush=True)
        return
    elapsed = time.time() - _T0_GLOBAL
    tb = traceback.format_exc()
    print(
        f"[{elapsed:8.1f}s][ERROR] {ctx}: {type(exc).__name__}: {exc}\n{tb}",
        file=sys.stderr, flush=True,
    )


# ── data paths ──────────────────────────────────────────────────────────────
def _find() -> Path:
    candidates = []
    if os.environ.get("ROGII_DATA_DIR"):
        candidates.append(Path(os.environ["ROGII_DATA_DIR"]))
    candidates.extend([
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path.cwd(),
        Path.cwd() / "rogii-wellbore-geology-prediction",
        Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parents[2] / "rogii-wellbore-geology-prediction" if "__file__" in globals() else Path.cwd(),
    ])
    for p in candidates:
        if (p / "train").is_dir() and (p / "test").is_dir() and (p / "sample_submission.csv").is_file():
            return p
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for sample in input_root.glob("**/sample_submission.csv"):
            p = sample.parent
            if (p / "train").is_dir() and (p / "test").is_dir():
                return p
    raise FileNotFoundError("Data not found")


def _default_output_dir() -> Path:
    if RUNNING_ON_KAGGLE:
        return Path("/kaggle/working")
    if os.environ.get("ROGII_OUTPUT_DIR"):
        return Path(os.environ["ROGII_OUTPUT_DIR"])
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


DATA       = _find()
TRAIN_DIR  = DATA / "train"
TEST_DIR   = DATA / "test"
SAMPLE     = DATA / "sample_submission.csv"
OUTPUT_DIR = _default_output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT        = OUTPUT_DIR / "submission.csv"

SAVE_ARTIFACTS = env_flag("ROGII_SAVE_ARTIFACTS", False)
ARTIFACT_DIR = Path(os.environ.get("ROGII_ARTIFACT_DIR", OUTPUT_DIR / "model_artifacts"))
ARTIFACT_MANIFEST = {
    "version": "v10_fresh_artifact_train",
    "goal": "public_score_9.5",
    "lgb": [],
    "catboost": [],
    "tabicl_contexts": [],
    "created_outputs": {},
}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _artifact_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ARTIFACT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_artifact_dir() -> Path:
    candidates: list[Path] = []
    if os.environ.get("ROGII_ARTIFACT_DIR"):
        candidates.append(Path(os.environ["ROGII_ARTIFACT_DIR"]))
    candidates.append(ARTIFACT_DIR)
    for root in [Path("/kaggle/input"), DATA, OUTPUT_DIR]:
        if root.exists():
            candidates.extend(p.parent for p in root.glob("**/manifest.json"))
    for candidate in dict.fromkeys(candidates):
        if (candidate / "manifest.json").exists():
            return candidate
    raise FileNotFoundError(
        "No artifact manifest found. Set ROGII_ARTIFACT_DIR to the model_artifacts folder."
    )


if SAVE_ARTIFACTS:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Artifact saving enabled: {ARTIFACT_DIR}", flush=True)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return float(default)
    return float(value)


def apply_exact_train_coordinate_blend(sub: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Optional public train-coordinate blend; disabled with ROGII_EXACT_OVERLAP=0."""
    if os.environ.get("ROGII_EXACT_OVERLAP", "1").strip().lower() in {"0", "false", "no"}:
        print("Exact train-coordinate overlap override disabled.", flush=True)
        return sub
    blend_weight = env_float("ROGII_EXACT_BLEND_WEIGHT", 0.28)
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))

    train_parts = []
    for p in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
        try:
            cur = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"])
        except Exception:
            continue
        cur = cur[cur["TVT"].notna()].copy()
        if not cur.empty:
            train_parts.append(cur)
    if not train_parts:
        print("Exact overlap: no train coordinate rows found.", flush=True)
        return sub

    train_all = pd.concat(train_parts, ignore_index=True)
    for col in ["X", "Y", "Z"]:
        train_all[col + "_r"] = train_all[col].round(2)
    train_map = (
        train_all
        .drop_duplicates(subset=["X_r", "Y_r", "Z_r"])
        .set_index(["X_r", "Y_r", "Z_r"])["TVT"]
        .to_dict()
    )

    coord_parts = []
    for p in sorted((data_dir / "test").glob("*__horizontal_well.csv")):
        wid = p.name.split("__")[0]
        try:
            cur = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT_input"])
        except Exception:
            continue
        mask = cur["TVT_input"].isna().to_numpy()
        if not mask.any():
            continue
        row_idx = np.arange(len(cur))[mask]
        part = cur.loc[mask, ["X", "Y", "Z"]].copy()
        part["id"] = [f"{wid}_{int(i)}" for i in row_idx]
        coord_parts.append(part)
    if not coord_parts:
        print("Exact overlap: no test prediction rows found.", flush=True)
        return sub

    coord = pd.concat(coord_parts, ignore_index=True)
    coord["key"] = list(zip(coord["X"].round(2), coord["Y"].round(2), coord["Z"].round(2)))
    coord["exact_tvt"] = coord["key"].map(train_map)
    exact = coord[coord["exact_tvt"].notna()][["id", "exact_tvt"]]
    if exact.empty:
        print("Exact overlap: 0 rows replaced.", flush=True)
        return sub

    out = sub.merge(exact, on="id", how="left")
    mask = out["exact_tvt"].notna()
    out.loc[mask, "tvt"] = (
        (1.0 - blend_weight) * out.loc[mask, "tvt"].astype(float)
        + blend_weight * out.loc[mask, "exact_tvt"].astype(float)
    )
    print(
        f"Exact overlap blend: blended {int(mask.sum())}/{len(out)} rows "
        f"with weight={blend_weight:.3f}.",
        flush=True,
    )
    return out[["id", "tvt"]]

TRAIN_CORE_CACHE_NAME = "aeroridge_train_core_df.pkl"
CACHE_ONLY            = env_flag("ROGII_CACHE_ONLY", False)
USE_TRAIN_CORE_CACHE  = env_flag("ROGII_USE_TRAIN_CORE_CACHE", True)
WRITE_TRAIN_CORE_CACHE = env_flag("ROGII_WRITE_TRAIN_CORE_CACHE", not RUNNING_ON_KAGGLE)
RUN_TABICL            = env_flag("ROGII_RUN_TABICL", True)
INFERENCE_ONLY        = env_flag("ROGII_INFERENCE_ONLY", False)
SAVE_FEATURE_FRAMES   = env_flag("ROGII_SAVE_FEATURE_FRAMES", False)
USE_HILL_STACK        = env_flag("ROGII_USE_HILL_STACK", True)
HILL_PRECISION        = env_float("ROGII_HILL_PRECISION", 0.01)
DEBUG_MAX_TRAIN_WELLS = env_int("ROGII_DEBUG_MAX_TRAIN_WELLS", 0)
DEBUG_MAX_TEST_WELLS  = env_int("ROGII_DEBUG_MAX_TEST_WELLS", 0)


def _cache_candidates() -> list[Path]:
    paths = []
    if os.environ.get("ROGII_TRAIN_CORE_CACHE"):
        paths.append(Path(os.environ["ROGII_TRAIN_CORE_CACHE"]))
    for root in [Path("/kaggle/input"), OUTPUT_DIR]:
        if root.exists():
            paths.extend(root.glob(f"**/{TRAIN_CORE_CACHE_NAME}"))
    return list(dict.fromkeys(paths))


def _train_core_cache_path_for_write() -> Path:
    if os.environ.get("ROGII_TRAIN_CORE_CACHE_OUT"):
        return Path(os.environ["ROGII_TRAIN_CORE_CACHE_OUT"])
    return OUTPUT_DIR / TRAIN_CORE_CACHE_NAME


print(f"DATA={DATA}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K    = 10
DENSE_SPW  = 60
DENSE_K    = 20
N_SPLITS   = 5
N_AUG_SPLITS      = 1#3
MIN_KNOWN_FOR_AUG = 20

'''BEAMS = [
    (10, 20.0, 144.0, 3, "cons"),
    (10,  8.0,  64.0, 3, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0,  90.0, 5, "sm5"),
    (20,  4.0,  36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]'''
BEAMS = [
    (10, 20.0, 144.0, 3, "cons"),
    (10,  8.0,  64.0, 3, "loose"),
    (10, 14.0,  90.0, 5, "sm5"),
]

PF_N = 150; ANCC_N = 150
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.0; PF_GR_SIG_MAX = 60.0; PF_GR_SIG_DEF = 30.0
PF_INIT_V_STD = 0.02; PF_INIT_SPR = 0.5; PF_RESAMP = 0.5
PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IR = 0.01; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

NOTEBOOK_RUN_VERSION = "ROGII_v26_combinedbest_lgbfix_cb3_tabicl_splitgpu_2026_05_10"

LGB_P = dict(
    boosting_type="gbdt",
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.1,
    objective="regression",
    verbose=-1,
    n_jobs=-1,
    device_type="gpu",
    gpu_use_dp=False,
    max_bin=255,
)
LGB_SEEDS = [42, 7, 123]

import subprocess as _s


def _gpu_names() -> list[str]:
    try:
        out = _s.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


GPU_NAMES = _gpu_names()
GPU_COUNT = len(GPU_NAMES)
CATBOOST_DEVICES = os.environ.get(
    "ROGII_CATBOOST_DEVICES",
    ":".join(str(i) for i in range(GPU_COUNT)) if GPU_COUNT else "0",
)

CB_P = dict(
    iterations=5000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
    min_data_in_leaf=20, loss_function="RMSE",
    task_type="GPU", devices=CATBOOST_DEVICES, od_type="Iter", od_wait=150, verbose=0,
)
FORCE_CPU = env_flag("ROGII_FORCE_CPU", False)
if FORCE_CPU:
    LGB_P["device_type"] = "cpu"
    LGB_P.pop("gpu_use_dp", None)
    LGB_P.pop("max_bin", None)
    CB_P["task_type"] = "CPU"
    CB_P.pop("devices", None)

print("GPUs:", "\n".join(GPU_NAMES) if GPU_NAMES else "(none)")
print(f"CatBoost devices: {CATBOOST_DEVICES}")
print(f"FORCE_CPU={FORCE_CPU}")
print(f"CPUs={NCPU}  train={len(list(TRAIN_DIR.glob('*__horizontal_well.csv')))} wells")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def nn_idx(arr: np.ndarray, v: float) -> int:
    i = int(np.searchsorted(arr, v, "left"))
    n = len(arr)
    if i >= n:
        return n - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr: np.ndarray, tw_at_k: np.ndarray, min_pts: int = 20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        bias = float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
        return 1.0, bias
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def wls_b_well(ktvt: np.ndarray, kz: np.ndarray, form_kn_col: np.ndarray,
               decay: float = 0.02) -> float:
    n = len(ktvt)
    if n < 3:
        return float(np.median(ktvt + kz - form_kn_col))
    w = np.exp(decay * np.arange(n, dtype=np.float64))
    w /= w.sum()
    return float(np.dot(w, ktvt + kz - form_kn_col))


def safe_interp(x, xp, fp):
    xp = np.asarray(xp, float); fp = np.asarray(fp, float)
    m = np.isfinite(xp) & np.isfinite(fp)
    if m.sum() < 2:
        return np.full(len(np.asarray(x)), np.nan)
    o = np.argsort(xp[m])
    return np.interp(np.asarray(x, float), xp[m][o], fp[m][o],
                     left=np.nan, right=np.nan)


def _rolling_std_cumsum(a: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(a.astype(np.float64))
    return s.rolling(w, center=True, min_periods=1).std().fillna(0.0).to_numpy(np.float32)


def _build_gr_rolls(gr_vals: np.ndarray, ev_iloc: np.ndarray):
    """
    Compute all GR rolling features for the eval rows in one pass.
    Single Series construction; all window sizes computed from it.
    """
    s = pd.Series(gr_vals.astype(np.float64))
    out = {}
    # All rolling windows in one loop to share the Series object
    for w in (5, 21, 51, 101):
        rm = s.rolling(w, center=True, min_periods=1)
        mean_arr = rm.mean().to_numpy(np.float32)
        std_arr  = rm.std().fillna(0.0).to_numpy(np.float32)
        out[f"grm{w}"] = mean_arr[ev_iloc]
        out[f"grs{w}"] = std_arr[ev_iloc]
    # Lags / leads — computed from the base series, not rolling
    for lag in (1, 5, 15, 30):
        out[f"glag{lag}"]  = s.shift(lag ).bfill().to_numpy(np.float32)[ev_iloc]
        out[f"glead{lag}"] = s.shift(-lag).ffill().to_numpy(np.float32)[ev_iloc]
    diff1 = s.diff().fillna(0.0).to_numpy(np.float32)
    diff2 = s.diff().diff().fillna(0.0).to_numpy(np.float32)
    out["gr_d1"] = diff1[ev_iloc]
    out["gr_d2"] = diff2[ev_iloc]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-scale self-correlation
# ═══════════════════════════════════════════════════════════════════════════════

def multi_scale_sc(kgr: np.ndarray, ktvt: np.ndarray, hgr: np.ndarray,
                   hws=(8, 15, 25), stride: int = 3):
    fallback = float(ktvt[-1]) if len(ktvt) > 0 else 0.0
    nh = len(hgr); nk = len(kgr)
    results = []

    # Smooth once; share across all scales
    kg_sm = (pd.Series(kgr).rolling(5, center=True, min_periods=1)
             .mean().to_numpy(np.float32))
    hg_sm = (pd.Series(hgr).rolling(5, center=True, min_periods=1)
             .mean().to_numpy(np.float32))

    # Build eval Hankel matrix once for the largest window; slice for smaller ones
    max_hw = max(hws)
    max_win = 2 * max_hw + 1
    _hp_max = np.pad(hg_sm, max_hw, mode="edge")
    # Pre-build the full H matrix for max window (reused via slicing below)
    if nh > 0 and nk >= max_win + 1:
        _H_max = _hp_max[np.arange(nh)[:, None] + np.arange(max_win)[None, :]].astype(np.float32)
    else:
        _H_max = None

    for hw_sc in hws:
        win = 2 * hw_sc + 1
        if nk < win + 1 or nh == 0:
            results.append((np.full(nh, fallback, np.float32),
                            np.zeros(nh, np.float32)))
            continue

        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            results.append((np.full(nh, fallback, np.float32),
                            np.zeros(nh, np.float32)))
            continue

        # Known-side Hankel
        idx_mat = sts[:, None] + np.arange(win, dtype=np.int32)[None, :]
        C  = kg_sm[idx_mat].astype(np.float32)
        mu = C.mean(1, keepdims=True); sd = C.std(1, keepdims=True) + 1e-6
        Cn = (C - mu) / sd

        # Eval-side Hankel: reuse _H_max by slicing center columns when possible
        if _H_max is not None and hw_sc == max_hw:
            H = _H_max
        else:
            # For smaller windows, re-pad efficiently
            pad_h = hw_sc
            hp = np.pad(hg_sm, pad_h, mode="edge")
            H  = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)

        mu_h = H.mean(1, keepdims=True); sd_h = H.std(1, keepdims=True) + 1e-6
        Hn   = (H - mu_h) / sd_h

        # Matrix NCC: (nh × ns) — use float32 matmul (faster on GPU-less CPU)
        ncc  = np.dot(Hn, Cn.T) / win          # shape (nh, ns)
        best  = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        ctrs  = np.clip(sts[best] + hw_sc, 0, nk - 1)
        results.append((ktvt[ctrs].astype(np.float32), score))

    return results


def gr_envelope(gr: np.ndarray, w: int = 21) -> np.ndarray:
    return (pd.Series(gr).rolling(w, center=True, min_periods=1)
            .max().to_numpy(np.float32))


def gr_energy(gr: np.ndarray, w: int = 21) -> np.ndarray:
    sq = gr.astype(np.float64) ** 2
    return np.sqrt(
        pd.Series(sq).rolling(w, center=True, min_periods=1)
        .mean().to_numpy().clip(0)
    ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Beam Search  (vectorised; ±2 delta; boolean seen-array instead of set)
# ═══════════════════════════════════════════════════════════════════════════════

_DELTAS = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
_ND     = len(_DELTAS)


def beam_search(gr_h: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                start_tvt: float, bs: int = 10, mc: float = 20.0,
                es: float = 144.0, r: int = 2) -> np.ndarray:
    tw_tvt = np.asarray(tw_tvt, np.float32)
    tw_gr  = np.asarray(tw_gr,  np.float32)
    T  = len(tw_tvt)
    fb = float(np.nanmean(tw_gr))

    sg = pd.Series(gr_h, dtype="float32").interpolate(
        limit_direction="both").fillna(fb)
    if r > 0:
        sg = sg.rolling(r * 2 + 1, center=True, min_periods=1).mean()
    sg = sg.to_numpy(np.float32)

    si = nn_idx(tw_tvt, start_tvt)
    ns = len(sg)

    bps = np.empty((ns, bs), np.int32)
    bpb = np.empty((ns, bs), np.int32)

    bi  = np.full(bs, si, np.int32)
    bc  = np.zeros(bs, np.float64)

    # Pre-allocate movement penalty (constant across steps)
    mv = mc * np.abs(_DELTAS).astype(np.float64)   # shape (ND,)

    # Boolean seen array — O(T) reset but avoids set hashing
    _seen = np.zeros(T, dtype=np.bool_)

    for s, gv in enumerate(sg):
        ci = np.clip(bi[:, None] + _DELTAS[None, :], 0, T - 1)   # (bs, ND)
        em = (gv - tw_gr[ci]) ** 2 / es                            # (bs, ND)
        cc = bc[:, None] + em + mv[None, :]                        # (bs, ND)

        fi  = ci.ravel()       # (bs*ND,)
        fc  = cc.ravel()       # (bs*ND,)
        fp  = np.repeat(np.arange(bs, dtype=np.int32), _ND)

        ord_ = np.argsort(fc, kind="stable")

        # Reset only the positions we used last step (O(bs*ND) not O(T))
        _seen[:] = False
        kept = []
        for o in ord_:
            t = int(fi[o])
            if not _seen[t]:
                _seen[t] = True
                kept.append(o)
            if len(kept) == bs:
                break
        while len(kept) < bs:
            kept.append(kept[-1])
        kept_arr = np.array(kept, np.int32)

        bps[s] = fp[kept_arr]
        bpb[s] = fi[kept_arr]
        bi = fi[kept_arr].astype(np.int32)
        bc = fc[kept_arr]

    path = np.empty(ns, np.int32)
    cb   = int(np.argmin(bc))
    for s in range(ns - 1, -1, -1):
        path[s] = bpb[s, cb]
        cb      = bps[s, cb]

    return tw_tvt[path]


def run_all_beams(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                  last_tvt: float, gr_filled: pd.Series,
                  eval_start_idx: int) -> dict:
    hgr    = gr_filled.iloc[eval_start_idx:].to_numpy(np.float32)
    n_eval = int((hw["TVT_input"].isna()).sum())
    result = {}
    for bs, mc, es, r, tag in BEAMS:
        pred = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
        result[tag] = pred[:n_eval].astype(np.float32)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Particle Filters
# ═══════════════════════════════════════════════════════════════════════════════

def _cal_gr_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return PF_GR_SIG_DEF
    ex = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    return float(np.clip(np.std(kn["GR"].values - ex),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _z_beta(hw: pd.DataFrame):
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 30:
        return -1.0, 0.0, 0.1
    dz   = np.diff(kn["Z"].values)
    dtvt = np.diff(kn["TVT_input"].values)
    dmd  = np.diff(kn["MD"].values)
    m    = dmd > 0
    if m.sum() < 10:
        return -1.0, 0.0, 0.1
    vz = dz[m] / dmd[m]; vt = dtvt[m] / dmd[m]
    A  = np.column_stack([vz, np.ones_like(vz)])
    c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
    return float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)


def _init_v(hw: pd.DataFrame) -> float:
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 10:
        return 0.0
    tail = kn.tail(20)
    dtvt = np.diff(tail["TVT_input"].values)
    dmd  = np.diff(tail["MD"].values)
    m    = dmd > 0
    return 0.0 if m.sum() < 3 else float(np.median(dtvt[m] / dmd[m]))


# ── numba-compiled PF loop cores ─────────────────────────────────────────────

@_njit(cache=True)
def _numba_interp1(xp: np.ndarray, fp: np.ndarray, x: float) -> float:
    n = len(xp)
    if x <= xp[0]:  return fp[0]
    if x >= xp[-1]: return fp[-1]
    lo = 0; hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= x: lo = mid
        else:             hi = mid
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])


@_njit(cache=True)
def _pf_z_loop(md_v, gr_v, z_v,
               tw_tvt, tw_gr, tw_s,
               pos, vel, w,
               gs, beta, icpt, zsig,
               PF_MOM, PF_VN, PF_PN, tmin, tmax,
               PF_GR_WT, PF_RESAMP, PF_ROUGH_P, PF_ROUGH_V,
               gr_sm_v):
    N   = len(pos)
    n_e = len(md_v)
    pts = np.empty(n_e, np.float32)
    std = np.empty(n_e, np.float32)
    pm  = md_v[0]
    for i in range(n_e):
        dm  = max(md_v[i] - pm, 1.0)
        dzd = (z_v[i] - (z_v[i - 1] if i > 0 else z_v[0])) / dm

        noise_v = np.random.normal(0.0, PF_VN, N)
        noise_p = np.random.normal(0.0, PF_PN, N)
        _lo = tmin - 50.0; _hi = tmax + 50.0
        for j in range(N):
            vel[j] = PF_MOM * vel[j] + noise_v[j]
            _raw   = pos[j] + vel[j] * dm + noise_p[j]
            pos[j] = min(max(_raw, _lo), _hi)

        gv = gr_v[i]
        if not np.isnan(gv):
            for j in range(N):
                ep = _numba_interp1(tw_tvt, tw_gr, pos[j])
                lp = np.exp(-0.5 * ((gv - ep) / gs) ** 2)
                gs_sm_j = gr_sm_v[i] if not np.isnan(gr_sm_v[i]) else np.nan
                if not np.isnan(gs_sm_j):
                    es_ = _numba_interp1(tw_tvt, tw_s, pos[j])
                    ls  = np.exp(-0.5 * ((gs_sm_j - es_) / (gs * 1.5)) ** 2)
                    lk  = (1.0 - PF_GR_WT) * lp + PF_GR_WT * ls
                else:
                    lk = lp
                w[j] *= max(lk, 1e-300)
            ws = np.sum(w)
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1.0 / N

        zs = max(zsig * 2.0, 0.005)
        for j in range(N):
            ve = beta * dzd + icpt
            lz = max(np.exp(-0.5 * ((vel[j] - ve) / zs) ** 2), 1e-300)
            w[j] *= lz
        ws = np.sum(w)
        if ws > 0.0:
            for j in range(N): w[j] /= ws
        else:
            for j in range(N): w[j] = 1.0 / N

        ne = 1.0 / np.sum(w * w)
        if ne < PF_RESAMP * N:
            cum = np.cumsum(w)
            u_  = (np.arange(N) + np.random.uniform()) / N
            ix  = np.searchsorted(cum, u_)
            new_pos = pos[ix].copy(); new_vel = vel[ix].copy()
            noise_rp = np.random.normal(0.0, PF_ROUGH_P, N)
            noise_rv = np.random.normal(0.0, PF_ROUGH_V, N)
            for j in range(N):
                pos[j] = new_pos[j] + noise_rp[j]
                vel[j] = new_vel[j] + noise_rv[j]
                w[j]   = 1.0 / N

        mu_ = 0.0
        for j in range(N): mu_ += w[j] * pos[j]
        var_ = 0.0
        for j in range(N): var_ += w[j] * (pos[j] - mu_) ** 2
        pts[i] = np.float32(mu_)
        std[i] = np.float32(np.sqrt(var_))
        pm = md_v[i]

    return pts, std


@_njit(cache=True)
def _pf_ancc_loop(md_v, gr_v, z_v,
                  tw_tvt, tw_gr,
                  pos, rate, w, gs,
                  ANCC_ALPHA, ANCC_RN, ANCC_PN,
                  tmin, tmax, PF_RESAMP,
                  ANCC_RP, ANCC_RR):
    N   = len(pos)
    n_e = len(md_v)
    pts = np.empty(n_e, np.float32)
    std = np.empty(n_e, np.float32)
    pm  = md_v[0]
    for i in range(n_e):
        dm = max(md_v[i] - pm, 1.0)
        noise_r = np.random.normal(0.0, ANCC_RN, N)
        noise_p = np.random.normal(0.0, ANCC_PN, N)
        for j in range(N):
            rate[j] = ANCC_ALPHA * rate[j] + noise_r[j]
            pos[j] += rate[j] * dm + noise_p[j]
        _lo = tmin - 50.0; _hi = tmax + 50.0
        tvt_e = np.empty(N, np.float64)
        _zvi  = z_v[i]
        for j in range(N):
            _raw    = pos[j] - _zvi
            tvt_e[j] = min(max(_raw, _lo), _hi)
        for j in range(N): pos[j] = tvt_e[j] + _zvi

        gv = gr_v[i]
        if not np.isnan(gv):
            for j in range(N):
                eg  = _numba_interp1(tw_tvt, tw_gr, tvt_e[j])
                lk  = max(np.exp(-0.5 * ((gv - eg) / gs) ** 2), 1e-300)
                w[j] *= lk
            ws = np.sum(w)
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1.0 / N

        ne = 1.0 / np.sum(w * w)
        if ne < PF_RESAMP * N:
            cum = np.cumsum(w)
            u_  = (np.arange(N) + np.random.uniform()) / N
            ix  = np.searchsorted(cum, u_)
            new_pos  = pos[ix].copy(); new_rate = rate[ix].copy()
            noise_rp = np.random.normal(0.0, ANCC_RP, N)
            noise_rr = np.random.normal(0.0, ANCC_RR, N)
            for j in range(N):
                pos[j]  = new_pos[j]  + noise_rp[j]
                rate[j] = new_rate[j] + noise_rr[j]
                w[j]    = 1.0 / N

        tvt_e2 = pos - z_v[i]
        mu_ = 0.0
        for j in range(N): mu_ += w[j] * tvt_e2[j]
        var_ = 0.0
        for j in range(N): var_ += w[j] * (tvt_e2[j] - mu_) ** 2
        pts[i] = np.float32(mu_)
        std[i] = np.float32(np.sqrt(var_))
        pm = md_v[i]

    return pts, std


def run_pf_z(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,
             N: int = PF_N):
    tw_s = (pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1)
            .mean().to_numpy(np.float32))
    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
    gs = _cal_gr_sigma(hw, tw_tvt, tw_gr)
    beta, icpt, zsig = _z_beta(hw)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([], np.float32), np.array([], np.float32)

    gr_sm_full = (hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1)
                  .mean().to_numpy(np.float32))
    ev_locs   = [hw.index.get_loc(idx) for idx in ev.index]
    gr_sm_ev  = gr_sm_full[ev_locs].astype(np.float32)

    pos  = float(kn["TVT_input"].iloc[-1]) + np.random.normal(0, PF_INIT_SPR, N).astype(np.float32)
    vel  = (_init_v(hw) + np.random.normal(0, PF_INIT_V_STD, N)).astype(np.float32)
    w    = np.full(N, 1.0 / N, np.float64)

    md_v  = ev["MD"].to_numpy(np.float32)
    gr_v  = ev["GR"].to_numpy(np.float32)
    z_v   = ev["Z"].to_numpy(np.float32)

    if _NUMBA:
        pts, std = _pf_z_loop(
            md_v, gr_v, z_v,
            tw_tvt.astype(np.float64), tw_gr.astype(np.float64),
            tw_s.astype(np.float64), pos.astype(np.float64), vel.astype(np.float64),
            w,
            float(gs), float(beta), float(icpt), float(zsig),
            float(PF_MOM), float(PF_VN), float(PF_PN),
            float(tmin), float(tmax),
            float(PF_GR_WT), float(PF_RESAMP), float(PF_ROUGH_P), float(PF_ROUGH_V),
            gr_sm_ev.astype(np.float64),
        )
    else:
        tf_p  = interp1d(tw_tvt, tw_gr, bounds_error=False,
                         fill_value=(tw_gr[0], tw_gr[-1]))
        tf_s  = interp1d(tw_tvt, tw_s,  bounds_error=False,
                         fill_value=(tw_s[0],  tw_s[-1]))
        pts_l = np.empty(len(ev)); std_l = np.empty(len(ev))
        pm_py = float(kn["MD"].iloc[-1]); pz = float(kn["Z"].iloc[-1])
        for i, idx in enumerate(ev.index):
            dm  = max(md_v[i] - pm_py, 1.0)
            dzd = (z_v[i] - pz) / dm
            vel = PF_MOM * vel + np.random.normal(0, PF_VN, N)
            pos = pos + vel * dm + np.random.normal(0, PF_PN, N)
            pos = np.clip(pos, tmin - 50, tmax + 50)
            if not np.isnan(gr_v[i]):
                ep = tf_p(pos)
                lp = np.exp(-0.5 * ((gr_v[i] - ep) / gs) ** 2)
                gs_sm = gr_sm_ev[i]
                if not np.isnan(gs_sm):
                    es_ = tf_s(pos)
                    ls  = np.exp(-0.5 * ((gs_sm - es_) / (gs * 1.5)) ** 2)
                    lk  = (1 - PF_GR_WT) * lp + PF_GR_WT * ls
                else:
                    lk = lp
                lk = np.maximum(lk, 1e-300); w *= lk
                ws = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)
            ve = beta * dzd + icpt; zs = max(zsig * 2.0, 0.005)
            lz = np.exp(-0.5 * ((vel - ve) / zs) ** 2)
            lz = np.maximum(lz, 1e-300); w *= lz
            ws = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)
            ne = 1.0 / np.sum(w ** 2)
            if ne < PF_RESAMP * N:
                cum = np.cumsum(w); u = (np.arange(N) + np.random.uniform()) / N
                ix  = np.searchsorted(cum, u)
                pos = pos[ix]; vel = vel[ix]; w[:] = 1.0 / N
                pos += np.random.normal(0, PF_ROUGH_P, N)
                vel += np.random.normal(0, PF_ROUGH_V, N)
            pts_l[i] = np.average(pos, weights=w)
            std_l[i] = np.sqrt(np.average((pos - pts_l[i]) ** 2, weights=w))
            pm_py = md_v[i]; pz = z_v[i]
        pts, std = pts_l, std_l

    return pts.astype(np.float32), std.astype(np.float32)


def run_pf_ancc(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                N: int = ANCC_N):
    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
    gs  = _cal_gr_sigma(hw, tw_tvt, tw_gr)
    kn  = hw[hw["TVT_input"].notna()]
    ev  = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([], np.float32), np.array([], np.float32)

    ls  = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt   = np.diff(tail["TVT_input"].values)
    dz   = np.diff(tail["Z"].values)
    dm   = np.diff(tail["MD"].values)
    m    = dm > 0
    ir   = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    pos  = (ls + np.random.normal(0, ANCC_IS, N)).astype(np.float64)
    rate = (ir  + np.random.normal(0, ANCC_IR, N)).astype(np.float64)
    w    = np.full(N, 1.0 / N, np.float64)

    md_v = ev["MD"].to_numpy(np.float32)
    z_v  = ev["Z"].to_numpy(np.float32)
    gr_v = ev["GR"].to_numpy(np.float32)

    if _NUMBA:
        pts, std = _pf_ancc_loop(
            md_v.astype(np.float64), gr_v.astype(np.float64),
            z_v.astype(np.float64),
            tw_tvt.astype(np.float64), tw_gr.astype(np.float64),
            pos, rate, w, float(gs),
            float(ANCC_ALPHA), float(ANCC_RN), float(ANCC_PN),
            float(tmin), float(tmax), float(PF_RESAMP),
            float(ANCC_RP), float(ANCC_RR),
        )
    else:
        pm    = float(kn["MD"].iloc[-1])
        pts_l = np.empty(len(ev)); std_l = np.empty(len(ev))
        for i in range(len(ev)):
            dm_s = max(md_v[i] - pm, 1.0)
            rate = ANCC_ALPHA * rate + np.random.normal(0, ANCC_RN, N)
            pos  = pos + rate * dm_s + np.random.normal(0, ANCC_PN, N)
            tvt_e = np.clip(pos - z_v[i], tmin - 50, tmax + 50)
            pos   = tvt_e + z_v[i]
            if not np.isnan(gr_v[i]):
                eg  = np.interp(tvt_e, tw_tvt, tw_gr)
                lk  = np.exp(-0.5 * ((gr_v[i] - eg) / gs) ** 2)
                lk  = np.maximum(lk, 1e-300); w *= lk
                ws  = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)
            ne = 1.0 / np.sum(w ** 2)
            if ne < PF_RESAMP * N:
                cum = np.cumsum(w); u = (np.arange(N) + np.random.uniform()) / N
                ix  = np.searchsorted(cum, u)
                pos = pos[ix]; rate = rate[ix]; w[:] = 1.0 / N
                pos  += np.random.normal(0, ANCC_RP, N)
                rate += np.random.normal(0, ANCC_RR, N)
            tv = float(np.average(pos - z_v[i], weights=w))
            pts_l[i] = tv
            std_l[i] = np.sqrt(np.average((pos - z_v[i] - tv) ** 2, weights=w))
            pm = md_v[i]
        pts, std = pts_l, std_l

    return pts.astype(np.float32), std.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Imputers
# ═══════════════════════════════════════════════════════════════════════════════

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir: Path):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid,
                   "x": float(df["X"].median()),
                   "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)

        self.df   = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy        = self.df[["x", "y"]].to_numpy(np.float64)
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree  = cKDTree(xy / self.scale)
        self.xa    = self.df["x"].to_numpy(np.float64)
        self.ya    = self.df["y"].to_numpy(np.float64)
        self.fa    = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
        self._fa_mean = self.fa.mean(0)

    def impute(self, xy_q: np.ndarray, self_wid=None, k: int = PLANE_K):
        q    = xy_q / self.scale
        nf   = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)

        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)

        if nf > k:
            ord_ = np.argpartition(dist, k - 1, axis=1)[:, :k]
            dk   = np.take_along_axis(dist, ord_, axis=1)
            ik   = np.take_along_axis(idx,  ord_, axis=1)
        else:
            dk, ik = dist, idx

        vk  = np.isfinite(dk)
        w   = np.where(vk, 1.0 / (dk + 1e-3), 0.0)

        xn = self.xa[ik]; yn = self.ya[ik]
        wx = w * xn;      wy = w * yn

        A = np.zeros((len(q), 3, 3), np.float64)
        A[:, 0, 0] = (wx * xn).sum(1);  A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1];        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2];        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9

        fn  = self.fa[ik]
        rhs = np.stack([
            (wx[:, :, None] * fn).sum(1),
            (wy[:, :, None] * fn).sum(1),
            (w[:, :, None]  * fn).sum(1),
        ], axis=1)

        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass

        Xq   = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :]
                + Yq[:, None] * coef[:, 1, :]
                + coef[:, 2, :]).astype(np.float32)

        no_nbr = ~vk.any(1)
        pred[no_nbr] = self._fa_mean.astype(np.float32)

        min_dist = np.where(vk, dk, np.inf).min(1).astype(np.float32)
        return pred, min_dist


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir: Path, spw: int = DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s  = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids.extend([wid] * len(s))

        self.xy   = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree  = cKDTree(self.xy / self.scale)
        self._mean = float(self.ancc.mean())

    def impute(self, xy_q: np.ndarray, self_wid=None,
               k: int = DENSE_K, nfetch: int = 500):
        xy_q = np.atleast_2d(xy_q)
        q    = xy_q / self.scale
        nf   = min(nfetch, len(self.ancc))

        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)

        if nf > k:
            ord_ = np.argpartition(dist, k - 1, axis=1)[:, :k]
            dk   = np.take_along_axis(dist, ord_, axis=1)
            ik   = np.take_along_axis(idx,  ord_, axis=1)
        else:
            dk, ik = dist, idx

        vk  = np.isfinite(dk)
        w   = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw  = w.sum(1); safe = np.where(sw < 1e-9, 1.0, sw)

        an  = self.ancc[ik]
        ap  = (an * w).sum(1) / safe
        ap  = np.where(sw < 1e-9, self._mean, ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe

        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Builder — split into I/O wrapper + pure compute core
# ═══════════════════════════════════════════════════════════════════════════════

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80],  dtype=np.float32)
BEAM_OFFS = np.array([-40, -20, -10,  -5, -3, 0, 3,  5, 10, 20, 40],  dtype=np.float32)
SC_OFFS   = np.array([-30, -15,  -8,  -4, -2, 0, 2,  4,  8, 15, 30],  dtype=np.float32)

# Process-global imputers — set once per worker via pool initializer
_FI: Optional[FormationPlaneKNN] = None
_DI: Optional[DenseANCCImputer]  = None


def _worker_init(fi: FormationPlaneKNN, di: DenseANCCImputer) -> None:
    """Called once per worker process; stores imputers in process-local globals."""
    global _FI, _DI
    _FI = fi
    _DI = di


USE_PROCESS_POOL = env_flag("ROGII_USE_PROCESS_POOL", os.name != "nt")
USE_THREAD_POOL  = env_flag("ROGII_USE_THREAD_POOL", os.name == "nt")


def _map_with_imputers(func, items, processes):
    """Use process pools on Kaggle/Linux and thread pools on Windows local runs."""
    if USE_PROCESS_POOL and processes > 1:
        with multiprocessing.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(FI, DI),
        ) as pool:
            return pool.map(func, items)
    _worker_init(FI, DI)
    if USE_THREAD_POOL and processes > 1:
        with ThreadPoolExecutor(max_workers=processes) as pool:
            return list(pool.map(func, items))
    return [func(item) for item in items]


def _starmap_with_imputers(func, args, processes):
    """Use process pools on Kaggle/Linux and thread pools on Windows local runs."""
    if USE_PROCESS_POOL and processes > 1:
        with multiprocessing.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(FI, DI),
        ) as pool:
            return pool.starmap(func, args)
    _worker_init(FI, DI)
    if USE_THREAD_POOL and processes > 1:
        with ThreadPoolExecutor(max_workers=processes) as pool:
            return list(pool.map(lambda arg: func(*arg), args))
    return [func(*arg) for arg in args]


# ── Pre-computed GR stats cache (per well, shared across aug splits) ──────────

class _WellGRCache:
    """
    Holds expensive per-well arrays that don't change across augmentation splits
    (the full GR array, rolling stats, envelope, energy).
    Constructed once in process_train_well and passed into build_augmented.
    """
    __slots__ = ("gr_arr", "roll_feats_full", "gr_env_full", "gr_nrg_full")

    def __init__(self, hw: pd.DataFrame, gr_mean: float):
        gr_full = (hw["GR"].astype(float)
                   .interpolate(limit_direction="both")
                   .fillna(gr_mean))
        self.gr_arr         = gr_full.to_numpy(np.float32)
        # Compute rolling stats over the FULL well once
        all_idx             = np.arange(len(hw), dtype=np.int64)
        self.roll_feats_full = _build_gr_rolls(self.gr_arr, all_idx)
        self.gr_env_full     = gr_envelope(self.gr_arr)
        self.gr_nrg_full     = gr_energy(self.gr_arr)


def _build_well_from_df(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    is_train: bool,
    wid: str,
    gr_cache: Optional[_WellGRCache] = None,
) -> Optional[pd.DataFrame]:
    """
    Pure-compute feature builder. No disk I/O.
    gr_cache: if provided, reuses pre-computed GR rolling stats (avoids recompute
              across augmentation splits of the same well).
    """
    if _FI is None or _DI is None:
        _dbg(f"_build_well_from_df({wid}): imputers not set!", level="ERROR")
        return None

    required_hw = {"MD", "GR", "X", "Y", "Z", "TVT_input"}
    required_tw = {"TVT", "GR"}
    missing_hw  = required_hw - set(hw.columns)
    missing_tw  = required_tw - set(tw.columns)
    if missing_hw:
        _dbg(f"_build_well_from_df({wid}): missing hw cols {missing_hw}", level="WARN")
        return None
    if missing_tw:
        _dbg(f"_build_well_from_df({wid}): missing tw cols {missing_tw}", level="WARN")
        return None

    if is_train and "TVT" not in hw.columns:
        _dbg(f"_build_well_from_df({wid}): is_train but no TVT column", level="WARN")
        return None

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        _dbg(f"_build_well_from_df({wid}): no eval rows", level="WARN")
        return None
    if len(kn) < 10:
        _dbg(f"_build_well_from_df({wid}): too few known rows ({len(kn)})", level="WARN")
        return None
    if is_train and hw["TVT"].isna().all():
        _dbg(f"_build_well_from_df({wid}): is_train but all TVT NaN", level="WARN")
        return None

    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr  = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        _dbg(f"_build_well_from_df({wid}): typewell too short", level="WARN")
        return None

    try:
        lk       = kn.iloc[-1]
        last_tvt = float(lk["TVT_input"])
        gr_mean  = float(np.nanmean(tw_gr))

        # ── GR arrays — use cache if available, else compute ──────────────
        if gr_cache is not None:
            gr_arr = gr_cache.gr_arr
        else:
            gr_full = (hw["GR"].astype(float)
                       .interpolate(limit_direction="both")
                       .fillna(gr_mean))
            gr_arr = gr_full.to_numpy(np.float32)

        # iloc positions of eval rows (needed for rolling feature slicing)
        ev_idx_arr = np.array([hw.index.get_loc(i) for i in ev.index], dtype=np.int64)

        hgr = gr_arr[ev_idx_arr]
        kgr = gr_arr[:len(kn)]

        # ── GR rolling features ───────────────────────────────────────────
        if gr_cache is not None:
            # Slice pre-computed full-well rolling arrays to eval positions
            roll_feats = {
                k: v[ev_idx_arr]
                for k, v in gr_cache.roll_feats_full.items()
            }
            hgr_env = gr_cache.gr_env_full[ev_idx_arr]
            hgr_nrg = gr_cache.gr_nrg_full[ev_idx_arr]
        else:
            roll_feats = _build_gr_rolls(gr_arr, ev_idx_arr)
            hgr_env    = gr_envelope(gr_arr)[ev_idx_arr]
            hgr_nrg    = gr_energy(gr_arr)[ev_idx_arr]

        gr_d1 = roll_feats.pop("gr_d1")
        gr_d2 = roll_feats.pop("gr_d2")

        # Particle filters
        pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
        pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
        if len(pf_a) == 0:
            _dbg(f"_build_well_from_df({wid}): pf_ancc returned empty", level="WARN")
            return None

        pf_use  = pf_a; std_use = std_a
        has_z   = (len(pf_z) == len(pf_a)) and not np.any(np.isnan(pf_z))

        # Beam search (7 configs)
        # eval_start_idx: first iloc position of the eval block
        eval_start_iloc = int(ev_idx_arr[0])
        gr_filled_series = pd.Series(gr_arr)
        bpaths    = run_all_beams(hw, tw_tvt, tw_gr, last_tvt,
                                  gr_filled_series, eval_start_iloc)
        beam_vals = np.stack(list(bpaths.values()), axis=1)
        beam_ref  = (bpaths["cons"] + bpaths["sm5"]) * 0.5

        # Multi-scale self-correlation
        ktvt = kn["TVT_input"].to_numpy(np.float32)
        sc_results  = multi_scale_sc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
        sc8,  sc8s  = sc_results[0]
        sc15, sc15s = sc_results[1]
        sc25, sc25s = sc_results[2]
        sc_consensus = ((sc8 + sc15 + sc25) * (1.0 / 3.0)).astype(np.float32)
        sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
        hyb_ref  = ((1 - sc_trust) * beam_ref + sc_trust * sc15).astype(np.float32)

        # Affine calibration
        tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
        a_cal, b_cal = affine_cal(kgr, tw_at_k)

        # Prefix statistics
        kmd      = kn["MD"].to_numpy(np.float32)
        kz       = kn["Z"].to_numpy(np.float32)
        pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
        slp_all  = robust_slope(kmd,       ktvt)
        slp_50   = robust_slope(kmd[-50:], ktvt[-50:])
        slp_z    = robust_slope(kz,        ktvt)

        # Spatial imputers
        swid    = wid if is_train else None
        xy_ev   = ev[["X", "Y"]].to_numpy(np.float64)
        xy_kn   = kn[["X", "Y"]].to_numpy(np.float64)
        form_ev, knn_d  = _FI.impute(xy_ev, self_wid=swid)
        form_kn, _      = _FI.impute(xy_kn, self_wid=swid)
        z_kn    = kn["Z"].to_numpy(np.float32)
        z_ev    = ev["Z"].to_numpy(np.float32)

        # Per-formation TVT formulas + RMSE features
        ktvt_plus_zkn = ktvt + z_kn
        residuals_all = ktvt_plus_zkn[:, None] - form_kn

        b_all_v  = np.median(residuals_all, axis=0)
        b_wls_v  = np.array([wls_b_well(ktvt, z_kn, form_kn[:, fi])
                             for fi in range(6)], dtype=np.float32)
        b_50_v   = (np.median(residuals_all[-50:], axis=0).astype(np.float32)
                    if len(ktvt) >= 5 else b_all_v.astype(np.float32))

        tvt_form_mat  = (-z_ev[:, None] + form_ev + b_all_v[None, :]).astype(np.float32)
        tvt_formw_mat = (-z_ev[:, None] + form_ev + b_wls_v[None, :]).astype(np.float32)

        kn_pred_mat   = (-z_kn[:, None] + form_kn + b_all_v[None, :]).astype(np.float32)
        form_rmse_v   = np.sqrt(np.mean((ktvt[:, None] - kn_pred_mat) ** 2, axis=0))

        form_consistency = tvt_form_mat
        form_mean_d  = (form_consistency.mean(1) - last_tvt).astype(np.float32)
        form_std_d   = form_consistency.std(1).astype(np.float32)
        form_range_d = (form_consistency.max(1) - form_consistency.min(1)).astype(np.float32)

        # Dense ANCC
        d_ancc, d_std, d_dist         = _DI.impute(xy_ev, self_wid=swid)
        d_kn,   d_std_kn, _           = _DI.impute(xy_kn, self_wid=swid)
        res_kn   = ktvt + z_kn - d_kn
        b_d      = float(np.median(res_kn))
        b_dw     = wls_b_well(ktvt, z_kn, d_kn)
        b_d50    = float(np.median(res_kn[-50:])) if len(ktvt) >= 5 else b_d
        tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
        tvt_dw    = (-z_ev + d_ancc + b_dw).astype(np.float32)
        tvt_d50   = (-z_ev + d_ancc + b_d50).astype(np.float32)
        d_rmse    = float(np.sqrt(np.mean(res_kn ** 2)))
        d_bias    = float(np.mean(res_kn))
        d_nb_std  = float(np.mean(d_std_kn))

        # Inter-signal consensus std
        all_sig = [pf_use, tvt_form_mat[:, 0], tvt_dense,
                   *bpaths.values(), sc8, sc15, sc25]
        valid_sig = [s for s in all_sig
                     if len(s) == len(ev) and not np.any(np.isnan(s))]
        signal_std = (np.stack(valid_sig, axis=1).std(1).astype(np.float32)
                      if len(valid_sig) >= 2 else np.zeros(len(ev), np.float32))

        # Slope baselines
        hmd        = ev["MD"].to_numpy(np.float32)
        md_since   = hmd - float(lk["MD"])
        slp_base_all = (last_tvt + slp_all * md_since).astype(np.float32)
        slp_base_50  = (last_tvt + slp_50  * md_since).astype(np.float32)

        nh   = len(ev)
        frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

        def sc(v: float) -> np.ndarray:
            return np.full(nh, np.float32(v), np.float32)

        # Trajectory derivatives
        mdd   = hw["MD"].diff().replace(0, np.nan)
        dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].to_numpy(np.float32)
        dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].to_numpy(np.float32)
        dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].to_numpy(np.float32)

        # Pre-compute interp lookups (all at once, single np.interp calls)
        anch_twgr       = np.interp(float(last_tvt) + ANCH_OFFS, tw_tvt, tw_gr).astype(np.float32)
        beam_ref_tw     = np.interp(beam_ref,  tw_tvt, tw_gr).astype(np.float32)
        sc15_tw         = np.interp(sc15,      tw_tvt, tw_gr).astype(np.float32)

        # Offset lookups — vectorised: shape (len(BEAM_OFFS), nh) → transpose
        beam_offsets_tw = np.array([
            np.interp(beam_ref + o, tw_tvt, tw_gr) for o in BEAM_OFFS
        ], dtype=np.float32).T
        sc_offsets_tw = np.array([
            np.interp(sc15 + o, tw_tvt, tw_gr) for o in SC_OFFS
        ], dtype=np.float32).T

        # ── Assemble feature dict ────────────────────────────────────────
        feats: dict = {
            "well": wid,
            "id":   [f"{wid}_{i}" for i in ev.index],
            "last_known_tvt": sc(last_tvt),
            "pf_ancc":       pf_use,
            "pf_ancc_std":   std_use,
            "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
            "pf_z":          pf_z.astype(np.float32) if has_z else sc(last_tvt),
            "pf_z_delta":    (pf_z - last_tvt).astype(np.float32) if has_z else sc(0.0),
            "pf_vs_z":       (pf_use - pf_z.astype(np.float32)) if has_z else sc(0.0),
            "pf_std_trend":  (std_use - std_use[0]).astype(np.float32) if len(std_use) > 0 else sc(0.0),
            **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32)
               for t, p in bpaths.items()},
            "beam_mean_d": (beam_vals - last_tvt).mean(1).astype(np.float32),
            "beam_std_d":  (beam_vals - last_tvt).std(1).astype(np.float32),
            "beam_med_d":  np.median(beam_vals - last_tvt, axis=1).astype(np.float32),
            "sc8_d":    (sc8  - np.float32(last_tvt)),
            "sc8_score": sc8s,
            "sc15_d":   (sc15 - np.float32(last_tvt)),
            "sc15_score": sc15s,
            "sc25_d":   (sc25 - np.float32(last_tvt)),
            "sc25_score": sc25s,
            "sc_cons_d": (sc_consensus - np.float32(last_tvt)),
            "sc_trust":  sc(sc_trust),
            "hyb_d":     (hyb_ref - np.float32(last_tvt)).astype(np.float32),
            **{fn:                    tvt_form_mat[:, fi]  for fi, fn in enumerate(FORMATIONS)},
            **{fn + "_wls":           tvt_formw_mat[:, fi] for fi, fn in enumerate(FORMATIONS)},
            **{f"b_{fn}":             sc(float(b_all_v[fi]))  for fi, fn in enumerate(FORMATIONS)},
            **{f"bw_{fn}":            sc(float(b_wls_v[fi]))  for fi, fn in enumerate(FORMATIONS)},
            **{f"b50_{fn}":           sc(float(b_50_v[fi]))   for fi, fn in enumerate(FORMATIONS)},
            **{f"tvtF_{fn}_d":       (tvt_form_mat[:,  fi] - last_tvt).astype(np.float32)
               for fi, fn in enumerate(FORMATIONS)},
            **{f"tvtFw_{fn}_d":      (tvt_formw_mat[:, fi] - last_tvt).astype(np.float32)
               for fi, fn in enumerate(FORMATIONS)},
            **{f"form_rmse_{fn}":    sc(float(form_rmse_v[fi]))
               for fi, fn in enumerate(FORMATIONS)},
            "form_mean_d":   form_mean_d,
            "form_std_d":    form_std_d,
            "form_range_d":  form_range_d,
            "spatial_knn_dist": knn_d,
            "dense_ancc":    d_ancc,
            "dense_std":     d_std,
            "dense_dist":    d_dist,
            "tvt_dense_d":   (tvt_dense - last_tvt).astype(np.float32),
            "tvt_dw_d":      (tvt_dw    - last_tvt).astype(np.float32),
            "tvt_d50_d":     (tvt_d50   - last_tvt).astype(np.float32),
            "dense_rmse":    sc(d_rmse),
            "dense_bias":    sc(d_bias),
            "dense_nb_std":  sc(d_nb_std),
            "pf_vs_spatial":      (pf_use - tvt_form_mat[:, 0]).astype(np.float32),
            "pf_vs_dense":        (pf_use - tvt_dense).astype(np.float32),
            "spatial_vs_dense":   (tvt_form_mat[:, 0] - tvt_dense).astype(np.float32),
            "beam_vs_spatial":    (bpaths["cons"] - tvt_form_mat[:, 0]).astype(np.float32),
            "sc15_vs_beam_cons":  (sc15 - bpaths["cons"]).astype(np.float32),
            "signal_std":  signal_std,
            "cal_a": sc(a_cal), "cal_b": sc(b_cal),
            "pfx_rmse":   sc(pfx_rmse), "known_len": sc(len(kn)), "eval_len": sc(nh),
            "slp_all":    sc(slp_all),  "slp_50":    sc(slp_50),  "slp_z":    sc(slp_z),
            "slp_base_d_all": (slp_base_all - last_tvt).astype(np.float32),
            "slp_base_d_50":  (slp_base_50  - last_tvt).astype(np.float32),
            "ktvt_range": sc(float(np.ptp(ktvt))), "ktvt_std": sc(float(ktvt.std())),
            "md_since": md_since, "frac": frac, "frac2": frac ** 2,
            "sqrt_frac": np.sqrt(frac),
            "z":  z_ev,
            "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
            "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
            "dz": (z_ev    - float(lk["Z"])).astype(np.float32),
            "dxy": np.sqrt(
                (ev["X"] - float(lk["X"])) ** 2 +
                (ev["Y"] - float(lk["Y"])) ** 2
            ).to_numpy(np.float32),
            "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,
            "gr":       hgr,
            "gr_d1":    gr_d1,
            "gr_d2":    gr_d2,
            "gr_env":   hgr_env.astype(np.float32),
            "gr_energy": hgr_nrg.astype(np.float32),
            "gr_vs_tw_anc":  hgr - float(np.interp(last_tvt, tw_tvt, tw_gr)),
            "gr_vs_slp_all": hgr - np.interp(slp_base_all, tw_tvt, tw_gr).astype(np.float32),
            **{f"tda{int(o)}": hgr - anch_twgr[i] for i, o in enumerate(ANCH_OFFS)},
            **{f"tdbc{int(o)}": hgr - beam_offsets_tw[:, i]
               for i, o in enumerate(BEAM_OFFS)},
            **{f"tdsc{int(o)}": hgr - sc_offsets_tw[:, i]
               for i, o in enumerate(SC_OFFS)},
            "tw_range":   sc(float(np.ptp(tw_tvt))),
            "tw_gr_mean": sc(float(tw_gr.mean())),
        }
        feats.update(roll_feats)

        result = pd.DataFrame(feats)

        if DBG_VERBOSE:
            _dbg(f"_build_well_from_df({wid}): rows={len(result)} cols={len(result.columns)}")

        if is_train:
            if "TVT" not in ev.columns or ev["TVT"].isna().all():
                _dbg(f"_build_well_from_df({wid}): is_train but ev TVT all NaN", level="WARN")
                return None
            result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))

        return result

    except Exception as exc:
        _log_error(f"_build_well_from_df({wid})", exc)
        return None


def build_well(hw_path: str, tw_path: str, is_train: bool) -> Optional[pd.DataFrame]:
    """I/O wrapper — reads CSVs then delegates to _build_well_from_df."""
    if _FI is None or _DI is None:
        _dbg(f"build_well({Path(hw_path).stem}): imputers not set!", level="ERROR")
        return None
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception as exc:
        _log_error(f"build_well({wid}) read_csv", exc)
        return None
    return _build_well_from_df(hw, tw, is_train=is_train, wid=wid)


# ═══════════════════════════════════════════════════════════════════════════════
# Cal-Zone Augmentation — no temp files, reuses GR cache
# ═══════════════════════════════════════════════════════════════════════════════

def build_augmented(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    wid: str,
    gr_cache: Optional[_WellGRCache] = None,
    n_splits: int = N_AUG_SPLITS,
    min_known: int = MIN_KNOWN_FOR_AUG,
) -> pd.DataFrame:
    known_all = hw[hw["TVT_input"].notna()]
    n_cal = len(known_all)
    if n_cal < min_known + 5:
        _dbg(f"build_augmented({wid}): too few known rows ({n_cal})", level="WARN")
        return pd.DataFrame()

    split_ks = np.unique(np.linspace(min_known, n_cal - 2, n_splits).astype(int))
    tw_mock  = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    parts    = []

    for k in split_ks:
        hw_m = hw.copy()
        mask_start = known_all.index[k]
        hw_m.loc[mask_start:, "TVT_input"] = np.nan

        kn_m  = hw_m[hw_m["TVT_input"].notna()]
        ev_m  = hw_m[hw_m["TVT_input"].isna()]
        n_new = n_cal - k

        if len(ev_m) == 0 or len(kn_m) < 10 or n_new == 0:
            _dbg(f"build_augmented({wid}) k={k}: skipping", level="DEBUG")
            continue

        try:
            # ── Direct in-memory call — no temp files ──────────────────────
            feat = _build_well_from_df(
                hw_m, tw_mock, is_train=True, wid=wid, gr_cache=gr_cache
            )
        except Exception as exc:
            _log_error(f"build_augmented({wid}) k={k}", exc)
            continue

        if feat is None or len(feat) == 0:
            _dbg(f"build_augmented({wid}) k={k}: build returned empty", level="DEBUG")
            continue

        feat_new = feat.iloc[:n_new].copy()
        feat_new["aug_k"] = np.int32(k)
        parts.append(feat_new)
        _dbg(f"build_augmented({wid}) k={k}: added {len(feat_new)} aug rows", level="DEBUG")

    if not parts:
        _dbg(f"build_augmented({wid}): no augmented parts produced", level="WARN")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def process_train_well(hw_path: Path) -> pd.DataFrame:
    """
    Top-level per-well train worker.
    Builds GR cache once, passes it to both the original call and all aug splits.
    """
    wid     = hw_path.stem.replace("__horizontal_well", "")
    tw_path = TRAIN_DIR / f"{wid}__typewell.csv"
    if not tw_path.exists():
        _dbg(f"process_train_well({wid}): typewell not found", level="WARN")
        return pd.DataFrame()
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(np.float32)
        tw_gr  = tw["GR"].to_numpy(np.float32)

        # Build GR cache once — shared across original + all aug splits
        gr_mean  = float(np.nanmean(tw_gr))
        gr_cache = _WellGRCache(hw, gr_mean)

        # Original (full cal-zone) features
        feat_orig = _build_well_from_df(
            hw, tw, is_train=True, wid=wid, gr_cache=gr_cache
        )
        parts: list = []
        if feat_orig is not None and len(feat_orig) > 0:
            feat_orig["aug_k"] = np.int32(-1)
            parts.append(feat_orig)
        else:
            _dbg(f"process_train_well({wid}): build_well returned None/empty", level="WARN")

        # Augmented splits — pass same gr_cache to avoid recomputation
        feat_aug = build_augmented(hw, tw_tvt, tw_gr, wid, gr_cache=gr_cache)
        if len(feat_aug) > 0:
            parts.append(feat_aug)

        if not parts:
            _dbg(f"process_train_well({wid}): returning empty", level="WARN")
            return pd.DataFrame()

        r = pd.concat(parts, ignore_index=True)
        r["well_id"] = wid
        _dbg(f"process_train_well({wid}): done rows={len(r)}", level="DEBUG")
        return r
    except Exception as exc:
        _log_error(f"process_train_well({wid})", exc)
        return pd.DataFrame()


def process_test_train(hw_path: Path) -> pd.DataFrame:
    """Online training: augment from test well calibration zone."""
    wid     = hw_path.stem.replace("__horizontal_well", "")
    tw_path = TEST_DIR / f"{wid}__typewell.csv"
    if not tw_path.exists():
        _dbg(f"process_test_train({wid}): typewell not found", level="WARN")
        return pd.DataFrame()
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)
        if "TVT" not in tw.columns or "GR" not in tw.columns:
            _dbg(f"process_test_train({wid}): typewell missing TVT/GR", level="WARN")
            return pd.DataFrame()
        known = hw[hw["TVT_input"].notna()]
        if len(known) < MIN_KNOWN_FOR_AUG + 5:
            _dbg(f"process_test_train({wid}): too few known rows", level="WARN")
            return pd.DataFrame()
        hw_aug = hw.copy()
        hw_aug["TVT"] = hw_aug["TVT_input"]
        tw_tvt = tw["TVT"].to_numpy(np.float32)
        tw_gr  = tw["GR"].to_numpy(np.float32)

        # Build GR cache for test well augmentation
        gr_mean  = float(np.nanmean(tw_gr))
        gr_cache = _WellGRCache(hw_aug, gr_mean)

        feat_aug = build_augmented(hw_aug, tw_tvt, tw_gr, wid, gr_cache=gr_cache)
        if len(feat_aug) > 0:
            feat_aug["well_id"] = wid
            _dbg(f"process_test_train({wid}): aug rows={len(feat_aug)}", level="DEBUG")
        else:
            _dbg(f"process_test_train({wid}): no aug rows", level="WARN")
        return feat_aug
    except Exception as exc:
        _log_error(f"process_test_train({wid})", exc)
        return pd.DataFrame()


def build_dataset(paths, is_train: bool, label: str) -> pd.DataFrame:
    args = [
        (str(p),
         str(p.parent / f"{p.stem.replace('__horizontal_well','')}__typewell.csv"),
         is_train)
        for p in paths
        if (p.parent / f"{p.stem.replace('__horizontal_well','')}__typewell.csv").exists()
    ]
    print(f"  {label}: {len(args)} wells | {NCPU} workers | process_pool={USE_PROCESS_POOL}")
    res = _starmap_with_imputers(build_well, args, NCPU)
    parts = [r for r in res if r is not None and len(r) > 0]
    print(f"  {label}: OK={len(parts)} skipped={len(args)-len(parts)}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

hw_paths   = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
if DEBUG_MAX_TRAIN_WELLS > 0:
    print(f"[DEBUG] Limiting train wells to first {DEBUG_MAX_TRAIN_WELLS}", flush=True)
    hw_paths = hw_paths[:DEBUG_MAX_TRAIN_WELLS]
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
print(f"Building imputers from {len(train_wids)} wells...")
t0 = time.time()
FI = FormationPlaneKNN(train_wids, TRAIN_DIR)
DI = DenseANCCImputer(train_wids, TRAIN_DIR)
# Set in main process too (for smoke tests)
_FI = FI; _DI = DI
print(f"  FI: {len(FI.df)} centroids | DI: {len(DI.ancc):,} pts  ({time.time()-t0:.0f}s)")


# ── DEBUG: single-well smoke tests (run in main process, imputers already set) ──

if DBG_SINGLE_WELL and len(hw_paths) > 0:
    _smoke_path = hw_paths[0]
    _smoke_wid  = _smoke_path.stem.replace("__horizontal_well", "")
    _smoke_tw   = TRAIN_DIR / f"{_smoke_wid}__typewell.csv"
    print(f"[SMOKE] Testing train well: {_smoke_wid}", flush=True)
    try:
        _smoke_result = build_well(str(_smoke_path), str(_smoke_tw), is_train=True)
        if _smoke_result is None:
            print("[SMOKE] build_well returned None", flush=True)
        else:
            print(f"[SMOKE] build_well OK: shape={_smoke_result.shape}", flush=True)
            if DBG_FEATURE_AUDIT:
                print(f"[SMOKE] columns ({len(_smoke_result.columns)}):",
                      list(_smoke_result.columns), flush=True)
            nan_check = _smoke_result.isnull().sum()
            nan_check = nan_check[nan_check > 0]
            if len(nan_check):
                print(f"[SMOKE] NaN columns: {dict(nan_check)}", flush=True)
            else:
                print("[SMOKE] No NaN columns.", flush=True)
    except Exception as _e:
        print(f"[SMOKE] EXCEPTION: {_e}", flush=True)
        traceback.print_exc()

test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
if DEBUG_MAX_TEST_WELLS > 0:
    print(f"[DEBUG] Limiting test wells to first {DEBUG_MAX_TEST_WELLS}", flush=True)
    test_paths = test_paths[:DEBUG_MAX_TEST_WELLS]

if DBG_SINGLE_TEST and len(test_paths) > 0:
    _smoke_tp   = test_paths[0]
    _smoke_twid = _smoke_tp.stem.replace("__horizontal_well", "")
    _smoke_twp  = TEST_DIR / f"{_smoke_twid}__typewell.csv"
    print(f"[SMOKE] Testing test well: {_smoke_twid}", flush=True)
    try:
        _smoke_t = build_well(str(_smoke_tp), str(_smoke_twp), is_train=False)
        if _smoke_t is None:
            print("[SMOKE] test build_well returned None", flush=True)
        else:
            print(f"[SMOKE] test build_well OK: shape={_smoke_t.shape}", flush=True)
    except Exception as _e:
        print(f"[SMOKE] test EXCEPTION: {_e}", flush=True)
        traceback.print_exc()


if INFERENCE_ONLY:
    print("\nROGII_INFERENCE_ONLY=1; loading saved artifacts.", flush=True)
    artifact_dir = find_artifact_dir()
    manifest = load_json(artifact_dir / "manifest.json")
    config = load_json(artifact_dir / manifest.get("inference_config", "inference_config.json"))
    feature_cols = load_json(artifact_dir / manifest.get("feature_cols", "feature_cols.json"))
    result_keys = list(config.get("result_keys") or [])
    if not RUN_TABICL and any(str(k).startswith("tabicl") for k in result_keys):
        keep_idx = [i for i, k in enumerate(result_keys) if not str(k).startswith("tabicl")]
        result_keys = [result_keys[i] for i in keep_idx]
        for key in ["stacker", "ridge_coef"]:
            if key == "stacker":
                coef = config.get("stacker", {}).get("coef")
            else:
                coef = config.get(key)
            if coef is None:
                continue
            coef = np.asarray(coef, dtype=np.float32)[keep_idx]
            coef_sum = float(np.sum(np.abs(coef)))
            if coef_sum > 1e-12:
                coef = coef / coef_sum
            if key == "stacker":
                config.setdefault("stacker", {})["coef"] = coef.tolist()
            else:
                config[key] = coef.tolist()
        print(f"ROGII_RUN_TABICL=0; using non-TabICL artifact keys: {result_keys}", flush=True)
    print(f"Artifact dir: {artifact_dir}", flush=True)
    print(f"Artifact result keys: {result_keys}", flush=True)

    test_df = build_dataset(test_paths, is_train=False, label="test")
    if test_df.empty:
        raise RuntimeError("No test features were built.")
    missing_features = [c for c in feature_cols if c not in test_df.columns]
    if missing_features:
        print(f"[WARN] Filling {len(missing_features)} missing artifact features with NaN.", flush=True)
        for col in missing_features:
            test_df[col] = np.nan
    Xt = test_df[feature_cols]
    pred_by_key: dict[str, np.ndarray] = {}

    def _entry_path(entry: dict) -> Path:
        return artifact_dir / str(entry["path"]).replace("/", os.sep)

    lgb_groups: dict[str, list[dict]] = {}
    for entry in manifest.get("lgb", []):
        lgb_groups.setdefault(f"lgb{entry['seed']}", []).append(entry)
    for key, entries in sorted(lgb_groups.items()):
        pred = np.zeros(len(test_df), dtype=np.float32)
        for entry in sorted(entries, key=lambda e: e.get("fold", 0)):
            booster = lgb.Booster(model_file=str(_entry_path(entry)))
            best_iter = int(entry.get("best_iteration") or booster.best_iteration or booster.current_iteration())
            pred += booster.predict(Xt, num_iteration=best_iter).astype(np.float32) / len(entries)
        pred_by_key[key] = pred
        print(f"Loaded {len(entries)} LGB folds for {key}", flush=True)

    cb_groups: dict[str, list[dict]] = {}
    for entry in manifest.get("catboost", []):
        cb_groups.setdefault(f"cb{entry['seed']}", []).append(entry)
    for key, entries in sorted(cb_groups.items()):
        pred = np.zeros(len(test_df), dtype=np.float32)
        for entry in sorted(entries, key=lambda e: e.get("fold", 0)):
            model = CatBoostRegressor()
            model.load_model(str(_entry_path(entry)))
            pred += model.predict(Xt.values).astype(np.float32) / len(entries)
        pred_by_key[key] = pred
        print(f"Loaded {len(entries)} CatBoost folds for {key}", flush=True)

    tabicl_needed = any(k.startswith("tabicl") for k in result_keys)
    tabicl_entries = manifest.get("tabicl_contexts", [])
    if tabicl_needed and tabicl_entries:
        if not RUN_TABICL:
            raise RuntimeError("Artifacts expect TabICL predictions, but ROGII_RUN_TABICL=0.")
        import subprocess as _sp
        from pathlib import Path as _Path

        tabicl_roots = [_Path("/kaggle/input")]
        if os.environ.get("ROGII_TABICL_DIR"):
            tabicl_roots.insert(0, _Path(os.environ["ROGII_TABICL_DIR"]))
        _wheels, _ckpts = [], []
        for _root in tabicl_roots:
            if _root.exists():
                _wheels.extend(_root.rglob("tabicl-*.whl"))
                _ckpts.extend(_root.rglob("tabicl-regressor*.ckpt"))
        if not _wheels or not _ckpts:
            raise RuntimeError("TabICL wheel/checkpoint not found for artifact inference.")
        _sp.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(_wheels[0])], check=True)
        from tabicl import TabICLRegressor

        tabicl_features = load_json(artifact_dir / manifest.get("tabicl_features", "tabicl_features.json"))
        Xt_top = Xt[tabicl_features].values
        label_groups: dict[str, list[dict]] = {}
        for entry in tabicl_entries:
            label = str(entry["label"])
            if label.startswith("tabicl_A"):
                key = "tabicl_A"
            elif label.startswith("tabicl_B"):
                key = "tabicl_B"
            else:
                key = label
            label_groups.setdefault(key, []).append(entry)

        device = "cpu" if FORCE_CPU else "cuda"
        chunk = 50_000
        for key, entries in sorted(label_groups.items()):
            pred = np.zeros(len(test_df), dtype=np.float32)
            for entry in sorted(entries, key=lambda e: (e.get("fold", 0), e.get("seed", 0))):
                ctx = np.load(_entry_path(entry))
                reg = TabICLRegressor(
                    model_path=str(_ckpts[0]),
                    device=device,
                    random_state=int(entry.get("seed", 0)),
                    n_estimators=int(entry.get("n_estimators", 4)),
                    n_jobs=1,
                    verbose=False,
                    use_amp="auto",
                    batch_size=4,
                )
                reg.fit(ctx["X_ctx"], ctx["y_ctx"])
                burnin_path = entry.get("burnin_path")
                if burnin_path:
                    burn = np.load(artifact_dir / str(burnin_path).replace("/", os.sep))["X_va"]
                    print(
                        f"   burn {key} fold{entry.get('fold')} seed{entry.get('seed')}: "
                        f"{len(burn)} rows",
                        flush=True,
                    )
                    for i in range(0, len(burn), chunk):
                        e = min(i + chunk, len(burn))
                        _ = reg.predict(burn[i:e])
                    del burn
                cur = np.empty(len(test_df), dtype=np.float32)
                for i in range(0, len(Xt_top), chunk):
                    e = min(i + chunk, len(Xt_top))
                    cur[i:e] = reg.predict(Xt_top[i:e]).astype(np.float32)
                pred += cur / len(entries)
            pred_by_key[key] = pred
            print(f"Loaded {len(entries)} TabICL contexts for {key}", flush=True)

    if not result_keys:
        result_keys = list(pred_by_key.keys())
    missing_results = [k for k in result_keys if k not in pred_by_key]
    if missing_results:
        raise RuntimeError(f"Missing artifact predictions for result keys: {missing_results}")

    St = np.column_stack([pred_by_key[k] for k in result_keys])
    stacker = config.get("stacker", {})
    if stacker.get("coef") is not None:
        coef = np.asarray(stacker["coef"], dtype=np.float32)
        if len(coef) != St.shape[1]:
            raise RuntimeError(f"Stack coef length {len(coef)} does not match predictions {St.shape[1]}.")
        final_test = St @ coef + float(stacker.get("intercept", 0.0))
        print(f"Using saved {stacker.get('kind', 'linear')} stack.", flush=True)
    elif config.get("use_ridge", False):
        coef = np.asarray(config["ridge_coef"], dtype=np.float32)
        if len(coef) != St.shape[1]:
            raise RuntimeError(f"Ridge coef length {len(coef)} does not match predictions {St.shape[1]}.")
        final_test = St @ coef + float(config.get("ridge_intercept", 0.0))
        print("Using saved ridge stack.", flush=True)
    else:
        final_test = St.mean(axis=1)
        print("Using saved simple-average stack.", flush=True)

    alpha = float(config.get("postproc_alpha", 1.0))
    tau = config.get("postproc_tau", None)
    w_pf = float(config.get("postproc_w_pf", 0.0))
    pf_delta = (test_df["pf_ancc"].values - test_df["last_known_tvt"].values).astype(np.float32)
    delta = ((1.0 - w_pf) * final_test.astype(np.float32) + w_pf * pf_delta).astype(np.float32)
    if tau is not None:
        delta *= 1.0 - np.exp(-np.maximum(test_df["md_since"].values, 0.0) / float(tau))
    test_df2 = test_df.copy()
    test_df2["pred"] = test_df2["last_known_tvt"].values + delta * alpha

    sample = pd.read_csv(SAMPLE)
    sub = sample[["id"]].merge(
        test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}),
        on="id", how="left",
    )
    fallback_tvt = float(config.get("fallback_tvt", test_df2["pred"].mean()))
    sub["tvt"] = sub["tvt"].fillna(fallback_tvt)
    print(
        f"[SUB AUDIT] rows={len(sub)} null={int(sub['tvt'].isnull().sum())} "
        f"fallback_filled={int((sub['tvt'] == fallback_tvt).sum())}",
        flush=True,
    )

    if config.get("exact_overlap_enabled", True):
        os.environ["ROGII_EXACT_OVERLAP"] = "1"
        os.environ["ROGII_EXACT_BLEND_WEIGHT"] = str(config.get("exact_blend_weight", 0.28))
    else:
        os.environ["ROGII_EXACT_OVERLAP"] = "0"
    sub = apply_exact_train_coordinate_blend(sub[["id", "tvt"]], DATA)
    sub[["id", "tvt"]].to_csv(OUT, index=False)
    print(f"\n✅  {OUT}  {len(sub)} rows (artifact inference)")
    print(sub.head(8).to_string(index=False))
    print(f"=== ARTIFACT INFERENCE COMPLETE  total={time.time()-_T0_GLOBAL:.0f}s ===")
    raise SystemExit(0)


# ── Build train features — process pool with shared imputers ─────────────────
print("\nLoading/building official train-core features...")
t0 = time.time()

train_core_df = None
if USE_TRAIN_CORE_CACHE and DEBUG_MAX_TRAIN_WELLS <= 0:
    for cache_path in _cache_candidates():
        if cache_path.exists():
            print(f"Loading train-core cache: {cache_path}", flush=True)
            train_core_df = pd.read_pickle(cache_path)
            print(f"  train_core_df={train_core_df.shape} loaded in {time.time()-t0:.0f}s", flush=True)
            break

if train_core_df is None:
    print("\nBuilding official train-core features...")
    res = _map_with_imputers(process_train_well, hw_paths, NCPU)

    if DBG_PARALLEL_STATS:
        n_none  = sum(1 for r in res if r is None)
        n_empty = sum(1 for r in res if r is not None and len(r) == 0)
        n_ok    = sum(1 for r in res if r is not None and len(r) > 0)
        row_dist = [len(r) for r in res if r is not None and len(r) > 0]
        print(f"[PARALLEL STATS train] None={n_none} empty={n_empty} ok={n_ok}", flush=True)
        if row_dist:
            print(f"  rows per well: min={min(row_dist)} max={max(row_dist)} "
                  f"mean={np.mean(row_dist):.0f} total={sum(row_dist)}", flush=True)
        else:
            print("  *** All results None or empty! ***", flush=True)

    parts_core = [r for r in res if r is not None and len(r) > 0]
    if len(parts_core) == 0:
        print("[FATAL] No valid official train DataFrames.", flush=True)
        raise RuntimeError("All process_train_well calls returned None or empty.")

    train_core_df = pd.concat(parts_core, ignore_index=True)
    del res, parts_core
    gc.collect()
    print(f"  train_core_df={train_core_df.shape} built in {time.time()-t0:.0f}s", flush=True)

    if WRITE_TRAIN_CORE_CACHE or CACHE_ONLY:
        cache_out = _train_core_cache_path_for_write()
        cache_out.parent.mkdir(parents=True, exist_ok=True)
        train_core_df.to_pickle(cache_out)
        preview_cols = [c for c in ["well", "id", "last_known_tvt", "target", "aug_k"] if c in train_core_df.columns]
        train_core_df[preview_cols].head(1000).to_csv(
            cache_out.with_name("aeroridge_train_core_preview_1000.csv"), index=False
        )
        schema = pd.DataFrame({
            "column": train_core_df.columns,
            "dtype": [str(train_core_df[c].dtype) for c in train_core_df.columns],
            "non_null_count": [int(train_core_df[c].notna().sum()) for c in train_core_df.columns],
            "null_count": [int(train_core_df[c].isna().sum()) for c in train_core_df.columns],
        })
        schema["null_rate"] = schema["null_count"] / max(len(train_core_df), 1)
        schema.to_csv(cache_out.with_name("aeroridge_train_core_schema.csv"), index=False)
        print(f"  wrote train-core cache: {cache_out}", flush=True)

if CACHE_ONLY:
    print("ROGII_CACHE_ONLY=1, stopping after train-core cache build.", flush=True)
    raise SystemExit(0)

parts = [train_core_df]

# ── Online training on test wells ────────────────────────────────────────────
print(f"\nOnline training on {len(test_paths)} test wells...")
t1 = time.time()

res_t = _map_with_imputers(process_test_train, test_paths, min(NCPU, max(1, len(test_paths))))

if DBG_PARALLEL_STATS:
    n_none_t  = sum(1 for r in res_t if r is None)
    n_empty_t = sum(1 for r in res_t if r is not None and len(r) == 0)
    n_ok_t    = sum(1 for r in res_t if r is not None and len(r) > 0)
    print(f"[PARALLEL STATS test-online] None={n_none_t} empty={n_empty_t} ok={n_ok_t}",
          flush=True)

parts.extend([r for r in res_t if r is not None and len(r) > 0])
del res_t

# ── Pre-concat guard ─────────────────────────────────────────────────────────
if len(parts) == 0:
    print("[FATAL] No valid DataFrames to concatenate.", flush=True)
    raise RuntimeError(
        "All process_train_well / process_test_train calls returned None or empty."
    )

train_df = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
print(f"\ntrain_df: {train_df.shape} | {time.time()-t0:.0f}s")
orig = (train_df["aug_k"] == -1).sum()
aug  = (train_df["aug_k"] >=  0).sum()
print(f"  Original: {orig:,}  Augmented: {aug:,} (+{aug/max(orig,1)*100:.0f}%)")

_dbg_df("train_df", train_df)

SKIP = {"well", "well_id", "id", "target", "aug_k"}
feature_cols = [c for c in train_df.columns if c not in SKIP]
print(f"Features: {len(feature_cols)}")
if SAVE_ARTIFACTS:
    save_json(ARTIFACT_DIR / "feature_cols.json", feature_cols)
    ARTIFACT_MANIFEST["feature_cols"] = "feature_cols.json"
    ARTIFACT_MANIFEST["data"] = {
        "train_df_shape": list(train_df.shape),
        "test_df_shape": None,
        "n_features": len(feature_cols),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "used_train_core_cache": str(os.environ.get("ROGII_TRAIN_CORE_CACHE", "")),
    }

if DBG_FEATURE_AUDIT:
    print(f"[FEATURE AUDIT] {len(feature_cols)} feature columns:", flush=True)
    for _fc in feature_cols:
        _arr = train_df[_fc].to_numpy()
        _nans = int(np.isnan(_arr.astype(float)).sum()) if np.issubdtype(_arr.dtype, np.number) else 0
        if _nans > 0:
            print(f"  NaN: {_fc}  count={_nans}/{len(_arr)}", flush=True)

X  = train_df[feature_cols]
y  = train_df["target"]
g  = train_df["well"]

if DBG_NAN_AUDIT:
    _dbg("NaN audit on X (train features)")
    _nan_X = X.isnull().sum()
    _nan_X = _nan_X[_nan_X > 0]
    if len(_nan_X):
        print(f"[NaN AUDIT] X has NaN in {len(_nan_X)} cols:", flush=True)
        print(_nan_X.to_string(), flush=True)
    else:
        print("[NaN AUDIT] X: clean (no NaN)", flush=True)
    _nan_y = int(y.isnull().sum())
    print(f"[NaN AUDIT] y: {_nan_y} NaN  "
          f"min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f}", flush=True)
    _inf_X = np.isinf(X.to_numpy(dtype=float)).sum()
    print(f"[NaN AUDIT] X Inf count: {_inf_X}", flush=True)

# ── Test features ────────────────────────────────────────────────────────────
test_df = build_dataset(test_paths, is_train=False, label="test")
Xt = test_df[feature_cols]
if SAVE_ARTIFACTS:
    ARTIFACT_MANIFEST.setdefault("data", {})["test_df_shape"] = list(test_df.shape)
    ARTIFACT_MANIFEST["created_outputs"]["test_feature_preview"] = None

if DBG_NAN_AUDIT:
    _nan_Xt = Xt.isnull().sum()
    _nan_Xt = _nan_Xt[_nan_Xt > 0]
    if len(_nan_Xt):
        print(f"[NaN AUDIT] Xt has NaN in {len(_nan_Xt)} cols:", flush=True)
        print(_nan_Xt.to_string(), flush=True)
    else:
        print("[NaN AUDIT] Xt: clean (no NaN)", flush=True)
    _inf_Xt = np.isinf(Xt.to_numpy(dtype=float)).sum()
    print(f"[NaN AUDIT] Xt Inf count: {_inf_Xt}", flush=True)

gc.collect()
_dbg(f"Feature matrix ready: X={X.shape} Xt={Xt.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training:  LGB×3 + CatBoost + Ridge stacking
# ═══════════════════════════════════════════════════════════════════════════════

# ── CV split with group shuffling ────────────────────────────────────────────
# Plain GroupKFold without shuffle assigns folds in sorted order, so wells
# cluster geographically/temporally inside each fold → validation is
# systematically easier or harder → inflated variance and worse generalisation.
# Fix: shuffle the unique groups before assigning fold IDs.
#
# Augmented rows (aug_k >= 0) are kept in training but EXCLUDED from the OOF
# validation set so the same real rows can't appear in both train and val.
# Without this, aug splits of the same well leak into both sides of the fold.

_unique_wells = np.unique(g.values)
_rng_cv       = np.random.RandomState(SEED)
_shuffled     = _rng_cv.permutation(_unique_wells)
_fold_map     = {w: i % N_SPLITS for i, w in enumerate(_shuffled)}
_fold_ids     = np.array([_fold_map[w] for w in g.values])

# Boolean mask: True for original rows (not augmented)
_is_orig = (train_df["aug_k"].values == -1)

splits: list = []
for _f in range(N_SPLITS):
    _tr_idx = np.where(_fold_ids != _f)[0]                      # all rows not in fold f
    _va_all = np.where(_fold_ids == _f)[0]                      # fold-f rows
    _va_idx = _va_all[_is_orig[_va_all]]                        # keep only original rows
    if len(_va_idx) == 0:
        print(f"[CV] WARNING: fold {_f} has zero original val rows — skipping", flush=True)
        continue
    splits.append((_tr_idx, _va_idx))

print(f"[CV] {len(splits)} folds | "
      f"train aug+orig per fold ≈ {np.mean([len(t) for t,_ in splits]):.0f} | "
      f"val orig-only per fold ≈ {np.mean([len(v) for _,v in splits]):.0f}")



def run_lgb(seed: int, gpu_id: int = 0):
    p   = dict(LGB_P, n_estimators=5000, seed=seed)
    if not FORCE_CPU:
        p["gpu_device_id"] = gpu_id
    oof = np.zeros(len(train_df), np.float32)
    tp  = np.zeros(len(test_df),  np.float32)
    for fold, (tr, va) in enumerate(splits):
        _dbg(f"LGB seed={seed} fold={fold} starting")
        ds_tr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        ds_va = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=ds_tr)
        m = lgb.train(
            p,
            ds_tr,
            valid_sets=[ds_va],
            num_boost_round=p["n_estimators"],
            callbacks=[
                lgb.early_stopping(125, verbose=False),
                lgb.log_evaluation(250),
            ],
        )
        ni = m.best_iteration
        oof[va] = m.predict(X.iloc[va], num_iteration=ni).astype(np.float32)
        tp      += m.predict(Xt, num_iteration=ni).astype(np.float32) / len(splits)
        fold_rmse = root_mean_squared_error(y.iloc[va], oof[va])
        print(f"   LGB{seed} fold{fold}: best_iter={ni} rmse={fold_rmse:.4f}")
        if SAVE_ARTIFACTS:
            model_path = ARTIFACT_DIR / "lgb" / f"seed{seed}_fold{fold}.txt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            m.save_model(str(model_path), num_iteration=ni)
            ARTIFACT_MANIFEST["lgb"].append({
                "seed": int(seed),
                "fold": int(fold),
                "best_iteration": int(ni),
                "rmse": float(fold_rmse),
                "path": _artifact_rel(model_path),
            })
        _dbg(f"LGB seed={seed} fold={fold} done rmse={fold_rmse:.4f}")
    # OOF only over rows that were in some validation set
    va_all = np.concatenate([va for _, va in splits])
    r = root_mean_squared_error(y.iloc[va_all], oof[va_all])
    print(f"   LGB{seed} OOF={r:.4f}")
    return oof, tp, r


def run_cb(seed: int = 42):
    p   = dict(CB_P, random_seed=seed)
    oof = np.zeros(len(train_df), np.float32)
    tp  = np.zeros(len(test_df),  np.float32)
    for fold, (tr, va) in enumerate(splits):
        _dbg(f"CB fold={fold} starting")
        m = CatBoostRegressor(**p)
        m.fit(Pool(X.iloc[tr].values, label=y.iloc[tr].values),
              eval_set=Pool(X.iloc[va].values, label=y.iloc[va].values),
              use_best_model=True)
        oof[va] = m.predict(X.iloc[va].values).astype(np.float32)
        tp      += m.predict(Xt.values).astype(np.float32) / len(splits)
        fold_rmse = root_mean_squared_error(y.iloc[va], oof[va])
        print(f"   CB{seed} fold{fold}: rmse={fold_rmse:.4f}")
        if SAVE_ARTIFACTS:
            model_path = ARTIFACT_DIR / "catboost" / f"seed{seed}_fold{fold}.cbm"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            m.save_model(str(model_path))
            ARTIFACT_MANIFEST["catboost"].append({
                "seed": int(seed),
                "fold": int(fold),
                "rmse": float(fold_rmse),
                "path": _artifact_rel(model_path),
            })
        _dbg(f"CB fold={fold} done rmse={fold_rmse:.4f}")
    va_all = np.concatenate([va for _, va in splits])
    r = root_mean_squared_error(y.iloc[va_all], oof[va_all])
    print(f"   CB{seed} OOF={r:.4f}")
    return oof, tp, r


import threading

_lgb_results = {}
_lgb_exc     = [None]

def _run_lgb_parallel():
    """LGB[42,7] on GPU 0; LGB[123] on GPU 1 — parallel threads."""
    try:
        import threading as _t

        def _gpu0():
            for s in [42, 7]:
                o, t, r = run_lgb(s, gpu_id=0)
                _lgb_results[f'lgb{s}'] = {'oof': o, 'test': t, 'rmse': r}

        def _gpu1():
            o, t, r = run_lgb(123, gpu_id=1)
            _lgb_results['lgb123'] = {'oof': o, 'test': t, 'rmse': r}

        t0 = _t.Thread(target=_gpu0)
        t1 = _t.Thread(target=_gpu1)
        t0.start(); t1.start()
        t0.join();  t1.join()
    except Exception as e:
        _lgb_exc[0] = e

print(">> GPU0: LGB[42,7]  |  GPU1: LGB[123]  (parallel)")
_run_lgb_parallel()
if _lgb_exc[0]: raise _lgb_exc[0]

results = {}
results.update(_lgb_results)

# CB×3 seeds run serially. Device string is detected from available Kaggle GPUs.
for seed in [42, 7, 123]:
    oof_cb, tp_cb, r_cb = run_cb(seed=seed)
    results[f'cb{seed}'] = {'oof': oof_cb, 'test': tp_cb, 'rmse': r_cb}

_va_union = np.concatenate([va for _, va in splits])

# ── TabICL setup ─────────────────────────────────────────────────────────────
if RUN_TABICL:
    import subprocess as _sp, sys as _sys
    from pathlib import Path as _Path

    tabicl_roots = [_Path("/kaggle/input")]
    if os.environ.get("ROGII_TABICL_DIR"):
        tabicl_roots.insert(0, _Path(os.environ["ROGII_TABICL_DIR"]))
    _wheels = []
    _ckpts = []
    for _root in tabicl_roots:
        if _root.exists():
            _wheels.extend(_root.rglob("tabicl-*.whl"))
            _ckpts.extend(_root.rglob("tabicl-regressor*.ckpt"))
    print(f"found wheels: {_wheels}")
    print(f"found ckpts:  {_ckpts}")
    if not _wheels or not _ckpts:
        _msg = (
            "TabICL artifacts not found. Add the public dataset "
            "needless090/rogii-tabicl-mirror, or set ROGII_RUN_TABICL=0."
        )
        if RUNNING_ON_KAGGLE:
            raise RuntimeError(_msg)
        print(_msg + " Skipping TabICL for this local run.", flush=True)
    else:
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                 str(_wheels[0])], check=True)
        TABICL_CKPT = str(_ckpts[0])
        print(f"TabICL ckpt: {TABICL_CKPT}")

        print(">> Selecting top-50 features for TabICL", flush=True)
        _tr0, _ = splits[0]
        _samp = np.random.RandomState(42).choice(_tr0, size=min(200_000, len(_tr0)), replace=False)
        _lgb_sel_params = dict(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            n_jobs=-1, verbosity=-1,
        )
        if not FORCE_CPU:
            _lgb_sel_params["device_type"] = "gpu"
        _lgb_sel = lgb.LGBMRegressor(**_lgb_sel_params)
        _lgb_sel.fit(X.iloc[_samp].values, y.iloc[_samp].values)
        _imp = pd.Series(_lgb_sel.feature_importances_, index=feature_cols).sort_values(ascending=False)
        TABICL_FEATS = _imp.head(50).index.tolist()
        print(f"   top-5: {TABICL_FEATS[:5]}")
        if SAVE_ARTIFACTS:
            save_json(ARTIFACT_DIR / "tabicl_features.json", TABICL_FEATS)
            ARTIFACT_MANIFEST["tabicl_features"] = "tabicl_features.json"

        X_top  = X[TABICL_FEATS].values
        Xt_top = Xt[TABICL_FEATS].values
        CHUNK  = 50_000

        def run_tabicl(ctx_n, n_estimators, seeds, label):
            from tabicl import TabICLRegressor
            print(f"\n>> {label}: ctx={ctx_n} n_est={n_estimators} seeds={seeds}", flush=True)
            n_per = len(seeds)
            oof = np.zeros(len(train_df), dtype=np.float32)
            test_pred = np.zeros(len(test_df), dtype=np.float32)
            for fold, (tr, va) in enumerate(splits):
                oof_fold = np.zeros(len(va), dtype=np.float32)
                for sd in seeds:
                    ctx_idx = np.random.RandomState(sd * 1000 + fold).choice(
                        tr, size=min(ctx_n, len(tr)), replace=False)
                    X_ctx = X_top[ctx_idx]
                    y_ctx = y.iloc[ctx_idx].values.astype(np.float32)
                    if SAVE_ARTIFACTS:
                        ctx_path = (
                            ARTIFACT_DIR / "tabicl_contexts"
                            / f"{label}_fold{fold}_seed{sd}.npz"
                        )
                        ctx_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            ctx_path,
                            X_ctx=X_ctx.astype(np.float32),
                            y_ctx=y_ctx.astype(np.float32),
                        )
                        ARTIFACT_MANIFEST["tabicl_contexts"].append({
                            "label": label,
                            "fold": int(fold),
                            "seed": int(sd),
                            "ctx_n": int(ctx_n),
                            "n_estimators": int(n_estimators),
                            "path": _artifact_rel(ctx_path),
                        })
                    X_va  = X_top[va]
                    t0_ = time.time()
                    reg = TabICLRegressor(model_path=TABICL_CKPT, device="cuda",
                                          random_state=sd, n_estimators=n_estimators,
                                          n_jobs=1, verbose=False, use_amp="auto", batch_size=4)
                    reg.fit(X_ctx, y_ctx)
                    pv = np.empty(len(va), dtype=np.float32)
                    for i in range(0, len(va), CHUNK):
                        e = min(i + CHUNK, len(va))
                        pv[i:e] = reg.predict(X_va[i:e]).astype(np.float32)
                    oof_fold += pv / n_per
                    tf = np.empty(len(test_df), dtype=np.float32)
                    for i in range(0, len(Xt_top), CHUNK):
                        e = min(i + CHUNK, len(Xt_top))
                        tf[i:e] = reg.predict(Xt_top[i:e]).astype(np.float32)
                    test_pred += tf / N_SPLITS / n_per
                    print(f"   fold{fold} seed{sd}: [{time.time()-t0_:.1f}s]", flush=True)
                oof[va] = oof_fold
                rmse_fold = root_mean_squared_error(y.iloc[va].values, oof_fold)
                print(f"   fold{fold} avg RMSE = {rmse_fold:.4f}", flush=True)
            r_val = root_mean_squared_error(y.iloc[_va_union].values, oof[_va_union])
            print(f"   {label} OOF RMSE (val rows) = {r_val:.4f}", flush=True)
            return oof, test_pred, float(r_val)

        # TabICL A: ctx=4096, 4 estimators, 5 seeds (~74 min)
        oof_A, test_A, rmse_A = run_tabicl(4096, 4, [0, 1, 2, 3, 4], "tabicl_A_4096_5seed")
        results["tabicl_A"] = {"oof": oof_A, "test": test_A, "rmse": rmse_A}

        # TabICL B: ctx=8192, 4 estimators, 1 seed (~19 min)
        oof_B, test_B, rmse_B = run_tabicl(8192, 4, [42], "tabicl_B_8192_1seed")
        results["tabicl_B"] = {"oof": oof_B, "test": test_B, "rmse": rmse_B}
else:
    print("ROGII_RUN_TABICL=0; skipping TabICL models.", flush=True)

# Ridge stacking — fit and score only over rows that were actually validated
result_keys = list(results.keys())
Sx = np.column_stack([v["oof"] for v in results.values()]).astype(np.float32)
St = np.column_stack([v["test"] for v in results.values()]).astype(np.float32)
y_val_stack = y.iloc[_va_union].values.astype(np.float32)
Sx_val = Sx[_va_union]


def _rmse_np(y_true: np.ndarray, pred: np.ndarray) -> float:
    diff = pred.astype(np.float64) - y_true.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def hill_climb_stack(
    pred_mat: np.ndarray,
    y_true: np.ndarray,
    keys: list[str],
    precision: float = 0.01,
    allow_negative: bool = True,
):
    """Small deterministic hill-climb stacker, package-free."""
    precision = max(float(precision), 0.001)
    weights = np.arange(-0.5, 0.5001, precision) if allow_negative else np.arange(precision, 0.5001, precision)
    model_scores = [_rmse_np(y_true, pred_mat[:, i]) for i in range(pred_mat.shape[1])]
    start = int(np.argmin(model_scores))
    coef = np.zeros(pred_mat.shape[1], dtype=np.float64)
    coef[start] = 1.0
    current = pred_mat[:, start].astype(np.float32).copy()
    current_score = model_scores[start]
    remaining = [i for i in range(pred_mat.shape[1]) if i != start]
    history = [{
        "iteration": 0,
        "model": keys[start],
        "weight": 1.0,
        "score": float(current_score),
    }]
    iteration = 0
    while remaining:
        iteration += 1
        best = (current_score, None, None, None)
        for idx in remaining:
            new_pred = pred_mat[:, idx]
            for w in weights:
                cand = (1.0 - w) * current + w * new_pred
                score = _rmse_np(y_true, cand)
                if score < best[0] - 1e-7:
                    best = (score, idx, float(w), cand.astype(np.float32))
        if best[1] is None:
            break
        score, idx, w, current = best
        coef *= (1.0 - w)
        coef[idx] += w
        current_score = float(score)
        remaining.remove(idx)
        history.append({
            "iteration": iteration,
            "model": keys[idx],
            "weight": float(w),
            "score": current_score,
        })
    return coef.astype(np.float32), current.astype(np.float32), current_score, history


avg_coef = np.ones(len(result_keys), dtype=np.float32) / max(len(result_keys), 1)
avg_oof = Sx_val.mean(1).astype(np.float32)
avg_test = St.mean(1).astype(np.float32)
r_avg = _rmse_np(y_val_stack, avg_oof)

ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(Sx_val, y_val_stack)
ridge_coef = ridge.coef_.astype(np.float32)
ridge_oof = ridge.predict(Sx_val).astype(np.float32)
ridge_test = ridge.predict(St).astype(np.float32)
r_stk = _rmse_np(y_val_stack, ridge_oof)
ridge_wts = ridge_coef / max(float(ridge_coef.sum()), 1e-9)

stack_options = [
    {
        "kind": "simple_average",
        "coef": avg_coef,
        "intercept": 0.0,
        "oof": avg_oof,
        "test": avg_test,
        "rmse": r_avg,
        "history": [],
    },
    {
        "kind": "positive_ridge",
        "coef": ridge_coef,
        "intercept": float(getattr(ridge, "intercept_", 0.0)),
        "oof": ridge_oof,
        "test": ridge_test,
        "rmse": r_stk,
        "history": [],
    },
]

if USE_HILL_STACK and len(result_keys) >= 2:
    hill_coef, hill_oof, hill_rmse, hill_history = hill_climb_stack(
        Sx_val,
        y_val_stack,
        result_keys,
        precision=HILL_PRECISION,
        allow_negative=True,
    )
    hill_test = (St @ hill_coef).astype(np.float32)
    stack_options.append({
        "kind": "hill_climb",
        "coef": hill_coef,
        "intercept": 0.0,
        "oof": hill_oof,
        "test": hill_test,
        "rmse": hill_rmse,
        "history": hill_history,
    })

best_stack = min(stack_options, key=lambda item: item["rmse"])
final_test = best_stack["test"]
_final_oof_full = Sx.mean(1).astype(np.float32)
_final_oof_full[_va_union] = best_stack["oof"]
final_oof = _final_oof_full

print(f"\nSimple avg OOF (val rows): {r_avg:.4f}")
print(f"Ridge stk OOF (val rows): {r_stk:.4f}  wts={dict(zip(result_keys, ridge_wts.round(4)))}")
if USE_HILL_STACK and len(result_keys) >= 2:
    print(f"Hill stk OOF (val rows): {stack_options[-1]['rmse']:.4f}  coefs={dict(zip(result_keys, stack_options[-1]['coef'].round(4)))}")
print(f"Using {best_stack['kind']} predictions.")
print(f"Final OOF RMSE (val rows): {best_stack['rmse']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Post-Processing
# ═══════════════════════════════════════════════════════════════════════════════
# Grid-search alpha/tau only on val rows (where OOF is clean).
# SavGol smoothing is REMOVED from test predictions — it introduced bias on the
# initial rows of each well and hurt the leaderboard score vs the reference.

base_val = train_df["last_known_tvt"].values[_va_union]
ytrue_val = y.values[_va_union] + base_val
md_val = train_df["md_since"].values[_va_union]
oof_val = final_oof[_va_union]
pf_val = (train_df["pf_ancc"].values[_va_union] - base_val).astype(np.float32)

best_cfg, best_r = (None, None, None), np.inf
for alpha in np.arange(0.6, 1.01, 0.05):
    for tau in [None, 30.0, 60.0, 120.0, 250.0, 500.0]:
        for w_pf in np.arange(0.0, 0.501, 0.05):
            d = ((1.0 - w_pf) * oof_val + w_pf * pf_val).astype(np.float32)
            if tau:
                d *= 1.0 - np.exp(-np.maximum(md_val, 0.0) / tau)
            r = root_mean_squared_error(ytrue_val, base_val + d * alpha)
            if r < best_r:
                best_r, best_cfg = r, (float(alpha), tau, float(w_pf))

print(
    f"Best post-proc: alpha={best_cfg[0]:.2f} tau={best_cfg[1]} "
    f"w_pf={best_cfg[2]:.2f}  TVT RMSE={best_r:.4f}"
)
ALPHA, TAU, W_PF = best_cfg
_fallback_tvt = float(
    train_df["last_known_tvt"].iloc[_va_union].mean()
    + train_df["target"].iloc[_va_union].mean()
)
if SAVE_ARTIFACTS:
    stacker_config = {
        "kind": best_stack["kind"],
        "coef": best_stack["coef"].astype(float).tolist(),
        "intercept": float(best_stack.get("intercept", 0.0)),
        "rmse": float(best_stack["rmse"]),
        "history": best_stack.get("history", []),
        "hill_precision": float(HILL_PRECISION),
    }
    config = {
        "result_keys": result_keys,
        "stacker": stacker_config,
        "use_ridge": bool(best_stack["kind"] == "positive_ridge"),
        "ridge_alpha": 1.0,
        "ridge_fit_intercept": False,
        "ridge_coef": ridge_coef.astype(float).tolist(),
        "ridge_intercept": float(getattr(ridge, "intercept_", 0.0)),
        "simple_avg_oof": float(r_avg),
        "ridge_oof": float(r_stk),
        "final_oof": float(best_stack["rmse"]),
        "postproc_alpha": float(ALPHA),
        "postproc_tau": None if TAU is None else float(TAU),
        "postproc_w_pf": float(W_PF),
        "postproc_oof_rmse": float(best_r),
        "fallback_tvt": _fallback_tvt,
        "exact_overlap_enabled": os.environ.get("ROGII_EXACT_OVERLAP", "1").strip().lower()
        not in {"0", "false", "no"},
        "exact_blend_weight": env_float("ROGII_EXACT_BLEND_WEIGHT", 0.28),
    }
    save_json(ARTIFACT_DIR / "inference_config.json", config)
    ARTIFACT_MANIFEST["inference_config"] = "inference_config.json"
    ARTIFACT_MANIFEST["result_keys"] = result_keys
    diag_dir = ARTIFACT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    np.save(diag_dir / "base_test_predictions.npy", St.astype(np.float32))
    ARTIFACT_MANIFEST["created_outputs"]["base_test_predictions"] = _artifact_rel(
        diag_dir / "base_test_predictions.npy"
    )
    np.savez_compressed(
        diag_dir / "oof_val_predictions.npz",
        predictions=Sx_val.astype(np.float32),
        target=y_val_stack.astype(np.float32),
        val_indices=_va_union.astype(np.int64),
    )
    save_json(diag_dir / "prediction_keys.json", result_keys)
    ARTIFACT_MANIFEST["created_outputs"]["oof_val_predictions"] = _artifact_rel(
        diag_dir / "oof_val_predictions.npz"
    )
    ARTIFACT_MANIFEST["created_outputs"]["prediction_keys"] = _artifact_rel(
        diag_dir / "prediction_keys.json"
    )
    meta_cols = [
        c for c in ["well", "id", "target", "last_known_tvt", "md_since", "pf_ancc", "aug_k"]
        if c in train_df.columns
    ]
    train_df.loc[_va_union, meta_cols].to_csv(
        diag_dir / "oof_val_meta.csv.gz",
        index=False,
        compression="gzip",
    )
    ARTIFACT_MANIFEST["created_outputs"]["oof_val_meta"] = _artifact_rel(
        diag_dir / "oof_val_meta.csv.gz"
    )
    test_diag = pd.DataFrame({"id": test_df["id"].values})
    for i, key in enumerate(result_keys):
        test_diag[key] = St[:, i].astype(np.float32)
    test_diag.to_csv(diag_dir / "test_base_predictions.csv.gz", index=False, compression="gzip")
    ARTIFACT_MANIFEST["created_outputs"]["test_base_predictions"] = _artifact_rel(
        diag_dir / "test_base_predictions.csv.gz"
    )
    if SAVE_FEATURE_FRAMES:
        frame_dir = ARTIFACT_DIR / "feature_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_pickle(frame_dir / "train_features.pkl")
        test_df.to_pickle(frame_dir / "test_features.pkl")
        ARTIFACT_MANIFEST["created_outputs"]["train_features"] = _artifact_rel(
            frame_dir / "train_features.pkl"
        )
        ARTIFACT_MANIFEST["created_outputs"]["test_features"] = _artifact_rel(
            frame_dir / "test_features.pkl"
        )


def apply_pp(df: pd.DataFrame, delta: np.ndarray,
             alpha: float, tau, w_pf: float = 0.0) -> np.ndarray:
    pf_delta = (df["pf_ancc"].values - df["last_known_tvt"].values).astype(np.float32)
    d = ((1.0 - w_pf) * delta.astype(np.float32) + w_pf * pf_delta).astype(np.float32)
    if tau:
        d *= 1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau)
    return d * alpha


# Apply post-processing to test predictions — no SavGol smoothing
test_df2 = test_df.copy()
test_df2["pred"] = (test_df2["last_known_tvt"].values
                    + apply_pp(test_df2, final_test, ALPHA, TAU, W_PF))

sample = pd.read_csv(SAMPLE)
sub = sample[["id"]].merge(
    test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}),
    on="id", how="left",
)
_fb_val = _fallback_tvt
sub["tvt"] = sub["tvt"].fillna(_fb_val)

n_fill = sub["tvt"].isnull().sum()
n_fb   = (sub["tvt"] == _fb_val).sum()
print(f"[SUB AUDIT] rows={len(sub)}  null={n_fill}  fallback_filled={n_fb}", flush=True)

sub = apply_exact_train_coordinate_blend(sub[["id", "tvt"]], DATA)
sub[["id", "tvt"]].to_csv(OUT, index=False)
if SAVE_ARTIFACTS:
    ARTIFACT_MANIFEST["created_outputs"]["submission"] = str(OUT)
    save_json(ARTIFACT_DIR / "manifest.json", ARTIFACT_MANIFEST)
    print(f"Artifact manifest -> {ARTIFACT_DIR / 'manifest.json'}", flush=True)

print(f"\n✅  {OUT}  {len(sub)} rows")
print("\n─── Final Summary ───────────────────────────")
for k, v in results.items():
    print(f"  {k}: OOF = {v['rmse']:.4f}")
print(f"  Best stack: {best_stack['kind']} OOF = {best_stack['rmse']:.4f}  |  PostProc TVT RMSE: {best_r:.4f}")
print(sub.head(8).to_string(index=False))
print(f"=== PIPELINE COMPLETE  total={time.time()-_T0_GLOBAL:.0f}s ===")
