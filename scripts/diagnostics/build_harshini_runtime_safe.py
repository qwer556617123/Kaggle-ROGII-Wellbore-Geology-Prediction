from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCE = Path(
    "kaggle/cache/latest_public/harshini-f/harshini-submission-f.ipynb"
)
PROFILES = {
    "fast1": {
        "slug": "rogii-harshini-f-fast",
        "cache_wells": 765,
        "pf_seeds": 10,
        "pf_particles": 220,
        "train_stride": 12,
        "lgb1_estimators": 800,
        "lgb2_estimators": 1200,
        "catboost_iterations": 700,
    },
    "fast2": {
        "slug": "rogii-harshini-f-fast2",
        "cache_wells": 240,
        "pf_seeds": 8,
        "pf_particles": 180,
        "train_stride": 20,
        "lgb1_estimators": 400,
        "lgb2_estimators": 600,
        "catboost_iterations": 300,
    },
}


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def prefix(cell: dict, stop_marker: str) -> dict:
    text = source(cell)
    if stop_marker not in text:
        raise RuntimeError(f"trim marker missing: {stop_marker}")
    out = copy.deepcopy(cell)
    out["source"] = text.split(stop_marker, 1)[0].rstrip() + "\n"
    out["execution_count"] = None
    out["outputs"] = []
    return out


def build(profile_name: str = "fast1") -> Path:
    profile = PROFILES[profile_name]
    slug = str(profile["slug"])
    output = Path("kaggle/cache/our-runs") / slug
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    harness = prefix(cells[0], "dz=[]; dtvt=[]; flat_res=[]")
    harness["source"] = (
        "import time as _hs_time\n_HS_STARTED = _hs_time.time()\n" + source(harness)
    )

    field_cloud = prefix(cells[2], "# ---- leave-one-well-out evaluation ----")
    detrended = prefix(cells[3], "# --- 3. evaluate ---")
    detrended["source"] = (
        source(detrended)
        + "\nHFm = {w:(hf,tf,ps) for w,hf,tf,ps,ln in TR}\n"
    )
    local_surface = prefix(cells[4], "def try_cfg(name, fn, **kw):")
    cache = prefix(cells[5], "# THE DECISIVE TEST: what does an EXACT datum buy?")
    cache["source"] = source(cache).rsplit("# ============================================================", 1)[0]
    cache_wells = int(profile["cache_wells"])
    if cache_wells < 765:
        cache_text = source(cache)
        cache_text = cache_text.replace(
            "CACHE = {}\nt0 = time.time()",
            f"""CACHE = {{}}
_hs_rng = np.random.default_rng(20260730)
_hs_all_wells = np.asarray([w for w, _, _, _, _ in TR], dtype=object)
_HS_CACHE_WELLS = set(_hs_rng.choice(
    _hs_all_wells, size=min({cache_wells}, len(_hs_all_wells)), replace=False
).tolist())
t0 = time.time()""",
            1,
        )
        cache_text = cache_text.replace(
            "for k, (wid, hf, tf, ps, ln) in enumerate(TR):\n    if wid not in PT: continue",
            "for k, (wid, hf, tf, ps, ln) in enumerate(TR):\n"
            "    if wid not in _HS_CACHE_WELLS: continue\n"
            "    if wid not in PT: continue",
            1,
        )
        cache["source"] = cache_text
    trust = copy.deepcopy(cells[6])
    final = copy.deepcopy(cells[8])

    for cell in (trust, final):
        cell["execution_count"] = None
        cell["outputs"] = []

    final_text = source(final)
    replacements = {
        "PF_TRAIN_WELLS=765; PF_SEEDS,PF_PART=24,300; TRAIN_STRIDE=8": (
            f"PF_TRAIN_WELLS={cache_wells}; PF_SEEDS,PF_PART="
            f"{profile['pf_seeds']},{profile['pf_particles']}; "
            f"TRAIN_STRIDE={profile['train_stride']}"
        ),
        "n_estimators=3000": f"n_estimators={profile['lgb1_estimators']}",
        "n_estimators=5000": f"n_estimators={profile['lgb2_estimators']}",
        "iterations=4000": f"iterations={profile['catboost_iterations']}",
    }
    for old, new in replacements.items():
        if old not in final_text:
            raise RuntimeError(f"runtime replacement missing: {old}")
        final_text = final_text.replace(old, new, 1)
    final["source"] = final_text

    audit = code_cell(
        """# Runtime and output contract audit.
import hashlib as _hs_hashlib
import json as _hs_json
from pathlib import Path as _HsPath
import numpy as _hs_np
import pandas as _hs_pd

_HS_WORK = _HsPath('/kaggle/working') if _HsPath('/kaggle/working').exists() else _HsPath('.')
_HS_SUB = _HS_WORK / 'submission.csv'
_hs_sub = _hs_pd.read_csv(_HS_SUB, dtype={'id': 'string'})
_hs_sample = _hs_pd.read_csv(_HsPath(DATA) / 'sample_submission.csv', dtype={'id': 'string'})
if not _hs_sub['id'].equals(_hs_sample['id']):
    raise RuntimeError('Harshini fast output ID order mismatch')
if not _hs_np.isfinite(_hs_sub['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('Harshini fast output contains non-finite TVT')
_HS_AUDIT = {
    'strategy': 'harshini_dynamic_surface_oof_runtime_safe',
    'rows': int(len(_hs_sub)),
    'wells': int(_hs_sub['id'].str.rsplit('_', n=1).str[0].nunique()),
    'pf_train_wells': int(PF_TRAIN_WELLS),
    'pf_seeds': int(PF_SEEDS),
    'pf_particles': int(PF_PART),
    'train_stride': int(TRAIN_STRIDE),
    'catboost_enabled': bool(HAVE_CB and USE_CB),
    'runtime_seconds': float(_hs_time.time() - _HS_STARTED),
    'submission_sha256': _hs_hashlib.sha256(_HS_SUB.read_bytes()).hexdigest(),
    'fixed_public_ids_used': False,
}
(_HS_WORK / 'harshini_runtime_audit.json').write_text(
    _hs_json.dumps(_HS_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('Harshini runtime audit:', _hs_json.dumps(_HS_AUDIT, indent=2, sort_keys=True))
"""
    )
    audit["source"] = source(audit).replace(
        "'strategy': 'harshini_dynamic_surface_oof_runtime_safe',",
        f"'strategy': 'harshini_dynamic_surface_oof_{profile_name}',\n"
        "    'cache_wells': int(len(CACHE)),",
        1,
    )

    notebook["cells"] = [
        harness,
        field_cloud,
        detrended,
        local_surface,
        cache,
        trust,
        final,
        audit,
    ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    joined = "\n".join(source(cell) for cell in notebook["cells"])
    compile(joined, f"{slug}.py", "exec")
    if "00e12e8b" in joined:
        raise RuntimeError("fixed public well ID leaked into Harshini fast build")
    expected_pf = (
        f"PF_TRAIN_WELLS={cache_wells}; PF_SEEDS,PF_PART="
        f"{profile['pf_seeds']},{profile['pf_particles']}; "
        f"TRAIN_STRIDE={profile['train_stride']}"
    )
    if expected_pf not in joined:
        raise RuntimeError("runtime-safe PF profile missing")

    output.mkdir(parents=True, exist_ok=True)
    notebook_path = output / f"{slug}.ipynb"
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=True), encoding="utf-8"
    )
    metadata = {
        "id": f"qwer556617123/{slug}",
        "title": slug,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return notebook_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="fast1")
    args = parser.parse_args()
    print(build(args.profile))
