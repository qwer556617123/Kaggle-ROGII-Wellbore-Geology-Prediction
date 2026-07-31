"""Build the post-dose 7.039 anchor and exact-HMM Kaggle notebooks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


YUSUKE_SOURCE = Path(
    "kaggle/external_reviews/yusuke-another-approach/rogii-another-approach.ipynb"
)
PK_SOURCE = Path(
    "kaggle/external_reviews/rogii-pk-adopt-lb7061/"
    "rogii-pk-adopt-pb-lb-7-061.ipynb"
)
HMM_SOURCE = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")

YUSUKE_OUTPUT = Path("kaggle/rogii-yusuke7039-rebuild")
HMM_OUTPUT = Path("kaggle/rogii-pkadopt-hmm015")


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def clean_code_cells(path: Path, skipped: set[int]) -> tuple[dict, list[dict]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, original in enumerate(notebook["cells"]):
        if index in skipped or original.get("cell_type") != "code":
            continue
        if not source(original).strip():
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        cell.get("metadata", {}).pop("execution", None)
        cell.get("metadata", {}).pop("papermill", None)
        text = source(cell)
        text = text.replace(
            "print(f'  PF {int(globals().get('SELECTOR_PF_SEEDS', "
            "SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK "
            "scales={SELECTOR_SCALES}')",
            "print(f\"  PF {int(globals().get('SELECTOR_PF_SEEDS', "
            "SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK "
            "scales={SELECTOR_SCALES}\")",
        )
        cell["source"] = text.splitlines(keepends=True)
        cells.append(cell)
    return notebook, cells


def integrity_audit(strategy: str, started_name: str, extra: str = "") -> str:
    return f'''# Final rerun audit.
import hashlib as _sd_hashlib
import json as _sd_json
import time as _sd_time
from pathlib import Path as _SDPath
import numpy as _sd_np
import pandas as _sd_pd

_sd_work = _SDPath('/kaggle/working') if _SDPath('/kaggle/working').exists() else _SDPath('.')
_sd_data = _SDPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_sd_data / 'sample_submission.csv').exists():
    _sd_data = _SDPath('/kaggle/input/rogii-wellbore-geology-prediction')
_sd_path = _sd_work / 'submission.csv'
_sd_sub = _sd_pd.read_csv(_sd_path)[['id', 'tvt']]
_sd_sample = _sd_pd.read_csv(_sd_data / 'sample_submission.csv')[['id']]
_sd_sub['id'] = _sd_sub['id'].astype(str)
_sd_sample['id'] = _sd_sample['id'].astype(str)
_sd_values = _sd_sub['tvt'].to_numpy(dtype=float)
if len(_sd_sub) != len(_sd_sample) or not _sd_sub['id'].equals(_sd_sample['id']):
    raise RuntimeError('submission/sample alignment mismatch')
if not _sd_np.isfinite(_sd_values).all():
    raise RuntimeError('submission contains non-finite tvt')

def _sd_sha(path):
    digest = _sd_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

_sd_audit = {{
    'strategy': {strategy!r},
    'rows': int(len(_sd_sub)),
    'id_order_matches_sample': True,
    'runtime_sec': float(_sd_time.time() - {started_name}),
    'submission_sha256': _sd_sha(_sd_path),
    'tvt_min': float(_sd_values.min()),
    'tvt_max': float(_sd_values.max()),
    'tvt_mean': float(_sd_values.mean()),
    'tvt_std': float(_sd_values.std()),
}}
{extra}
(_sd_work / 'stopdose_submission_audit.json').write_text(
    _sd_json.dumps(_sd_audit, indent=2, sort_keys=True), encoding='utf-8'
)
print('stop-dose final audit:', _sd_audit, flush=True)
'''


def write_metadata(source_path: Path, output: Path, kernel_id: str) -> None:
    metadata = json.loads(
        (source_path.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    metadata.pop("id_no", None)
    metadata.update(
        {
            "id": f"qwer556617123/{kernel_id}",
            "title": kernel_id,
            "code_file": f"{kernel_id}.ipynb",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "kernel_sources": [],
            "competition_sources": ["rogii-wellbore-geology-prediction"],
            "model_sources": [],
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_notebook(notebook: dict, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{label}_cell_{index}", "exec")


def build_yusuke7039() -> Path:
    # 1 is an unnecessary online pip install; 26/27/30/36 are plots; 42 is the
    # post-7.039 A030 hedge and must not be included in the exact anchor.
    notebook, cells = clean_code_cells(
        YUSUKE_SOURCE, skipped={1, 26, 27, 30, 36, 42}
    )
    joined = "\n".join(source(cell) for cell in cells)
    if "Gold visible-prefix calibration overlay" not in joined:
        raise RuntimeError("7.039 visible-prefix final layer not found")
    if "A030 final tail-distance-weighted" in joined:
        raise RuntimeError("A030 hedge leaked into exact 7.039 anchor")
    start = code_cell("import time as _sd_time\n_SD_STARTED = _sd_time.time()\n")
    notebook["cells"] = [
        start,
        *cells,
        code_cell(integrity_audit("yusuke_exact_7039_source_rebuild", "_SD_STARTED")),
    ]
    validate_notebook(notebook, "yusuke7039")
    YUSUKE_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = YUSUKE_OUTPUT / "rogii-yusuke7039-rebuild.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    write_metadata(YUSUKE_SOURCE, YUSUKE_OUTPUT, "rogii-yusuke7039-rebuild")
    return path


def hmm_runner() -> str:
    return r'''# Exact second-order HMM posterior-mean blend over the 7.053 anchor.
import hashlib as _hmm_hashlib
import json as _hmm_json
import os as _hmm_os
import time as _hmm_time
from pathlib import Path as _HmmPath
import numpy as _hmm_np
import pandas as _hmm_pd
from joblib import Parallel as _HmmParallel
from joblib import delayed as _hmm_delayed

_HMM_STARTED = _hmm_time.time()
_HMM_WEIGHT = 0.15
_hmm_work = _HmmPath('/kaggle/working') if _HmmPath('/kaggle/working').exists() else _HmmPath('.')
_hmm_data = _HmmPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_hmm_data / 'sample_submission.csv').exists():
    _hmm_data = _HmmPath('/kaggle/input/rogii-wellbore-geology-prediction')
_hmm_base_path = _hmm_work / 'submission.csv'
_hmm_base = _hmm_pd.read_csv(_hmm_base_path)[['id', 'tvt']].copy()
_hmm_base['id'] = _hmm_base['id'].astype(str)
_hmm_base.to_csv(_hmm_work / 'submission_before_exact_hmm.csv', index=False)

def _hmm_one(wid):
    started = _hmm_time.time()
    hw = _hmm_pd.read_csv(_hmm_data / 'test' / f'{wid}__horizontal_well.csv')
    tw = _hmm_pd.read_csv(_hmm_data / 'test' / f'{wid}__typewell.csv')[['TVT', 'GR']]
    result = run_hmm2(hw[TEST_COLS].copy(), tw)
    ev = hw['TVT_input'].isna().to_numpy()
    rows = _hmm_np.flatnonzero(ev)
    pred = _hmm_np.asarray(result['pred'], dtype=float)[ev]
    std = _hmm_np.asarray(result['std_eval'], dtype=float)
    frame = _hmm_pd.DataFrame({
        'id': [f'{wid}_{int(row)}' for row in rows],
        'hmm_tvt': pred,
        'hmm_std': std,
    })
    audit = {
        'well': wid,
        'rows': int(len(frame)),
        'loglik': float(result['loglik']),
        'posterior_std_mean': float(_hmm_np.mean(std)),
        'posterior_std_p95': float(_hmm_np.quantile(std, 0.95)),
        'posterior_std_max': float(_hmm_np.max(std)),
        'runtime_sec': float(_hmm_time.time() - started),
    }
    return frame, audit

_hmm_wells = sorted(_hmm_base['id'].str.rsplit('_', n=1).str[0].unique())
# The numba kernel releases the GIL. Cap workers to avoid multiplying its large
# forward/backward state tensor on hidden reruns.
_hmm_jobs = max(1, min(3, len(_hmm_wells), _hmm_os.cpu_count() or 1))
_hmm_results = _HmmParallel(n_jobs=_hmm_jobs, prefer='threads')(
    _hmm_delayed(_hmm_one)(wid) for wid in _hmm_wells
)
_hmm_pred = _hmm_pd.concat([item[0] for item in _hmm_results], ignore_index=True)
_hmm_well_audits = [item[1] for item in _hmm_results]
_hmm_merged = _hmm_base.merge(_hmm_pred, on='id', how='left', validate='one_to_one')
if _hmm_merged['hmm_tvt'].isna().any():
    raise RuntimeError('exact HMM did not cover every submission row')
_hmm_base_values = _hmm_merged['tvt'].to_numpy(dtype=float)
_hmm_values = _hmm_merged['hmm_tvt'].to_numpy(dtype=float)
_hmm_final_values = (1.0 - _HMM_WEIGHT) * _hmm_base_values + _HMM_WEIGHT * _hmm_values
if not _hmm_np.isfinite(_hmm_final_values).all():
    raise RuntimeError('non-finite exact HMM blend')
_hmm_final = _hmm_merged[['id']].copy()
_hmm_final['tvt'] = _hmm_final_values
_hmm_final.to_csv(_hmm_work / 'submission.csv', index=False)
_hmm_pred.to_csv(_hmm_work / 'exact_hmm_predictions.csv', index=False)

def _hmm_sha(path):
    return _hmm_hashlib.sha256(_HmmPath(path).read_bytes()).hexdigest()

_hmm_delta = _hmm_final_values - _hmm_base_values
_HMM_AUDIT = {
    'strategy': 'pkadopt_7053_plus_exact_second_order_hmm',
    'weight': _HMM_WEIGHT,
    'workers': int(_hmm_jobs),
    'rows': int(len(_hmm_final)),
    'wells': int(len(_hmm_wells)),
    'runtime_sec_hmm_layer': float(_hmm_time.time() - _HMM_STARTED),
    'mean_abs_move': float(_hmm_np.mean(_hmm_np.abs(_hmm_delta))),
    'p95_abs_move': float(_hmm_np.quantile(_hmm_np.abs(_hmm_delta), 0.95)),
    'max_abs_move': float(_hmm_np.max(_hmm_np.abs(_hmm_delta))),
    'base_sha256': _hmm_sha(_hmm_work / 'submission_before_exact_hmm.csv'),
    'final_sha256': _hmm_sha(_hmm_work / 'submission.csv'),
    'well_audits': _hmm_well_audits,
}
(_hmm_work / 'exact_hmm_audit.json').write_text(
    _hmm_json.dumps(_HMM_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('exact HMM audit:', _HMM_AUDIT, flush=True)
'''


def build_hmm() -> Path:
    # Disabled diagnostics and plot-only cells are removed. The 7.053 model
    # package correction and every prediction-changing cell remain untouched.
    notebook, cells = clean_code_cells(
        PK_SOURCE, skipped={12, 34, 35, 38, 44, 47, 53}
    )
    hmm_notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    hmm_core = copy.deepcopy(hmm_notebook["cells"][58])
    hmm_core["execution_count"] = None
    hmm_core["outputs"] = []
    joined = "\n".join(source(cell) for cell in cells)
    if "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_010'" not in joined:
        raise RuntimeError("verified 0.010 anchor profile not found")
    if "COMPAT SHIM" not in joined:
        raise RuntimeError("CPU compatibility shim missing from 7.053 source")
    start = code_cell("import time as _sd_time\n_SD_STARTED = _sd_time.time()\n")
    extra = "_sd_audit['hmm'] = _HMM_AUDIT"
    notebook["cells"] = [
        start,
        *cells,
        hmm_core,
        code_cell(hmm_runner()),
        code_cell(
            integrity_audit(
                "pkadopt_7053_exact_hmm_weight_015", "_SD_STARTED", extra
            )
        ),
    ]
    validate_notebook(notebook, "pkadopt_hmm")
    HMM_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = HMM_OUTPUT / "rogii-pkadopt-hmm015.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    write_metadata(PK_SOURCE, HMM_OUTPUT, "rogii-pkadopt-hmm015")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()
    targets = args.targets or ["yusuke7039", "hmm"]
    unknown = sorted(set(targets) - {"yusuke7039", "hmm"})
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}")
    for target in targets:
        path = build_yusuke7039() if target == "yusuke7039" else build_hmm()
        print(f"built {path}")


if __name__ == "__main__":
    main()
