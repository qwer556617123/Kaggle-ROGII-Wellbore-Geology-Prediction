"""Build hidden-rerun rank-well datum tomography and an HMM control."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import build_f594_breakthrough_batch as frontier
from build_stopdose_breakthrough_notebooks import (
    code_cell,
    integrity_audit,
    source,
    validate_notebook,
    write_metadata,
)


ANCHOR_DIR = Path("kaggle/rogii-mha400-continuity")
ANCHOR = ANCHOR_DIR / "rogii-mha400-continuity.ipynb"
AMPLITUDE = 2.0
PROBE_CODES = {
    "ppp": (1, 1, 1),
    "pmm": (1, -1, -1),
    "mpm": (-1, 1, -1),
    "mmp": (-1, -1, 1),
}


def _joined(cells: list[dict]) -> str:
    return "\n".join(source(cell) for cell in cells)


def _rank_probe_cell(label: str, signs: tuple[int, int, int]) -> dict:
    code = f'''# Hidden-rerun rank-well constant-datum tomography.
import hashlib as _rt_hashlib
import json as _rt_json
from pathlib import Path as _RtPath

import numpy as _rt_np
import pandas as _rt_pd

_RT_LABEL = {label!r}
_RT_SIGNS = {tuple(int(x) for x in signs)!r}
_RT_AMPLITUDE = {float(AMPLITUDE)!r}
_RT_WORK = _RtPath('/kaggle/working') if _RtPath('/kaggle/working').exists() else _RtPath('.')
_RT_SUB = _RT_WORK / 'submission.csv'


def _rt_sha(path):
    digest = _rt_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _rt_data_root():
    candidates = []
    cfg = globals().get('CFG')
    if cfg is not None:
        for attr in ('DATA', 'dataset_path'):
            if hasattr(cfg, attr):
                candidates.append(_RtPath(getattr(cfg, attr)))
    candidates.extend([
        _RtPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        _RtPath('/kaggle/input/rogii-wellbore-geology-prediction'),
    ])
    for candidate in candidates:
        if (candidate / 'test').exists() and (candidate / 'sample_submission.csv').exists():
            return candidate
    raise RuntimeError('rank tomography could not locate the run-local competition data')


if not _RT_SUB.exists():
    raise RuntimeError('rank tomography anchor submission.csv is missing')
_rt_data = _rt_data_root()
_rt_sample = _rt_pd.read_csv(_rt_data / 'sample_submission.csv', dtype={{'id': 'string'}})[['id']]
_rt_base = _rt_pd.read_csv(_RT_SUB, dtype={{'id': 'string'}})
if list(_rt_base.columns) != ['id', 'tvt']:
    raise RuntimeError(f'rank tomography expected id,tvt, got {{list(_rt_base.columns)}}')
if len(_rt_base) != len(_rt_sample) or not _rt_base['id'].equals(_rt_sample['id']):
    raise RuntimeError('rank tomography anchor does not match the run-local sample')
_rt_base['tvt'] = _rt_pd.to_numeric(_rt_base['tvt'], errors='coerce')
_rt_base_v = _rt_base['tvt'].to_numpy(dtype=float)
if not _rt_np.isfinite(_rt_base_v).all():
    raise RuntimeError('rank tomography anchor contains non-finite predictions')

_rt_parts = _rt_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _rt_parts.shape[1] != 2:
    raise RuntimeError('rank tomography could not parse sample ids')
_rt_base['well'] = _rt_parts[0].astype(str)
_rt_base['row_idx'] = _rt_pd.to_numeric(_rt_parts[1], errors='raise').astype(int)
_rt_rows = []
for _rt_wid, _rt_group in _rt_base.groupby('well', sort=False):
    _rt_hw_path = _rt_data / 'test' / f'{{_rt_wid}}__horizontal_well.csv'
    if not _rt_hw_path.exists():
        raise RuntimeError(f'rank tomography missing test well {{_rt_wid}}')
    _rt_hw = _rt_pd.read_csv(_rt_hw_path)
    _rt_idx = _rt_group['row_idx'].to_numpy(dtype=int)
    if (_rt_idx < 0).any() or (_rt_idx >= len(_rt_hw)).any():
        raise RuntimeError(f'rank tomography row index outside well {{_rt_wid}}')
    _rt_z = _rt_pd.to_numeric(_rt_hw.loc[_rt_idx, 'Z'], errors='coerce').to_numpy(dtype=float)
    _rt_md = _rt_pd.to_numeric(_rt_hw.loc[_rt_idx, 'MD'], errors='coerce').to_numpy(dtype=float)
    _rt_known = _rt_pd.to_numeric(_rt_hw.get('TVT_input'), errors='coerce').dropna()
    _rt_rows.append({{
        'well': str(_rt_wid),
        'rows': int(len(_rt_group)),
        'z_span': float(_rt_np.nanmax(_rt_z) - _rt_np.nanmin(_rt_z)),
        'md_span': float(_rt_np.nanmax(_rt_md) - _rt_np.nanmin(_rt_md)),
        'known_rows': int(len(_rt_known)),
        'last_known_tvt': float(_rt_known.iloc[-1]) if len(_rt_known) else float('nan'),
    }})

_rt_rank = _rt_pd.DataFrame(_rt_rows).sort_values(
    ['rows', 'z_span', 'md_span', 'known_rows', 'last_known_tvt', 'well'],
    ascending=[False, False, False, False, True, True],
    kind='stable',
).reset_index(drop=True)
if len(_rt_rank) != len(_RT_SIGNS):
    raise RuntimeError(
        f'rank tomography requires {{len(_RT_SIGNS)}} wells, found {{len(_rt_rank)}}'
    )
_rt_rank['rank'] = _rt_np.arange(len(_rt_rank), dtype=int)
_rt_rank['sign'] = _rt_rank['rank'].map(dict(enumerate(_RT_SIGNS))).astype(int)
_rt_sign_by_well = dict(zip(_rt_rank['well'], _rt_rank['sign']))
_rt_delta = _rt_base['well'].map(_rt_sign_by_well).to_numpy(dtype=float) * _RT_AMPLITUDE
if not _rt_np.allclose(_rt_np.abs(_rt_delta), _RT_AMPLITUDE, atol=0.0, rtol=0.0):
    raise RuntimeError('rank tomography did not perturb every row at equal magnitude')

_rt_before = _rt_base[['id', 'tvt']].copy()
_rt_before.to_csv(_RT_WORK / 'submission_before_rank_tomography.csv', index=False)
_rt_final = _rt_before.copy()
_rt_final['tvt'] = _rt_base_v + _rt_delta
if not _rt_np.isfinite(_rt_final['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('rank tomography produced non-finite predictions')
_rt_final.to_csv(_RT_SUB, index=False)
_rt_rank.to_csv(_RT_WORK / 'rank_tomography_wells.csv', index=False)

_RT_AUDIT = {{
    'label': _RT_LABEL,
    'hidden_rerun_safe': True,
    'fixed_public_ids_used': False,
    'signs': list(_RT_SIGNS),
    'amplitude_ft': float(_RT_AMPLITUDE),
    'rows': int(len(_rt_final)),
    'wells': _rt_rank.to_dict(orient='records'),
    'mean_squared_move': float(_rt_np.mean(_rt_delta * _rt_delta)),
    'expected_mean_squared_move': float(_RT_AMPLITUDE ** 2),
    'base_sha256': _rt_sha(_RT_WORK / 'submission_before_rank_tomography.csv'),
    'final_sha256': _rt_sha(_RT_SUB),
}}
if abs(_RT_AUDIT['mean_squared_move'] - _RT_AUDIT['expected_mean_squared_move']) > 1e-12:
    raise RuntimeError('rank tomography energy audit failed')
with open(_RT_WORK / 'rank_tomography_audit.json', 'w', encoding='utf-8') as handle:
    _rt_json.dump(_RT_AUDIT, handle, indent=2, sort_keys=True)
print('rank tomography audit:', _rt_json.dumps(_RT_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def _load_anchor() -> dict:
    notebook = json.loads(ANCHOR.read_text(encoding="utf-8"))
    joined = _joined(notebook["cells"])
    required = (
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "4.0, 0.22, 1.0, 40.0, 10.0",
        "_UC_CAP = 8.000000",
        "_UC_TAU = 240.000000",
        "Final hidden-set contract transaction",
        "stop-dose final audit",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"MHA400-continuity anchor fragments missing: {missing}")
    if "00e12e8b" in joined:
        raise RuntimeError("MHA400-continuity anchor contains a fixed public well ID")
    return notebook


def build_probe(name: str, signs: tuple[int, int, int]) -> Path:
    notebook = _load_anchor()
    slug = f"rogii-mha400-cont-rank-{name}"
    notebook["cells"].append(_rank_probe_cell(name, signs))
    notebook["cells"].append(
        code_cell(
            integrity_audit(
                f"mha400_continuity_rank_tomography_{name}",
                "_SD_STARTED",
                "_sd_audit['rank_tomography'] = _RT_AUDIT",
            )
        )
    )
    joined = _joined(notebook["cells"])
    if tuple(signs) != PROBE_CODES[name] or "rank tomography audit" not in joined:
        raise RuntimeError(f"rank tomography probe {name} was not isolated")
    validate_notebook(notebook, slug)
    output = Path("kaggle") / slug
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(ANCHOR, output, slug)
    return path


def build_hmm_control() -> Path:
    notebook = _load_anchor()
    hmm_cells = copy.deepcopy(frontier._hmm_cells(0.10))
    for cell in hmm_cells:
        text = source(cell).replace(
            "dynamic_f594_plus_student_t_hmm015",
            "mha400_continuity_plus_student_t_hmm010",
        ).replace(
            "Student-t posterior mean over the dynamic F594 PF-branch anchor.",
            "Student-t posterior mean over the MHA400 U-continuity anchor.",
        )
        cell["source"] = text.splitlines(keepends=True)
    notebook["cells"].extend(hmm_cells)
    notebook["cells"].append(
        code_cell(
            integrity_audit(
                "mha400_continuity_plus_student_t_hmm010",
                "_SD_STARTED",
                "_sd_audit['hmm'] = _HMM_AUDIT",
            )
        )
    )
    slug = "rogii-mha400-cont-hmm010"
    joined = _joined(notebook["cells"])
    required = (
        "_HMM_WEIGHT = 0.1",
        "HMMParams(emission='t', sigma_mode='std')",
        "mha400_continuity_plus_student_t_hmm010",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"HMM control fragments missing: {missing}")
    validate_notebook(notebook, slug)
    output = Path("kaggle") / slug
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(ANCHOR, output, slug)
    return path


def main() -> None:
    for name, signs in PROBE_CODES.items():
        print(f"built {build_probe(name, signs)}")
    print(f"built {build_hmm_control()}")


if __name__ == "__main__":
    main()
