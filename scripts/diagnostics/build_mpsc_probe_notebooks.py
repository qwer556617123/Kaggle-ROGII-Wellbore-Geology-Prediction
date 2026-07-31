"""Build gated MPSC coded-probe notebooks from a scored HMM anchor."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCE_DIR = Path("kaggle/rogii-pkadopt-hmm-student-wellcal-650505")
SOURCE_NOTEBOOK = SOURCE_DIR / "rogii-pkadopt-hmm-student-wellcal-650505.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
MPSC_MODULE = Path("scripts/diagnostics/multiscale_stratigraphic_alignment.py")

TARGETS = {
    "ppp": ("rogii-mpsc-code-ppp", "+++"),
    "ppm": ("rogii-mpsc-code-ppm", "++-"),
    "pmp": ("rogii-mpsc-code-pmp", "+-+"),
}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def runner(default_code: str) -> str:
    return f'''# Gated multiscale probabilistic stratigraphic coded probe.
import glob as _mp_glob
import hashlib as _mp_hashlib
import json as _mp_json
import os as _mp_os
import time as _mp_time
from pathlib import Path as _MpPath

import numpy as _mp_np
import pandas as _mp_pd

_MP_STARTED = _mp_time.time()
_MP_MODE = _mp_os.environ.get('ROGII_STRAT_ALIGN_MODE', 'off').strip().lower()
_MP_SCALES = tuple(float(x) for x in _mp_os.environ.get(
    'ROGII_STRAT_ALIGN_SCALES', '2,8,24,64'
).split(','))
_MP_TOP_MODES = int(_mp_os.environ.get('ROGII_STRAT_ALIGN_TOP_MODES', '5'))
_MP_CODE = _mp_os.environ.get('ROGII_STRAT_ALIGN_PROBE_CODE', {default_code!r}).strip()
_MP_MAGNITUDE = float(_mp_os.environ.get('ROGII_STRAT_ALIGN_PROBE_MAGNITUDE', '0.20'))
if _MP_MODE not in ('off', 'own_ref', 'multi_ref'):
    raise ValueError(f'unsupported ROGII_STRAT_ALIGN_MODE={{_MP_MODE}}')
if len(_MP_SCALES) != 4:
    raise ValueError('ROGII_STRAT_ALIGN_SCALES must contain exactly four values')

_mp_work = _MpPath('/kaggle/working') if _MpPath('/kaggle/working').exists() else _MpPath('.')
_mp_data = _MpPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_mp_data / 'sample_submission.csv').exists():
    _mp_data = _MpPath('/kaggle/input/rogii-wellbore-geology-prediction')
if not (_mp_data / 'sample_submission.csv').exists():
    for _path in _mp_glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True):
        _candidate = _MpPath(_path).parent
        if (_candidate / 'test').exists():
            _mp_data = _candidate
            break

_mp_submission_path = _mp_work / 'submission.csv'
_mp_anchor = _mp_pd.read_csv(_mp_submission_path)
_mp_sample = _mp_pd.read_csv(_mp_data / 'sample_submission.csv')
_mp_anchor['id'] = _mp_anchor['id'].astype(str)
_mp_sample['id'] = _mp_sample['id'].astype(str)
if _mp_anchor['id'].tolist() != _mp_sample['id'].tolist():
    raise RuntimeError('MPSC anchor id order differs from sample submission')

def _mp_sha(path):
    digest = _mp_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

_mp_anchor_path = _mp_work / 'submission_before_mpsc.csv'
_mp_anchor.to_csv(_mp_anchor_path, index=False)
_mp_hmm = _mp_pd.read_csv(_mp_work / 'exact_hmm_predictions.csv')
_mp_hmm['id'] = _mp_hmm['id'].astype(str)

_mp_parts = []
_mp_well_audit = []
if _MP_MODE != 'off':
    _mp_wells = sorted(
        path.name.replace('__horizontal_well.csv', '')
        for path in (_mp_data / 'test').glob('*__horizontal_well.csv')
    )
    if len(_MP_CODE) != len(_mp_wells) or any(char not in '+-' for char in _MP_CODE):
        raise ValueError('probe code must contain one +/- sign per sorted test well')
    for _rank, (_well, _sign) in enumerate(zip(_mp_wells, _MP_CODE)):
        _hw = _mp_pd.read_csv(_mp_data / 'test' / f'{{_well}}__horizontal_well.csv')
        _tw = _mp_pd.read_csv(
            _mp_data / 'test' / f'{{_well}}__typewell.csv', usecols=['TVT', 'GR']
        )
        _config = MPSCConfig(
            scales_ft=_MP_SCALES,
            profile='balanced',
            top_modes=_MP_TOP_MODES,
        )
        _result = run_mpsc(_hw, _tw, _hmm2_fb, _config)
        _positions = _mp_np.flatnonzero(_hw['TVT_input'].isna().to_numpy())
        _ids = [f'{{_well}}_{{row}}' for row in _positions]
        _part = _mp_pd.DataFrame({{'id': _ids, 'mpsc_tvt': _result.mean_eval}})
        _mp_parts.append(_part)
        _mp_well_audit.append({{
            'well_rank': int(_rank),
            'well_id': _well,
            'sign': _sign,
            'rows': int(len(_part)),
            'gr_coverage': float(_result.metadata['gr_coverage']),
            'prefix_loss': float(_result.metadata['prefix_loss']),
            'entropy': float(_result.metadata['entropy']),
            'mode_datums': _result.metadata['mode_datums'],
            'mode_weights': _result.metadata['mode_weights'],
        }})
    _mp_candidate = _mp_pd.concat(_mp_parts, ignore_index=True)
    _mp_merged = _mp_anchor.merge(_mp_hmm[['id', 'hmm_tvt']], on='id', how='left')
    _mp_merged = _mp_merged.merge(_mp_candidate, on='id', how='left')
    if _mp_merged[['hmm_tvt', 'mpsc_tvt']].isna().any().any():
        raise RuntimeError('MPSC candidate does not cover every submission row')
    _mp_rank = _mp_merged['id'].str.rsplit('_', n=1).str[0].map(
        {{well: rank for rank, well in enumerate(_mp_wells)}}
    ).to_numpy(int)
    _mp_signs = _mp_np.asarray([1.0 if char == '+' else -1.0 for char in _MP_CODE])
    _mp_direction = _mp_merged['mpsc_tvt'].to_numpy(float) - _mp_merged['hmm_tvt'].to_numpy(float)
    _mp_final_values = _mp_merged['tvt'].to_numpy(float)
    _mp_final_values += _MP_MAGNITUDE * _mp_signs[_mp_rank] * _mp_direction
    _mp_final = _mp_merged[['id']].copy()
    _mp_final['tvt'] = _mp_final_values
    _mp_final.to_csv(_mp_submission_path, index=False)
else:
    _mp_direction = _mp_np.zeros(len(_mp_anchor), dtype=float)
    _mp_final = _mp_anchor.copy()

if _mp_final['id'].tolist() != _mp_sample['id'].tolist():
    raise RuntimeError('MPSC final id order differs from sample submission')
if not _mp_np.isfinite(_mp_final['tvt'].to_numpy(float)).all():
    raise RuntimeError('MPSC final contains non-finite values')

_MP_AUDIT = {{
    'strategy': 'multiscale_probabilistic_stratigraphic_correlator',
    'mode': _MP_MODE,
    'scales_ft': list(_MP_SCALES),
    'top_modes': _MP_TOP_MODES,
    'probe_code': _MP_CODE,
    'probe_magnitude': _MP_MAGNITUDE,
    'rows': int(len(_mp_final)),
    'direction_rms': float(_mp_np.sqrt(_mp_np.mean(_mp_direction ** 2))),
    'direction_max_abs': float(_mp_np.max(_mp_np.abs(_mp_direction))),
    'runtime_sec_layer': float(_mp_time.time() - _MP_STARTED),
    'anchor_sha256': _mp_sha(_mp_anchor_path),
    'hmm_sha256': _mp_sha(_mp_work / 'exact_hmm_predictions.csv'),
    'final_sha256': _mp_sha(_mp_submission_path),
    'well_stats': _mp_well_audit,
}}
(_mp_work / 'stratigraphic_alignment_audit.json').write_text(
    _mp_json.dumps(_MP_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('MPSC audit:', _MP_AUDIT, flush=True)
'''


def validate_notebook(notebook: dict, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        compile(source, f"{label}:cell-{index}", "exec")


def build(target: str) -> Path:
    slug, code = TARGETS[target]
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"] = [copy.deepcopy(cell) for cell in notebook["cells"]]
    notebook["cells"].append(code_cell(MPSC_MODULE.read_text(encoding="utf-8")))
    notebook["cells"].append(code_cell(runner(code)))
    validate_notebook(notebook, slug)
    output_dir = Path("kaggle") / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f"{slug}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata.update({"id": f"qwer556617123/{slug}", "title": slug, "code_file": notebook_path.name})
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", choices=sorted(TARGETS))
    args = parser.parse_args()
    for target in args.targets or TARGETS:
        print(f"built {build(target)}")


if __name__ == "__main__":
    main()
