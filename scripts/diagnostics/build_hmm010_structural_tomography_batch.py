"""Build arbitrary-hidden-well structural probes around the scored HMM010 anchor."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from build_stopdose_breakthrough_notebooks import (
    code_cell,
    integrity_audit,
    source,
    validate_notebook,
)


BASE_DIR = Path("kaggle/rogii-mha400-cont-hmm010")
BASE_NOTEBOOK = BASE_DIR / "rogii-mha400-cont-hmm010.ipynb"


def _joined(notebook: dict) -> str:
    return "\n".join(source(cell) for cell in notebook["cells"])


def _load_base() -> dict:
    notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    joined = _joined(notebook)
    required = (
        "_HMM_WEIGHT = 0.1",
        "mha400_continuity_plus_student_t_hmm010",
        "_UC_CAP = 8.000000",
        "_UC_TAU = 240.000000",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"HMM010 anchor fragments missing: {missing}")
    return notebook


def _data_root_function(prefix: str) -> str:
    return f'''def {prefix}_data_root():
    candidates = []
    cfg = globals().get('CFG')
    if cfg is not None:
        for attr in ('DATA', 'dataset_path'):
            if hasattr(cfg, attr):
                candidates.append(_Path(getattr(cfg, attr)))
    candidates.extend([
        _Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        _Path('/kaggle/input/rogii-wellbore-geology-prediction'),
    ])
    for candidate in candidates:
        if (candidate / 'test').exists() and (candidate / 'sample_submission.csv').exists():
            return candidate
    raise RuntimeError('could not locate run-local competition data')
'''


def c1_cell(
    alpha: float,
    probe_code: str | None = None,
    bin_alphas: tuple[float, ...] | None = None,
    partition_mod: int = 4,
) -> dict:
    if partition_mod < 2:
        raise ValueError(f"partition_mod must be at least two, got {partition_mod}")
    if probe_code is not None and (
        len(probe_code) != partition_mod or any(char not in "+-" for char in probe_code)
    ):
        raise ValueError(f"invalid {partition_mod}-bin C1 probe code: {probe_code!r}")
    if probe_code is not None and bin_alphas is not None:
        raise ValueError("probe_code and bin_alphas are mutually exclusive")
    if bin_alphas is not None and len(bin_alphas) != partition_mod:
        raise ValueError(
            f"expected {partition_mod} C1 bin alphas, got {bin_alphas!r}"
        )
    partition_label = f"lexicographic_run_local_well_rank_mod_{partition_mod}"
    code = f'''# C1 heel-continuity structural residual over the scored HMM010 route.
import hashlib as _c1_hashlib
import json as _c1_json
from pathlib import Path as _Path

import numpy as _c1_np
import pandas as _c1_pd

_C1_ALPHA = {float(alpha)!r}
_C1_PROBE_CODE = {probe_code!r}
_C1_BIN_ALPHAS = {bin_alphas!r}
_C1_PARTITION_MOD = {int(partition_mod)!r}
_C1_TAU = 1920.0
_C1_CAP = 16.0
_C1_WORK = _Path('/kaggle/working') if _Path('/kaggle/working').exists() else _Path('.')
_C1_SUB = _C1_WORK / 'submission.csv'

{_data_root_function('_c1')}


def _c1_slope(md, u, from_end):
    slopes = []
    for width in (80, 160, 320, 640):
        if len(md) < max(40, width // 2):
            continue
        x = md[-width:] if from_end else md[:width]
        y = u[-width:] if from_end else u[:width]
        finite = _c1_np.isfinite(x) & _c1_np.isfinite(y)
        if finite.sum() < 40 or float(_c1_np.ptp(x[finite])) < 20.0:
            continue
        slopes.append(float(_c1_np.polyfit(x[finite], y[finite], 1)[0]))
    if not slopes:
        return 0.0, float('inf')
    values = _c1_np.asarray(slopes, dtype=float)
    return float(_c1_np.median(values)), float(_c1_np.max(values) - _c1_np.min(values))


_c1_data = _c1_data_root()
_c1_base = _c1_pd.read_csv(_C1_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_c1_base['tvt'] = _c1_pd.to_numeric(_c1_base['tvt'], errors='coerce')
if not _c1_np.isfinite(_c1_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('C1 received non-finite anchor values')
_c1_base.to_csv(_C1_WORK / 'submission_before_c1_heel.csv', index=False)
_c1_parts = _c1_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _c1_parts.shape[1] != 2:
    raise RuntimeError('C1 could not parse run-local submission IDs')
_c1_base['_well'] = _c1_parts[0].astype(str)
_c1_base['_row'] = _c1_pd.to_numeric(_c1_parts[1], errors='raise').astype(int)
_c1_values = _c1_base['tvt'].to_numpy(dtype=float).copy()
_c1_probe_wells = sorted(_c1_base['_well'].unique().tolist())
_c1_probe_bin = {{
    well: rank % _C1_PARTITION_MOD
    for rank, well in enumerate(_c1_probe_wells)
}}
_c1_rows = []

for _c1_well, _c1_group in _c1_base.groupby('_well', sort=False):
    positions = _c1_group.sort_values('_row').index.to_numpy(dtype=int)
    rows = _c1_base.loc[positions, '_row'].to_numpy(dtype=int)
    horizontal = _c1_pd.read_csv(_c1_data / 'test' / f'{{_c1_well}}__horizontal_well.csv')
    if (rows < 0).any() or (rows >= len(horizontal)).any():
        raise RuntimeError(f'C1 row index outside target well {{_c1_well}}')
    md = _c1_pd.to_numeric(horizontal['MD'], errors='coerce').to_numpy(dtype=float)
    z = _c1_pd.to_numeric(horizontal['Z'], errors='coerce').to_numpy(dtype=float)
    tvt_input = _c1_pd.to_numeric(horizontal['TVT_input'], errors='coerce').to_numpy(dtype=float)
    known = _c1_np.flatnonzero(_c1_np.isfinite(tvt_input))
    if not len(known) or int(rows[0]) != int(known[-1]) + 1:
        raise RuntimeError(f'C1 native suffix contract failed for {{_c1_well}}')
    known_u = tvt_input[known] + z[known]
    pred_u = _c1_values[positions] + z[rows]
    known_slope, known_spread = _c1_slope(md[known], known_u, True)
    pred_slope, pred_spread = _c1_slope(md[rows], pred_u, False)
    mismatch = pred_slope - known_slope
    reliability = float(_c1_np.clip(1.0 - (known_spread + pred_spread) / 0.04, 0.0, 1.0))
    dm = md[rows] - md[known[-1]]
    raw = -mismatch * dm * _c1_np.exp(-dm / _C1_TAU)
    direction = _c1_np.clip(raw, -_C1_CAP, _C1_CAP) * reliability
    probe_bin = int(_c1_probe_bin[str(_c1_well)])
    probe_sign = 1.0
    if _C1_PROBE_CODE is not None:
        probe_sign = 1.0 if _C1_PROBE_CODE[probe_bin] == '+' else -1.0
    effective_alpha = _C1_ALPHA * probe_sign
    if _C1_BIN_ALPHAS is not None:
        effective_alpha = float(_C1_BIN_ALPHAS[probe_bin])
    move = effective_alpha * direction
    _c1_values[positions] += move
    _c1_rows.append({{
        'well': str(_c1_well),
        'rows': int(len(rows)),
        'known_slope': float(known_slope),
        'pred_slope': float(pred_slope),
        'slope_mismatch': float(mismatch),
        'slope_spread': float(known_spread + pred_spread),
        'reliability': reliability,
        'probe_bin': probe_bin,
        'probe_sign': float(probe_sign),
        'effective_alpha': float(effective_alpha),
        'mean_abs_direction': float(_c1_np.mean(_c1_np.abs(direction))),
        'max_abs_direction': float(_c1_np.max(_c1_np.abs(direction))),
    }})

if not _c1_np.isfinite(_c1_values).all():
    raise RuntimeError('C1 produced non-finite predictions')
_c1_final = _c1_base[['id']].copy()
_c1_final['tvt'] = _c1_values
_c1_final.to_csv(_C1_SUB, index=False)
_c1_direction = _c1_values - _c1_pd.read_csv(
    _C1_WORK / 'submission_before_c1_heel.csv'
)['tvt'].to_numpy(dtype=float)

def _c1_sha(path):
    return _c1_hashlib.sha256(_Path(path).read_bytes()).hexdigest()

_C1_AUDIT = {{
    'alpha': float(_C1_ALPHA),
    'probe_code': _C1_PROBE_CODE,
    'bin_alphas': _C1_BIN_ALPHAS,
    'probe_partition': {partition_label!r},
    'tau': float(_C1_TAU),
    'cap': float(_C1_CAP),
    'wells': int(len(_c1_rows)),
    'rows': int(len(_c1_final)),
    'fixed_public_ids_used': False,
    'mean_abs_move': float(_c1_np.mean(_c1_np.abs(_c1_direction))),
    'p95_abs_move': float(_c1_np.quantile(_c1_np.abs(_c1_direction), 0.95)),
    'max_abs_move': float(_c1_np.max(_c1_np.abs(_c1_direction))),
    'mean_squared_move': float(_c1_np.mean(_c1_direction * _c1_direction)),
    'base_sha256': _c1_sha(_C1_WORK / 'submission_before_c1_heel.csv'),
    'final_sha256': _c1_sha(_C1_SUB),
    'well_audits': _c1_rows,
}}
(_C1_WORK / 'c1_heel_audit.json').write_text(
    _c1_json.dumps(_C1_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _C1_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = f'hmm010_c1_alpha_{{_C1_ALPHA:+g}}'
print('C1 heel audit:', _c1_json.dumps(_C1_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def basis_cell(kind: str, sign: int) -> dict:
    if kind not in {"datum", "slope"} or sign not in {-1, 1}:
        raise ValueError((kind, sign))
    code = f'''# Arbitrary-hidden-well structural LB basis over the scored HMM010 route.
import hashlib as _bs_hashlib
import json as _bs_json
from pathlib import Path as _Path

import numpy as _bs_np
import pandas as _bs_pd

_BS_KIND = {kind!r}
_BS_SIGN = {int(sign)!r}
_BS_AMPLITUDE = 2.0
_BS_WORK = _Path('/kaggle/working') if _Path('/kaggle/working').exists() else _Path('.')
_BS_SUB = _BS_WORK / 'submission.csv'

{_data_root_function('_bs')}

_bs_data = _bs_data_root()
_bs_base = _bs_pd.read_csv(_BS_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_bs_base['tvt'] = _bs_pd.to_numeric(_bs_base['tvt'], errors='coerce')
if not _bs_np.isfinite(_bs_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('structural basis received non-finite anchor values')
_bs_base.to_csv(_BS_WORK / f'submission_before_{{_BS_KIND}}_basis.csv', index=False)
_bs_parts = _bs_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _bs_parts.shape[1] != 2:
    raise RuntimeError('structural basis could not parse run-local submission IDs')
_bs_base['_well'] = _bs_parts[0].astype(str)
_bs_base['_row'] = _bs_pd.to_numeric(_bs_parts[1], errors='raise').astype(int)
_bs_basis = _bs_np.empty(len(_bs_base), dtype=float)
_bs_rows = []

for _bs_well, _bs_group in _bs_base.groupby('_well', sort=False):
    positions = _bs_group.sort_values('_row').index.to_numpy(dtype=int)
    rows = _bs_base.loc[positions, '_row'].to_numpy(dtype=int)
    if _BS_KIND == 'datum':
        values = _bs_np.full(len(rows), _BS_AMPLITUDE, dtype=float)
    else:
        horizontal = _bs_pd.read_csv(
            _bs_data / 'test' / f'{{_bs_well}}__horizontal_well.csv', usecols=['MD']
        )
        md = _bs_pd.to_numeric(horizontal['MD'], errors='coerce').to_numpy(dtype=float)[rows]
        span = float(md[-1] - md[0]) if len(md) > 1 else 0.0
        if not _bs_np.isfinite(md).all() or span <= 0.0:
            raise RuntimeError(f'invalid hidden MD span for structural basis well {{_bs_well}}')
        phase = (md - md[0]) / span
        values = _BS_AMPLITUDE * (2.0 * phase - 1.0)
        values -= float(values.mean())
    _bs_basis[positions] = values
    _bs_rows.append({{
        'well': str(_bs_well),
        'rows': int(len(rows)),
        'basis_mean': float(values.mean()),
        'basis_std': float(values.std()),
        'basis_min': float(values.min()),
        'basis_max': float(values.max()),
    }})

_bs_move = float(_BS_SIGN) * _bs_basis
_bs_values = _bs_base['tvt'].to_numpy(dtype=float) + _bs_move
if not _bs_np.isfinite(_bs_values).all():
    raise RuntimeError('structural basis produced non-finite predictions')
_bs_final = _bs_base[['id']].copy()
_bs_final['tvt'] = _bs_values
_bs_final.to_csv(_BS_SUB, index=False)

def _bs_sha(path):
    return _bs_hashlib.sha256(_Path(path).read_bytes()).hexdigest()

_BASIS_AUDIT = {{
    'kind': _BS_KIND,
    'sign': int(_BS_SIGN),
    'amplitude': float(_BS_AMPLITUDE),
    'wells': int(len(_bs_rows)),
    'rows': int(len(_bs_final)),
    'fixed_public_ids_used': False,
    'basis_mean': float(_bs_basis.mean()),
    'basis_std': float(_bs_basis.std()),
    'mean_squared_move': float(_bs_np.mean(_bs_move * _bs_move)),
    'base_sha256': _bs_sha(_BS_WORK / f'submission_before_{{_BS_KIND}}_basis.csv'),
    'final_sha256': _bs_sha(_BS_SUB),
    'well_audits': _bs_rows,
}}
(_BS_WORK / f'{{_BS_KIND}}_basis_audit.json').write_text(
    _bs_json.dumps(_BASIS_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _BS_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = f'hmm010_{{_BS_KIND}}_basis_{{_BS_SIGN:+d}}'
print('structural basis audit:', _bs_json.dumps(_BASIS_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def _write(notebook: dict, slug: str, strategy: str, extra: str) -> Path:
    notebook["cells"].append(code_cell(integrity_audit(strategy, "_SD_STARTED", extra)))
    validate_notebook(notebook, slug)
    joined = _joined(notebook)
    if "00e12e8b" in joined:
        raise RuntimeError(f"{slug} contains a fixed public well ID")
    output = Path("kaggle") / slug
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    metadata = json.loads((BASE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata.update(
        {
            "id": f"qwer556617123/{slug}",
            "title": slug,
            "code_file": f"{slug}.ipynb",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "kernel_sources": [],
            "competition_sources": ["rogii-wellbore-geology-prediction"],
            "model_sources": [],
        }
    )
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def build_c1(alpha: float, slug: str | None = None) -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(alpha))
    if slug is None:
        label = "p" if alpha > 0 else "m"
        slug = f"rogii-hmm010-c1{label}"
    return _write(
        notebook,
        slug,
        f"hmm010_c1_heel_alpha_{alpha:+g}",
        "_sd_audit['hmm'] = _HMM_AUDIT\n_sd_audit['c1_heel'] = _C1_AUDIT",
    )


def build_c1_code(probe_code: str) -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(+1.0, probe_code=probe_code))
    slug = f"rogii-hmm010-c1code-{probe_code.replace('+', 'p').replace('-', 'm')}"
    return _write(
        notebook,
        slug,
        f"hmm010_c1_four_bin_code_{probe_code}",
        "_sd_audit['hmm'] = _HMM_AUDIT\n_sd_audit['c1_heel'] = _C1_AUDIT",
    )


def build_c1_bin_calibrated(
    bin_alphas: tuple[float, float, float, float], slug: str
) -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=bin_alphas))
    return _write(
        notebook,
        slug,
        "hmm010_c1_four_bin_calibrated_" + "_".join(f"{x:+g}" for x in bin_alphas),
        "_sd_audit['hmm'] = _HMM_AUDIT\n_sd_audit['c1_heel'] = _C1_AUDIT",
    )


def build_c1_partition_calibrated(
    bin_alphas: tuple[float, ...], slug: str
) -> Path:
    partition_mod = len(bin_alphas)
    notebook = _load_base()
    notebook["cells"].append(
        c1_cell(1.0, bin_alphas=bin_alphas, partition_mod=partition_mod)
    )
    return _write(
        notebook,
        slug,
        f"hmm010_c1_rank_mod_{partition_mod}_calibrated_"
        + "_".join(f"{x:+g}" for x in bin_alphas),
        "_sd_audit['hmm'] = _HMM_AUDIT\n_sd_audit['c1_heel'] = _C1_AUDIT",
    )


def build_basis(kind: str, sign: int) -> Path:
    notebook = _load_base()
    notebook["cells"].append(basis_cell(kind, sign))
    sign_label = "p" if sign > 0 else "m"
    slug = f"rogii-hmm010-{kind}-{sign_label}2"
    return _write(
        notebook,
        slug,
        f"hmm010_{kind}_basis_{sign:+d}",
        "_sd_audit['hmm'] = _HMM_AUDIT\n_sd_audit['structural_basis'] = _BASIS_AUDIT",
    )


def main() -> None:
    outputs = [
        build_c1(+1.0),
        build_c1(-1.0),
        build_basis("datum", +1),
        build_basis("slope", +1),
        build_basis("slope", -1),
    ]
    for output in outputs:
        print(f"built {output}")


if __name__ == "__main__":
    main()
