"""Build rerun-safe exact-HMM weight follow-up Kaggle notebooks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCE_DIR = Path("kaggle/rogii-pkadopt-hmm015")
SOURCE_NOTEBOOK = SOURCE_DIR / "rogii-pkadopt-hmm015.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"

TARGETS = {
    "hmm0225": {
        "slug": "rogii-pkadopt-hmm0225",
        "weight": 0.225,
        "strategy": "pkadopt_hmm_lb_calibrated_weight_0225",
    },
    "hmm025": {
        "slug": "rogii-pkadopt-hmm025",
        "weight": 0.25,
        "strategy": "pkadopt_hmm_source_oof_weight_025",
    },
    "student025": {
        "slug": "rogii-pkadopt-hmm-student025",
        "weight": 0.25,
        "strategy": "pkadopt_student_t_hmm_weight_025",
        "student_t": True,
    },
    "student_shape050": {
        "slug": "rogii-pkadopt-hmm-student-shape050",
        "weight": 0.25,
        "mean_weight": 0.25,
        "shape_weight": 0.50,
        "strategy": "pkadopt_student_t_hmm_mean025_shape050",
        "student_t": True,
        "native_wells": 30,
        "native_reference_row_rmse": 12.481066659052473,
        "native_candidate_row_rmse": 12.264785410315088,
        "public_reference_score": 6.909,
    },
    "student_lam125_shape050": {
        "slug": "rogii-pkadopt-hmm-student-lam125-shape050",
        "weight": 0.25,
        "mean_weight": 0.0,
        "shape_weight": 0.50,
        "strategy": "pkadopt_student_t_lam125_hmm_mean000_shape050",
        "student_t": True,
        "hmm_params": "HMMParams(emission='t', sigma_mode='std', lam=1.25)",
        "native_wells": 12,
        "native_reference_row_rmse": 9.644562129398759,
        "native_candidate_row_rmse": 9.161104474800608,
        "public_reference_score": 6.909,
    },
    "student_shape030_mean000": {
        "slug": "rogii-pkadopt-hmm-student-shape030-mean000",
        "weight": 0.25,
        "mean_weight": 0.0,
        "shape_weight": 0.30,
        "strategy": "pkadopt_student_t_hmm_mean000_shape030_lb_fit",
        "student_t": True,
        "native_wells": 30,
        "native_reference_row_rmse": 12.481066659052473,
        "native_candidate_row_rmse": None,
        "public_reference_score": 6.909,
    },
    "student_stdgate035": {
        "slug": "rogii-pkadopt-hmm-student-stdgate035",
        "weight": 0.25,
        "strategy": "pkadopt_student_t_hmm_row_std_gate_cap035_floor010",
        "student_t": True,
        "uncertainty_gate": {
            "cap": 0.35,
            "floor": 0.10,
            "low": 2.0,
            "high": 6.0,
        },
        "native_wells": 30,
        "native_reference_row_rmse": 12.481066659052473,
        "native_candidate_row_rmse": 12.337051952969997,
        "native_reference_well_p90": 22.180051619509133,
        "native_candidate_well_p90": 20.207594521032004,
        "public_reference_score": 6.909,
    },
    "student_code_ppm": {
        "slug": "rogii-pkadopt-hmm-student-code-ppm",
        "weight": 0.25,
        "strategy": "pkadopt_student_t_hmm_three_well_code_ppm",
        "student_t": True,
        "coded_signs": (1.0, 1.0, -1.0),
        "public_anchor_score": 7.053,
        "public_all_positive_score": 6.909,
    },
    "student_code_pmp": {
        "slug": "rogii-pkadopt-hmm-student-code-pmp-v2",
        "weight": 0.25,
        "strategy": "pkadopt_student_t_hmm_three_well_code_pmp",
        "student_t": True,
        "coded_signs": (1.0, -1.0, 1.0),
        "public_anchor_score": 7.053,
        "public_all_positive_score": 6.909,
    },
    "student_wellcal_650505": {
        "slug": "rogii-pkadopt-hmm-student-wellcal-650505",
        "weight": 0.25,
        "strategy": "pkadopt_student_t_hmm_three_well_calibrated_650505",
        "student_t": True,
        "well_weights": (0.65, 0.05, 0.05),
        "tomography_refs": (54707045, 54766267, 54766337),
        "tomography_scores": (6.909, 6.966, 6.963),
        "predicted_public_score": 6.663617510270342,
    },
}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def followup_cell(weight: float, strategy: str) -> str:
    return f'''# Reweight the rerun-generated exact HMM direction.
import hashlib as _fu_hashlib
import json as _fu_json
from pathlib import Path as _FuPath
import numpy as _fu_np
import pandas as _fu_pd

_FU_WEIGHT = {weight!r}
_fu_work = _FuPath('/kaggle/working') if _FuPath('/kaggle/working').exists() else _FuPath('.')
_fu_data = _FuPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_fu_data / 'sample_submission.csv').exists():
    _fu_data = _FuPath('/kaggle/input/rogii-wellbore-geology-prediction')

_fu_base = _fu_pd.read_csv(_fu_work / 'submission_before_exact_hmm.csv')[['id', 'tvt']]
_fu_hmm = _fu_pd.read_csv(_fu_work / 'exact_hmm_predictions.csv')[['id', 'hmm_tvt', 'hmm_std']]
_fu_sample = _fu_pd.read_csv(_fu_data / 'sample_submission.csv')[['id']]
for _frame in (_fu_base, _fu_hmm, _fu_sample):
    _frame['id'] = _frame['id'].astype(str)
_fu_merged = _fu_base.merge(_fu_hmm, on='id', how='left', validate='one_to_one')
if _fu_merged['hmm_tvt'].isna().any():
    raise RuntimeError('follow-up HMM vector does not cover every row')
_fu_base_values = _fu_merged['tvt'].to_numpy(dtype=float)
_fu_direction = _fu_merged['hmm_tvt'].to_numpy(dtype=float) - _fu_base_values
_fu_values = _fu_base_values + _FU_WEIGHT * _fu_direction
if not _fu_np.isfinite(_fu_values).all():
    raise RuntimeError('non-finite follow-up HMM values')
_fu_final = _fu_merged[['id']].copy()
_fu_final['tvt'] = _fu_values
if not _fu_final['id'].equals(_fu_sample['id']):
    raise RuntimeError('follow-up HMM id order mismatch')
_fu_final.to_csv(_fu_work / 'submission.csv', index=False)

def _fu_sha(path):
    return _fu_hashlib.sha256(_FuPath(path).read_bytes()).hexdigest()

_fu_delta = _fu_values - _fu_base_values
_FU_AUDIT = {{
    'strategy': {strategy!r},
    'weight': float(_FU_WEIGHT),
    'rows': int(len(_fu_final)),
    'id_order_matches_sample': True,
    'mean_abs_move': float(_fu_np.mean(_fu_np.abs(_fu_delta))),
    'rms_move': float(_fu_np.sqrt(_fu_np.mean(_fu_delta ** 2))),
    'p95_abs_move': float(_fu_np.quantile(_fu_np.abs(_fu_delta), 0.95)),
    'max_abs_move': float(_fu_np.max(_fu_np.abs(_fu_delta))),
    'posterior_std_mean': float(_fu_merged['hmm_std'].mean()),
    'base_sha256': _fu_sha(_fu_work / 'submission_before_exact_hmm.csv'),
    'final_sha256': _fu_sha(_fu_work / 'submission.csv'),
    'lb_anchor_score': 7.053,
    'lb_hmm015_score': 6.935,
    'lb_quadratic_optimum_visible_norm': 0.223246294845859,
}}
(_fu_work / 'hmm_followup_audit.json').write_text(
    _fu_json.dumps(_FU_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('HMM follow-up audit:', _FU_AUDIT, flush=True)
'''


def decomposition_cell(
    mean_weight: float,
    shape_weight: float,
    strategy: str,
    native_wells: int,
    native_reference_row_rmse: float,
    native_candidate_row_rmse: float,
    public_reference_score: float,
) -> str:
    return f'''# Reweight HMM datum and within-well shape as separately validated components.
import hashlib as _ds_hashlib
import json as _ds_json
from pathlib import Path as _DsPath
import numpy as _ds_np
import pandas as _ds_pd

_DS_MEAN_WEIGHT = {mean_weight!r}
_DS_SHAPE_WEIGHT = {shape_weight!r}
_ds_work = _DsPath('/kaggle/working') if _DsPath('/kaggle/working').exists() else _DsPath('.')
_ds_data = _DsPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_ds_data / 'sample_submission.csv').exists():
    _ds_data = _DsPath('/kaggle/input/rogii-wellbore-geology-prediction')

_ds_base = _ds_pd.read_csv(_ds_work / 'submission_before_exact_hmm.csv')[['id', 'tvt']]
_ds_hmm = _ds_pd.read_csv(_ds_work / 'exact_hmm_predictions.csv')[['id', 'hmm_tvt', 'hmm_std']]
_ds_sample = _ds_pd.read_csv(_ds_data / 'sample_submission.csv')[['id']]
for _frame in (_ds_base, _ds_hmm, _ds_sample):
    _frame['id'] = _frame['id'].astype(str)
_ds_merged = _ds_base.merge(_ds_hmm, on='id', how='left', validate='one_to_one')
if _ds_merged['hmm_tvt'].isna().any():
    raise RuntimeError('decomposed HMM vector does not cover every row')
_ds_base_values = _ds_merged['tvt'].to_numpy(dtype=float)
_ds_direction = _ds_merged['hmm_tvt'].to_numpy(dtype=float) - _ds_base_values
_ds_well = _ds_merged['id'].str.rsplit('_', n=1).str[0]
_ds_mean_direction = _ds_pd.Series(_ds_direction).groupby(_ds_well, sort=False).transform('mean').to_numpy()
_ds_shape_direction = _ds_direction - _ds_mean_direction
_ds_delta = _DS_MEAN_WEIGHT * _ds_mean_direction + _DS_SHAPE_WEIGHT * _ds_shape_direction
_ds_values = _ds_base_values + _ds_delta
if not _ds_np.isfinite(_ds_values).all():
    raise RuntimeError('non-finite decomposed HMM values')
_ds_final = _ds_merged[['id']].copy()
_ds_final['tvt'] = _ds_values
if not _ds_final['id'].equals(_ds_sample['id']):
    raise RuntimeError('decomposed HMM id order mismatch')
_ds_final.to_csv(_ds_work / 'submission.csv', index=False)

def _ds_sha(path):
    return _ds_hashlib.sha256(_DsPath(path).read_bytes()).hexdigest()

_DS_AUDIT = {{
    'strategy': {strategy!r},
    'mean_weight': float(_DS_MEAN_WEIGHT),
    'shape_weight': float(_DS_SHAPE_WEIGHT),
    'rows': int(len(_ds_final)),
    'id_order_matches_sample': True,
    'mean_abs_move': float(_ds_np.mean(_ds_np.abs(_ds_delta))),
    'rms_move': float(_ds_np.sqrt(_ds_np.mean(_ds_delta ** 2))),
    'p95_abs_move': float(_ds_np.quantile(_ds_np.abs(_ds_delta), 0.95)),
    'max_abs_move': float(_ds_np.max(_ds_np.abs(_ds_delta))),
    'mean_component_rms': float(_ds_np.sqrt(_ds_np.mean((_DS_MEAN_WEIGHT * _ds_mean_direction) ** 2))),
    'shape_component_rms': float(_ds_np.sqrt(_ds_np.mean((_DS_SHAPE_WEIGHT * _ds_shape_direction) ** 2))),
    'posterior_std_mean': float(_ds_merged['hmm_std'].mean()),
    'base_sha256': _ds_sha(_ds_work / 'submission_before_exact_hmm.csv'),
    'hmm_sha256': _ds_sha(_ds_work / 'exact_hmm_predictions.csv'),
    'final_sha256': _ds_sha(_ds_work / 'submission.csv'),
    'native_mask_wells': {native_wells!r},
    'native_mask_reference_row_rmse': {native_reference_row_rmse!r},
    'native_mask_candidate_row_rmse': {native_candidate_row_rmse!r},
    'lb_student_reference_score': {public_reference_score!r},
}}
(_ds_work / 'hmm_decomposition_audit.json').write_text(
    _ds_json.dumps(_DS_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('HMM decomposition audit:', _DS_AUDIT, flush=True)
'''


def uncertainty_gate_cell(config: dict) -> str:
    gate = config["uncertainty_gate"]
    return f'''# Confidence-gate the Student-t HMM direction using its own posterior std.
import hashlib as _ug_hashlib
import json as _ug_json
from pathlib import Path as _UgPath
import numpy as _ug_np
import pandas as _ug_pd

_UG_CAP = {gate["cap"]!r}
_UG_FLOOR = {gate["floor"]!r}
_UG_LOW = {gate["low"]!r}
_UG_HIGH = {gate["high"]!r}
_ug_work = _UgPath('/kaggle/working') if _UgPath('/kaggle/working').exists() else _UgPath('.')
_ug_data = _UgPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_ug_data / 'sample_submission.csv').exists():
    _ug_data = _UgPath('/kaggle/input/rogii-wellbore-geology-prediction')

_ug_base = _ug_pd.read_csv(_ug_work / 'submission_before_exact_hmm.csv')[['id', 'tvt']]
_ug_hmm = _ug_pd.read_csv(_ug_work / 'exact_hmm_predictions.csv')[['id', 'hmm_tvt', 'hmm_std']]
_ug_sample = _ug_pd.read_csv(_ug_data / 'sample_submission.csv')[['id']]
for _frame in (_ug_base, _ug_hmm, _ug_sample):
    _frame['id'] = _frame['id'].astype(str)
_ug_merged = _ug_base.merge(_ug_hmm, on='id', how='left', validate='one_to_one')
if _ug_merged[['hmm_tvt', 'hmm_std']].isna().any().any():
    raise RuntimeError('uncertainty gate HMM vector is incomplete')
_ug_base_values = _ug_merged['tvt'].to_numpy(dtype=float)
_ug_hmm_values = _ug_merged['hmm_tvt'].to_numpy(dtype=float)
_ug_std = _ug_merged['hmm_std'].to_numpy(dtype=float)
_ug_confidence = _ug_np.clip((_UG_HIGH - _ug_std) / (_UG_HIGH - _UG_LOW), 0.0, 1.0)
_ug_weight = _UG_FLOOR + (_UG_CAP - _UG_FLOOR) * _ug_confidence
_ug_delta = _ug_weight * (_ug_hmm_values - _ug_base_values)
_ug_values = _ug_base_values + _ug_delta
if not _ug_np.isfinite(_ug_values).all():
    raise RuntimeError('non-finite uncertainty-gated HMM values')
_ug_final = _ug_merged[['id']].copy()
_ug_final['tvt'] = _ug_values
if not _ug_final['id'].equals(_ug_sample['id']):
    raise RuntimeError('uncertainty-gated HMM id order mismatch')
_ug_final.to_csv(_ug_work / 'submission.csv', index=False)

def _ug_sha(path):
    return _ug_hashlib.sha256(_UgPath(path).read_bytes()).hexdigest()

_UG_AUDIT = {{
    'strategy': {config["strategy"]!r},
    'cap': float(_UG_CAP),
    'floor': float(_UG_FLOOR),
    'std_low': float(_UG_LOW),
    'std_high': float(_UG_HIGH),
    'rows': int(len(_ug_final)),
    'id_order_matches_sample': True,
    'weight_mean': float(_ug_np.mean(_ug_weight)),
    'weight_p10': float(_ug_np.quantile(_ug_weight, 0.10)),
    'weight_p90': float(_ug_np.quantile(_ug_weight, 0.90)),
    'mean_abs_move': float(_ug_np.mean(_ug_np.abs(_ug_delta))),
    'rms_move': float(_ug_np.sqrt(_ug_np.mean(_ug_delta ** 2))),
    'posterior_std_mean': float(_ug_np.mean(_ug_std)),
    'posterior_std_p90': float(_ug_np.quantile(_ug_std, 0.90)),
    'base_sha256': _ug_sha(_ug_work / 'submission_before_exact_hmm.csv'),
    'hmm_sha256': _ug_sha(_ug_work / 'exact_hmm_predictions.csv'),
    'final_sha256': _ug_sha(_ug_work / 'submission.csv'),
    'native_mask_wells': {config["native_wells"]!r},
    'native_mask_reference_row_rmse': {config["native_reference_row_rmse"]!r},
    'native_mask_candidate_row_rmse': {config["native_candidate_row_rmse"]!r},
    'native_mask_reference_well_p90': {config["native_reference_well_p90"]!r},
    'native_mask_candidate_well_p90': {config["native_candidate_well_p90"]!r},
    'lb_student_reference_score': {config["public_reference_score"]!r},
}}
(_ug_work / 'hmm_uncertainty_gate_audit.json').write_text(
    _ug_json.dumps(_UG_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('HMM uncertainty gate audit:', _UG_AUDIT, flush=True)
'''


def coded_well_cell(config: dict) -> str:
    return f'''# Orthogonally code the three dynamic well HMM directions for LB tomography.
import hashlib as _cw_hashlib
import json as _cw_json
from pathlib import Path as _CwPath
import numpy as _cw_np
import pandas as _cw_pd

_CW_MAGNITUDE = {config["weight"]!r}
_CW_SIGNS = {tuple(config["coded_signs"])!r}
_cw_work = _CwPath('/kaggle/working') if _CwPath('/kaggle/working').exists() else _CwPath('.')
_cw_data = _CwPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_cw_data / 'sample_submission.csv').exists():
    _cw_data = _CwPath('/kaggle/input/rogii-wellbore-geology-prediction')

_cw_base = _cw_pd.read_csv(_cw_work / 'submission_before_exact_hmm.csv')[['id', 'tvt']]
_cw_hmm = _cw_pd.read_csv(_cw_work / 'exact_hmm_predictions.csv')[['id', 'hmm_tvt', 'hmm_std']]
_cw_sample = _cw_pd.read_csv(_cw_data / 'sample_submission.csv')[['id']]
for _frame in (_cw_base, _cw_hmm, _cw_sample):
    _frame['id'] = _frame['id'].astype(str)
_cw_merged = _cw_base.merge(_cw_hmm, on='id', how='left', validate='one_to_one')
if _cw_merged[['hmm_tvt', 'hmm_std']].isna().any().any():
    raise RuntimeError('coded-well HMM vector is incomplete')
_cw_wells = _cw_merged['id'].str.rsplit('_', n=1).str[0]
_cw_unique = sorted(_cw_wells.unique().tolist())
if len(_cw_unique) != len(_CW_SIGNS):
    raise RuntimeError(f'expected {{len(_CW_SIGNS)}} dynamic wells, found {{len(_cw_unique)}}')
_cw_sign_map = dict(zip(_cw_unique, _CW_SIGNS))
_cw_row_sign = _cw_wells.map(_cw_sign_map).to_numpy(dtype=float)
_cw_base_values = _cw_merged['tvt'].to_numpy(dtype=float)
_cw_direction = _cw_merged['hmm_tvt'].to_numpy(dtype=float) - _cw_base_values
_cw_weight = _CW_MAGNITUDE * _cw_row_sign
_cw_delta = _cw_weight * _cw_direction
_cw_values = _cw_base_values + _cw_delta
if not _cw_np.isfinite(_cw_values).all():
    raise RuntimeError('non-finite coded-well HMM values')
_cw_final = _cw_merged[['id']].copy()
_cw_final['tvt'] = _cw_values
if not _cw_final['id'].equals(_cw_sample['id']):
    raise RuntimeError('coded-well HMM id order mismatch')
_cw_final.to_csv(_cw_work / 'submission.csv', index=False)

def _cw_sha(path):
    return _cw_hashlib.sha256(_CwPath(path).read_bytes()).hexdigest()

_cw_well_stats = []
for _wid in _cw_unique:
    _mask = _cw_wells.eq(_wid).to_numpy()
    _part = _cw_direction[_mask]
    _cw_well_stats.append({{
        'well_rank': int(_cw_unique.index(_wid)),
        'well_id': _wid,
        'sign': float(_cw_sign_map[_wid]),
        'rows': int(_mask.sum()),
        'direction_norm2': float(_cw_np.mean(_part ** 2)),
        'direction_sse_per_total_row': float(_cw_np.sum(_part ** 2) / len(_cw_final)),
        'rms_move': float(_cw_np.sqrt(_cw_np.mean((_CW_MAGNITUDE * _part) ** 2))),
        'posterior_std_mean': float(_cw_merged.loc[_mask, 'hmm_std'].mean()),
    }})
_CW_AUDIT = {{
    'strategy': {config["strategy"]!r},
    'magnitude': float(_CW_MAGNITUDE),
    'signs_by_sorted_well': list(_CW_SIGNS),
    'rows': int(len(_cw_final)),
    'id_order_matches_sample': True,
    'well_stats': _cw_well_stats,
    'base_sha256': _cw_sha(_cw_work / 'submission_before_exact_hmm.csv'),
    'hmm_sha256': _cw_sha(_cw_work / 'exact_hmm_predictions.csv'),
    'final_sha256': _cw_sha(_cw_work / 'submission.csv'),
    'public_anchor_score': {config["public_anchor_score"]!r},
    'public_all_positive_score': {config["public_all_positive_score"]!r},
}}
(_cw_work / 'hmm_coded_well_audit.json').write_text(
    _cw_json.dumps(_CW_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('HMM coded-well audit:', _CW_AUDIT, flush=True)
'''


def calibrated_well_cell(config: dict) -> str:
    return f'''# Apply the hash-verified three-well LB tomography solution.
import hashlib as _wc_hashlib
import json as _wc_json
from pathlib import Path as _WcPath
import numpy as _wc_np
import pandas as _wc_pd

_WC_WEIGHTS = {tuple(config["well_weights"])!r}
_wc_work = _WcPath('/kaggle/working') if _WcPath('/kaggle/working').exists() else _WcPath('.')
_wc_data = _WcPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_wc_data / 'sample_submission.csv').exists():
    _wc_data = _WcPath('/kaggle/input/rogii-wellbore-geology-prediction')

_wc_base = _wc_pd.read_csv(_wc_work / 'submission_before_exact_hmm.csv')[['id', 'tvt']]
_wc_hmm = _wc_pd.read_csv(_wc_work / 'exact_hmm_predictions.csv')[['id', 'hmm_tvt', 'hmm_std']]
_wc_sample = _wc_pd.read_csv(_wc_data / 'sample_submission.csv')[['id']]
for _frame in (_wc_base, _wc_hmm, _wc_sample):
    _frame['id'] = _frame['id'].astype(str)
_wc_merged = _wc_base.merge(_wc_hmm, on='id', how='left', validate='one_to_one')
if _wc_merged[['hmm_tvt', 'hmm_std']].isna().any().any():
    raise RuntimeError('calibrated-well HMM vector is incomplete')
_wc_wells = _wc_merged['id'].str.rsplit('_', n=1).str[0]
_wc_unique = sorted(_wc_wells.unique().tolist())
if len(_wc_unique) != len(_WC_WEIGHTS):
    raise RuntimeError(f'expected {{len(_WC_WEIGHTS)}} dynamic wells, found {{len(_wc_unique)}}')
_wc_weight_map = dict(zip(_wc_unique, _WC_WEIGHTS))
_wc_row_weight = _wc_wells.map(_wc_weight_map).to_numpy(dtype=float)
_wc_base_values = _wc_merged['tvt'].to_numpy(dtype=float)
_wc_direction = _wc_merged['hmm_tvt'].to_numpy(dtype=float) - _wc_base_values
_wc_delta = _wc_row_weight * _wc_direction
_wc_values = _wc_base_values + _wc_delta
if not _wc_np.isfinite(_wc_values).all():
    raise RuntimeError('non-finite calibrated-well HMM values')
_wc_final = _wc_merged[['id']].copy()
_wc_final['tvt'] = _wc_values
if not _wc_final['id'].equals(_wc_sample['id']):
    raise RuntimeError('calibrated-well HMM id order mismatch')
_wc_final.to_csv(_wc_work / 'submission.csv', index=False)

def _wc_sha(path):
    return _wc_hashlib.sha256(_WcPath(path).read_bytes()).hexdigest()

_wc_well_stats = []
for _rank, _wid in enumerate(_wc_unique):
    _mask = _wc_wells.eq(_wid).to_numpy()
    _part = _wc_direction[_mask]
    _weight = float(_wc_weight_map[_wid])
    _wc_well_stats.append({{
        'well_rank': int(_rank),
        'well_id': _wid,
        'weight': _weight,
        'rows': int(_mask.sum()),
        'direction_sse_per_total_row': float(_wc_np.sum(_part ** 2) / len(_wc_final)),
        'rms_move': float(_wc_np.sqrt(_wc_np.mean((_weight * _part) ** 2))),
        'posterior_std_mean': float(_wc_merged.loc[_mask, 'hmm_std'].mean()),
    }})
_WC_AUDIT = {{
    'strategy': {config["strategy"]!r},
    'weights_by_sorted_well': list(_WC_WEIGHTS),
    'tomography_refs': {list(config["tomography_refs"])!r},
    'tomography_scores': {list(config["tomography_scores"])!r},
    'predicted_public_score': {config["predicted_public_score"]!r},
    'rows': int(len(_wc_final)),
    'id_order_matches_sample': True,
    'well_stats': _wc_well_stats,
    'base_sha256': _wc_sha(_wc_work / 'submission_before_exact_hmm.csv'),
    'hmm_sha256': _wc_sha(_wc_work / 'exact_hmm_predictions.csv'),
    'final_sha256': _wc_sha(_wc_work / 'submission.csv'),
}}
(_wc_work / 'hmm_well_calibration_audit.json').write_text(
    _wc_json.dumps(_WC_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('HMM well calibration audit:', _WC_AUDIT, flush=True)
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
    config = TARGETS[target]
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"] = [copy.deepcopy(cell) for cell in notebook["cells"]]
    if config.get("student_t"):
        replacements = 0
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            old = "result = run_hmm2(hw[TEST_COLS].copy(), tw)"
            if old in source:
                hmm_params = config.get(
                    "hmm_params", "HMMParams(emission='t', sigma_mode='std')"
                )
                source = source.replace(
                    old,
                    f"result = run_hmm2(hw[TEST_COLS].copy(), tw, params={hmm_params})",
                )
                replacements += 1
            source = source.replace("_HMM_WEIGHT = 0.15", "_HMM_WEIGHT = 0.25")
            source = source.replace(
                "pkadopt_7053_plus_exact_second_order_hmm",
                "pkadopt_7053_plus_student_t_second_order_hmm",
            )
            source = source.replace(
                "pkadopt_7053_exact_hmm_weight_015",
                "pkadopt_7053_student_t_hmm_weight_025",
            )
            cell["source"] = source
        if replacements != 1:
            raise RuntimeError(f"expected one HMM call replacement, found {replacements}")
    if "well_weights" in config:
        followup = calibrated_well_cell(config)
    elif "coded_signs" in config:
        followup = coded_well_cell(config)
    elif "uncertainty_gate" in config:
        followup = uncertainty_gate_cell(config)
    elif "shape_weight" in config:
        followup = decomposition_cell(
            config["mean_weight"],
            config["shape_weight"],
            config["strategy"],
            config["native_wells"],
            config["native_reference_row_rmse"],
            config["native_candidate_row_rmse"],
            config["public_reference_score"],
        )
    else:
        followup = followup_cell(config["weight"], config["strategy"])
    notebook["cells"].append(code_cell(followup))
    validate_notebook(notebook, config["slug"])
    output_dir = Path("kaggle") / config["slug"]
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f'{config["slug"]}.ipynb'
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")

    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata["id"] = f'qwer556617123/{config["slug"]}'
    metadata["title"] = config["slug"]
    metadata["code_file"] = notebook_path.name
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()
    unknown = sorted(set(args.targets) - set(TARGETS))
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}")
    for target in args.targets or list(TARGETS):
        print(f"built {build(target)}")


if __name__ == "__main__":
    main()
