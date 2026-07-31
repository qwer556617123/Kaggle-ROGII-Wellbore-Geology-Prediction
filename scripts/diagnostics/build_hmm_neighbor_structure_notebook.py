"""Build the scored HMM anchor plus a native-gated neighbor-structure overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


OVERLAY_CELL = r'''# Cross-well structural-path transfer, native-mask selected alpha=0.10.
import hashlib as _ns_hashlib
import json as _ns_json
import time as _ns_time
from pathlib import Path as _NsPath
import numpy as _ns_np
import pandas as _ns_pd
from scipy.ndimage import gaussian_filter1d as _ns_gaussian
from scipy.spatial import cKDTree as _NsTree

_ns_started = _ns_time.time()
_NS_STRIDE = 4
_NS_ALPHA = 0.10
_NS_NEIGHBORS = 16
_NS_TOP_K = 3
_NS_PREFIX_ROWS = 512
_ns_work = _NsPath('/kaggle/working') if _NsPath('/kaggle/working').exists() else _NsPath('.')
_ns_data = _NsPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_ns_data / 'sample_submission.csv').exists():
    _ns_data = _NsPath('/kaggle/input/rogii-wellbore-geology-prediction')
_ns_submission_path = _ns_work / 'submission.csv'
_ns_anchor_path = _ns_work / 'submission_before_neighbor_structure.csv'
_ns_anchor = _ns_pd.read_csv(_ns_submission_path)[['id', 'tvt']]
_ns_anchor['id'] = _ns_anchor['id'].astype(str)
_ns_anchor.to_csv(_ns_anchor_path, index=False)

def _ns_unit_direction(xy):
    delta = _ns_np.asarray(xy[-1] - xy[0], dtype=float)
    return delta / max(float(_ns_np.linalg.norm(delta)), 1e-9)

def _ns_robust_affine(x, y):
    x = _ns_np.asarray(x, dtype=float)
    y = _ns_np.asarray(y, dtype=float)
    x0 = float(x[-1])
    scale = max(float(_ns_np.ptp(x)), 1.0)
    xn = (x - x0) / scale
    design = _ns_np.column_stack([_ns_np.ones(len(xn)), xn])
    weights = _ns_np.ones(len(xn), dtype=float)
    coef = _ns_np.array([float(_ns_np.median(y)), 0.0])
    sigma = 1.0
    for _ in range(8):
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        coef = _ns_np.linalg.solve(lhs + _ns_np.eye(2) * 1e-8, rhs)
        residual = y - design @ coef
        sigma = max(1.4826 * float(_ns_np.median(_ns_np.abs(residual))), 0.25)
        weights = _ns_np.minimum(1.0, 1.5 * sigma / _ns_np.maximum(_ns_np.abs(residual), 1e-9))
    residual = _ns_np.clip(y - design @ coef, -3.0 * sigma, 3.0 * sigma)
    error = float(_ns_np.sqrt(_ns_np.mean(residual ** 2)))
    return float(coef[0]), float(coef[1] / scale), error

_ns_profiles = {}
_ns_global_xy = []
_ns_global_well = []
_ns_train_paths = sorted((_ns_data / 'train').glob('*__horizontal_well.csv'))
for _ns_i, _ns_path in enumerate(_ns_train_paths, 1):
    _ns_well = _ns_path.name.split('__', 1)[0]
    _ns_frame = _ns_pd.read_csv(_ns_path, usecols=['MD', 'X', 'Y', 'Z', 'TVT'])
    _ns_rows = _ns_np.flatnonzero(_ns_frame['TVT'].notna().to_numpy())[::_NS_STRIDE]
    if len(_ns_rows) < 32:
        continue
    _ns_xy = _ns_frame.loc[_ns_rows, ['X', 'Y']].to_numpy(dtype=float)
    _ns_md = _ns_frame.loc[_ns_rows, 'MD'].to_numpy(dtype=float)
    _ns_u = (_ns_frame.loc[_ns_rows, 'TVT'].to_numpy(dtype=float)
             + _ns_frame.loc[_ns_rows, 'Z'].to_numpy(dtype=float))
    _ns_u = _ns_gaussian(_ns_u, sigma=max(1.0, 12.0 / _NS_STRIDE), mode='nearest')
    _ns_profiles[_ns_well] = {
        'xy': _ns_xy,
        'md': _ns_md,
        'u': _ns_u,
        'direction': _ns_unit_direction(_ns_xy),
    }
    _ns_global_xy.append(_ns_xy)
    _ns_global_well.extend([_ns_well] * len(_ns_xy))
    if _ns_i % 100 == 0:
        print(f'[neighbor-structure] loaded {_ns_i}/{len(_ns_train_paths)} profiles', flush=True)

_ns_global_xy = _ns_np.concatenate(_ns_global_xy, axis=0)
_ns_global_well = _ns_np.asarray(_ns_global_well, dtype=object)
_ns_global_tree = _NsTree(_ns_global_xy)
_ns_out = _ns_anchor.copy()
_ns_out['_well'] = _ns_out['id'].str.rsplit('_', n=1).str[0]
_ns_out['_row'] = _ns_out['id'].str.rsplit('_', n=1).str[1].astype(int)
_ns_well_audits = []

for _ns_well, _ns_group in _ns_out.groupby('_well', sort=False):
    _ns_hw = _ns_pd.read_csv(_ns_data / 'test' / f'{_ns_well}__horizontal_well.csv')
    _ns_known = _ns_hw['TVT_input'].notna().to_numpy()
    _ns_known_rows = _ns_np.flatnonzero(_ns_known)
    if len(_ns_known_rows) < 80:
        _ns_well_audits.append({'well': _ns_well, 'applied': False, 'reason': 'short_prefix'})
        continue
    _ns_target_xy = _ns_hw[['X', 'Y']].to_numpy(dtype=float)
    _ns_target_md = _ns_hw['MD'].to_numpy(dtype=float)
    _ns_target_u = (_ns_hw['TVT_input'].to_numpy(dtype=float)
                    + _ns_hw['Z'].to_numpy(dtype=float))
    _ns_target_direction = _ns_unit_direction(_ns_target_xy)
    _ns_heel_xy = _ns_target_xy[_ns_known_rows[-1]]
    _, _ns_point_indices = _ns_global_tree.query(
        _ns_heel_xy, k=min(len(_ns_global_well), max(512, _NS_NEIGHBORS * 64))
    )
    _ns_neighbor_ids = []
    for _ns_point_index in _ns_np.atleast_1d(_ns_point_indices):
        _ns_neighbor = str(_ns_global_well[int(_ns_point_index)])
        if _ns_neighbor == _ns_well or _ns_neighbor in _ns_neighbor_ids:
            continue
        _ns_neighbor_ids.append(_ns_neighbor)
        if len(_ns_neighbor_ids) >= _NS_NEIGHBORS:
            break

    _ns_aligned = []
    for _ns_neighbor in _ns_neighbor_ids:
        _ns_profile = _ns_profiles[_ns_neighbor]
        _ns_direction_similarity = abs(float(_ns_np.dot(_ns_target_direction, _ns_profile['direction'])))
        if _ns_direction_similarity < 0.65:
            continue
        _ns_distances, _ns_nearest = _NsTree(_ns_profile['xy']).query(_ns_target_xy, k=1)
        _ns_mapped_u = _ns_gaussian(_ns_profile['u'][_ns_nearest], sigma=10.0, mode='nearest')
        _ns_fit_rows = _ns_known_rows[-_NS_PREFIX_ROWS:]
        _ns_residual = _ns_target_u[_ns_fit_rows] - _ns_mapped_u[_ns_fit_rows]
        _ns_intercept, _ns_slope, _ns_prefix_error = _ns_robust_affine(
            _ns_target_md[_ns_fit_rows], _ns_residual
        )
        _ns_correction = _ns_intercept + _ns_slope * (
            _ns_target_md - _ns_target_md[_ns_fit_rows[-1]]
        )
        _ns_correction = _ns_np.clip(_ns_correction, _ns_intercept - 25.0, _ns_intercept + 25.0)
        _ns_pred_u = _ns_mapped_u + _ns_correction
        _ns_distance_median = float(_ns_np.median(_ns_distances[_ns_fit_rows]))
        _ns_score = (_ns_prefix_error + 0.0025 * _ns_distance_median
                     + 12.0 * (1.0 - _ns_direction_similarity)
                     + 0.05 * abs(_ns_correction[-1] - _ns_correction[_ns_fit_rows[-1]]))
        _ns_aligned.append((_ns_score, _ns_neighbor, _ns_pred_u, {
            'prefix_rmse': _ns_prefix_error,
            'distance_prefix_median': _ns_distance_median,
            'direction_similarity': _ns_direction_similarity,
            'datum_intercept': _ns_intercept,
            'datum_slope_per_ft': _ns_slope,
        }))

    _ns_aligned.sort(key=lambda item: item[0])
    _ns_selected = _ns_aligned[:_NS_TOP_K]
    if not _ns_selected:
        _ns_well_audits.append({'well': _ns_well, 'applied': False, 'reason': 'no_compatible_neighbor'})
        continue
    _ns_scores = _ns_np.asarray([item[0] for item in _ns_selected], dtype=float)
    _ns_weights = _ns_np.exp(-(_ns_scores - _ns_scores.min()) / max(float(_ns_np.std(_ns_scores)), 1.0))
    _ns_weights /= _ns_weights.sum()
    _ns_analog_u = _ns_np.sum(
        _ns_np.stack([item[2] for item in _ns_selected]) * _ns_weights[:, None], axis=0
    )
    _ns_rows = _ns_group['_row'].to_numpy(dtype=int)
    _ns_locations = _ns_group.index.to_numpy(dtype=int)
    _ns_analog_tvt = _ns_analog_u[_ns_rows] - _ns_hw['Z'].to_numpy(dtype=float)[_ns_rows]
    _ns_base_tvt = _ns_out.loc[_ns_locations, 'tvt'].to_numpy(dtype=float)
    _ns_final_tvt = _ns_base_tvt + _NS_ALPHA * (_ns_analog_tvt - _ns_base_tvt)
    _ns_out.loc[_ns_locations, 'tvt'] = _ns_final_tvt
    _ns_well_audits.append({
        'well': _ns_well,
        'applied': True,
        'neighbors': [item[1] for item in _ns_selected],
        'weights': [float(value) for value in _ns_weights],
        'scores': [float(value) for value in _ns_scores],
        'best_metadata': _ns_selected[0][3],
        'mean_move_ft': float(_ns_np.mean(_ns_final_tvt - _ns_base_tvt)),
        'rms_move_ft': float(_ns_np.sqrt(_ns_np.mean((_ns_final_tvt - _ns_base_tvt) ** 2))),
        'max_abs_move_ft': float(_ns_np.max(_ns_np.abs(_ns_final_tvt - _ns_base_tvt))),
    })

_ns_sample = _ns_pd.read_csv(_ns_data / 'sample_submission.csv')[['id']]
_ns_sample['id'] = _ns_sample['id'].astype(str)
if not _ns_out['id'].equals(_ns_sample['id']):
    raise RuntimeError('neighbor-structure submission/sample id mismatch')
if not _ns_np.isfinite(_ns_out['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('neighbor-structure output contains non-finite TVT')
_ns_out[['id', 'tvt']].to_csv(_ns_submission_path, index=False)

def _ns_sha(path):
    return _ns_hashlib.sha256(_NsPath(path).read_bytes()).hexdigest()

_ns_final_values = _ns_out['tvt'].to_numpy(dtype=float)
_ns_anchor_values = _ns_anchor['tvt'].to_numpy(dtype=float)
_ns_move = _ns_final_values - _ns_anchor_values
_ns_audit = {
    'strategy': 'pkadopt_wellcal_student_hmm_plus_neighbor_structure_alpha010',
    'native_mask_gate': {
        'wells': 100,
        'anchor_row_rmse': 11.470629,
        'candidate_row_rmse': 11.144423,
        'row_gain': 0.326206,
        'anchor_well_median': 5.935687,
        'candidate_well_median': 5.749694,
        'same_id_excluded': True,
    },
    'alpha': _NS_ALPHA,
    'rows': int(len(_ns_out)),
    'id_order_matches_sample': True,
    'runtime_sec': float(_ns_time.time() - _SD_STARTED),
    'overlay_runtime_sec': float(_ns_time.time() - _ns_started),
    'mean_move_ft': float(_ns_np.mean(_ns_move)),
    'rms_move_ft': float(_ns_np.sqrt(_ns_np.mean(_ns_move ** 2))),
    'max_abs_move_ft': float(_ns_np.max(_ns_np.abs(_ns_move))),
    'anchor_sha256': _ns_sha(_ns_anchor_path),
    'final_sha256': _ns_sha(_ns_submission_path),
    'wells': _ns_well_audits,
}
(_ns_work / 'neighbor_structure_audit.json').write_text(
    _ns_json.dumps(_ns_audit, indent=2, sort_keys=True), encoding='utf-8'
)
print('Neighbor-structure final audit:', _ns_json.dumps(_ns_audit, sort_keys=True), flush=True)
'''


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-name", default="rogii-pkadopt-hmm-student-wellcal-650505"
    )
    parser.add_argument(
        "--target-name", default="rogii-pkadopt-hmm-neighbor-structure010"
    )
    args = parser.parse_args()

    source_dir = ROOT / "kaggle" / args.source_name
    source_notebook = source_dir / f"{args.source_name}.ipynb"
    notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
    profile_cell = notebook["cells"][1]
    profile_source = _cell_source(profile_cell)
    marker = "_profile = PROFILE_PRESETS[SUBMISSION_PROFILE]\n"
    if marker not in profile_source:
        raise RuntimeError("could not locate expanded profile assignment")
    profile_source = profile_source.replace(
        marker,
        marker
        + "# Runtime-safe replay: both disabled layers were row-identical or gated off.\n"
        + "_profile['run_visible_prefix_calibration'] = False\n"
        + "_profile['run_model_package_correction'] = False\n",
        1,
    )
    profile_cell["source"] = profile_source
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": OVERLAY_CELL,
        }
    )
    for cell in notebook.get("cells", []):
        cell["execution_count"] = None
        cell["outputs"] = []

    target_dir = ROOT / "kaggle" / args.target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_notebook = target_dir / f"{args.target_name}.ipynb"
    target_notebook.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    metadata = json.loads(
        (source_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    metadata.update(
        {
            "id": f"qwer556617123/{args.target_name}",
            "title": args.target_name,
            "code_file": target_notebook.name,
        }
    )
    (target_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(target_notebook)


if __name__ == "__main__":
    main()
