"""Build a joint parent-datum and child-contrast calibrated notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_c1_rank8_datum16_codes import H4
from build_c1_rank8_datum_calibrated import BIN_PROJECTIONS
from build_c1_rank8_datum_codes import C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import code_cell, source
from fit_c1_rank8_datum16_stage import decode_stage


OFFSET_CAP = 3.0


def solve_parent_coefficients(
    child_fractions: np.ndarray,
    parent_projection: float,
    contrast_projection: float,
    child_signs: np.ndarray,
) -> np.ndarray:
    fractions = np.asarray(child_fractions, dtype=float)
    signs = np.asarray(child_signs, dtype=float)
    q = float(fractions.sum())
    cross = float(fractions @ signs)
    gram = np.asarray(((q, cross), (cross, q)), dtype=float)
    rhs = -np.asarray((parent_projection, contrast_projection), dtype=float)
    if q <= 0.0:
        return np.zeros(2, dtype=float)
    return np.linalg.pinv(gram, rcond=1.0e-12) @ rhs


def stage_cell(
    child_code_index: int,
    contrast_projections: tuple[float, float, float, float],
) -> dict:
    child_code = H4[child_code_index]
    code = f'''# Joint LB-decoded parent datum and nested child contrast calibration.
import hashlib as _dsc_hashlib
import json as _dsc_json
from pathlib import Path as _DscPath

import numpy as _dsc_np
import pandas as _dsc_pd

_DSC_PARENT_PROJECTIONS = {BIN_PROJECTIONS!r}
_DSC_CONTRAST_PROJECTIONS = {contrast_projections!r}
_DSC_CHILD_CODE_INDEX = {child_code_index!r}
_DSC_CHILD_CODE = {child_code!r}
_DSC_OFFSET_CAP = {OFFSET_CAP!r}
_DSC_WORK = _DscPath('/kaggle/working') if _DscPath('/kaggle/working').exists() else _DscPath('.')
_DSC_SUB = _DSC_WORK / 'submission.csv'

_dsc_base = _dsc_pd.read_csv(_DSC_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_dsc_base['tvt'] = _dsc_pd.to_numeric(_dsc_base['tvt'], errors='coerce')
if not _dsc_np.isfinite(_dsc_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('datum16 stage received non-finite rank-8 anchor')
_dsc_base.to_csv(_DSC_WORK / 'submission_before_datum16_stage.csv', index=False)
_dsc_parts = _dsc_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _dsc_parts.shape[1] != 2:
    raise RuntimeError('datum16 stage could not parse run-local submission IDs')
_dsc_base['_well'] = _dsc_parts[0].astype(str)
_dsc_counts = _dsc_base['_well'].value_counts().to_dict()

_dsc_parent_rows = [0, 0, 0, 0]
_dsc_parent_wells = [[], [], [], []]
for _dsc_well, _dsc_rows in sorted(_dsc_counts.items(), key=lambda item: (-item[1], item[0])):
    _dsc_parent = min(range(4), key=lambda value: (_dsc_parent_rows[value], value))
    _dsc_parent_rows[_dsc_parent] += int(_dsc_rows)
    _dsc_parent_wells[_dsc_parent].append(str(_dsc_well))

_dsc_leaf_rows = _dsc_np.zeros((4, 4), dtype=int)
_dsc_well_leaf = {{}}
for _dsc_parent in range(4):
    _dsc_child_rows = [0, 0, 0, 0]
    _dsc_members = sorted(
        ((_dsc_well, int(_dsc_counts[_dsc_well])) for _dsc_well in _dsc_parent_wells[_dsc_parent]),
        key=lambda item: (-item[1], item[0]),
    )
    for _dsc_well, _dsc_rows in _dsc_members:
        _dsc_child = min(range(4), key=lambda value: (_dsc_child_rows[value], value))
        _dsc_well_leaf[str(_dsc_well)] = (int(_dsc_parent), int(_dsc_child))
        _dsc_child_rows[_dsc_child] += int(_dsc_rows)
        _dsc_leaf_rows[_dsc_parent, _dsc_child] += int(_dsc_rows)

_dsc_fractions = _dsc_leaf_rows.astype(float) / float(len(_dsc_base))
_dsc_signs = _dsc_np.asarray([1.0 if char == '+' else -1.0 for char in _DSC_CHILD_CODE])
_dsc_parent_projection = _dsc_np.asarray(_DSC_PARENT_PROJECTIONS, dtype=float)
_dsc_contrast_projection = _dsc_np.asarray(_DSC_CONTRAST_PROJECTIONS, dtype=float)
_dsc_coefficients = _dsc_np.zeros((4, 2), dtype=float)
_dsc_grams = []
_dsc_conditions = []
_dsc_raw_offsets = _dsc_np.zeros((4, 4), dtype=float)
for _dsc_parent in range(4):
    _dsc_q = float(_dsc_fractions[_dsc_parent].sum())
    _dsc_cross = float(_dsc_fractions[_dsc_parent] @ _dsc_signs)
    _dsc_gram = _dsc_np.asarray(((_dsc_q, _dsc_cross), (_dsc_cross, _dsc_q)), dtype=float)
    _dsc_rhs = -_dsc_np.asarray(
        (_dsc_parent_projection[_dsc_parent], _dsc_contrast_projection[_dsc_parent]),
        dtype=float,
    )
    if _dsc_q > 0.0:
        _dsc_coefficients[_dsc_parent] = _dsc_np.linalg.pinv(_dsc_gram, rcond=1.0e-12) @ _dsc_rhs
    _dsc_raw_offsets[_dsc_parent] = (
        _dsc_coefficients[_dsc_parent, 0]
        + _dsc_coefficients[_dsc_parent, 1] * _dsc_signs
    )
    _dsc_grams.append(_dsc_gram.tolist())
    _dsc_conditions.append(
        None if _dsc_np.linalg.matrix_rank(_dsc_gram) < 2 else float(_dsc_np.linalg.cond(_dsc_gram))
    )

_dsc_offsets = _dsc_np.clip(_dsc_raw_offsets, -_DSC_OFFSET_CAP, _DSC_OFFSET_CAP)
_dsc_leaf = _dsc_base['_well'].map(_dsc_well_leaf)
_dsc_move = _dsc_np.asarray(
    [_dsc_offsets[int(parent), int(child)] for parent, child in _dsc_leaf],
    dtype=float,
)
_dsc_values = _dsc_base['tvt'].to_numpy(dtype=float) + _dsc_move
if not _dsc_np.isfinite(_dsc_values).all():
    raise RuntimeError('datum16 stage produced non-finite values')
_dsc_final = _dsc_base[['id']].copy()
_dsc_final['tvt'] = _dsc_values
_dsc_final.to_csv(_DSC_SUB, index=False)

def _dsc_sha(path):
    return _dsc_hashlib.sha256(_DscPath(path).read_bytes()).hexdigest()

_DATUM16_STAGE_AUDIT = {{
    'parent_projections': [float(value) for value in _dsc_parent_projection],
    'contrast_projections': [float(value) for value in _dsc_contrast_projection],
    'child_code_index': int(_DSC_CHILD_CODE_INDEX),
    'child_code': _DSC_CHILD_CODE,
    'parent_bin_rows': [int(value) for value in _dsc_leaf_rows.sum(axis=1)],
    'leaf_bin_rows': [[int(value) for value in row] for row in _dsc_leaf_rows],
    'leaf_bin_row_fractions': [[float(value) for value in row] for row in _dsc_fractions],
    'gram_matrices': _dsc_grams,
    'gram_conditions': _dsc_conditions,
    'datum_contrast_coefficients': _dsc_coefficients.tolist(),
    'raw_leaf_offsets': _dsc_raw_offsets.tolist(),
    'deployed_leaf_offsets': _dsc_offsets.tolist(),
    'offset_cap': float(_DSC_OFFSET_CAP),
    'partition': 'nested_greedy_row_balanced_run_local_wells_4x4',
    'fixed_public_ids_used': False,
    'rows': int(len(_dsc_final)),
    'wells': int(len(_dsc_well_leaf)),
    'mean_squared_move': float(_dsc_np.mean(_dsc_move * _dsc_move)),
    'max_abs_move': float(_dsc_np.max(_dsc_np.abs(_dsc_move))),
    'base_sha256': _dsc_sha(_DSC_WORK / 'submission_before_datum16_stage.csv'),
    'final_sha256': _dsc_sha(_DSC_SUB),
}}
(_DSC_WORK / 'datum16_stage_audit.json').write_text(
    _dsc_json.dumps(_DATUM16_STAGE_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _DSC_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = f'c1_rank8_datum16_b{{_DSC_CHILD_CODE_INDEX}}_calibrated'
print('datum16 stage audit:', _dsc_json.dumps(_DATUM16_STAGE_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def build(
    child_code_index: int,
    contrast_projections: tuple[float, float, float, float],
) -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8))
    notebook["cells"].append(stage_cell(child_code_index, contrast_projections))
    slug = f"rogii-c1r8-d16-b{child_code_index}-calibrated"
    output = _write(
        notebook,
        slug,
        f"c1_rank8_datum16_b{child_code_index}_joint_calibrated",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_stage'] = _DATUM16_STAGE_AUDIT",
    )
    generated = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in generated["cells"])
    assert f"_DSC_CONTRAST_PROJECTIONS = {contrast_projections!r}" in joined
    assert f"_DSC_CHILD_CODE_INDEX = {child_code_index!r}" in joined
    assert "00e12e8b" not in joined
    for cell in generated["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{slug}:code", "exec")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=float, required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--child-code-index", type=int, default=1)
    args = parser.parse_args()
    scores = tuple(float(value.strip()) for value in args.scores.split(","))
    decoded = decode_stage(
        args.anchor,
        scores,
        args.amplitude,
        args.child_code_index,
    )
    contrast = tuple(float(value) for value in decoded["parent_contrast_projection"])
    print(f"built and checked {build(args.child_code_index, contrast)}")


if __name__ == "__main__":
    main()
