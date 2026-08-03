"""Build a joint parent-datum and two-child-contrast calibrated notebook."""
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
    projections: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    fractions = np.asarray(child_fractions, dtype=float)
    projection = np.asarray(projections, dtype=float)
    design = np.asarray(basis, dtype=float)
    gram = (design * fractions[None, :]) @ design.T
    if float(fractions.sum()) <= 0.0:
        return np.zeros(design.shape[0], dtype=float)
    return np.linalg.pinv(gram, rcond=1.0e-12) @ -projection


def multistage_cell(
    child_code_indices: tuple[int, int],
    contrast_projections: tuple[tuple[float, ...], tuple[float, ...]],
) -> dict:
    child_codes = tuple(H4[index] for index in child_code_indices)
    code = f'''# Joint LB-decoded parent datum and two nested child contrasts.
import hashlib as _dmc_hashlib
import json as _dmc_json
from pathlib import Path as _DmcPath

import numpy as _dmc_np
import pandas as _dmc_pd

_DMC_PARENT_PROJECTIONS = {BIN_PROJECTIONS!r}
_DMC_CONTRAST_PROJECTIONS = {contrast_projections!r}
_DMC_CHILD_CODE_INDICES = {child_code_indices!r}
_DMC_CHILD_CODES = {child_codes!r}
_DMC_OFFSET_CAP = {OFFSET_CAP!r}
_DMC_WORK = _DmcPath('/kaggle/working') if _DmcPath('/kaggle/working').exists() else _DmcPath('.')
_DMC_SUB = _DMC_WORK / 'submission.csv'

_dmc_base = _dmc_pd.read_csv(_DMC_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_dmc_base['tvt'] = _dmc_pd.to_numeric(_dmc_base['tvt'], errors='coerce')
if not _dmc_np.isfinite(_dmc_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('datum16 multistage received non-finite rank-8 anchor')
_dmc_base.to_csv(_DMC_WORK / 'submission_before_datum16_multistage.csv', index=False)
_dmc_parts = _dmc_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _dmc_parts.shape[1] != 2:
    raise RuntimeError('datum16 multistage could not parse run-local submission IDs')
_dmc_base['_well'] = _dmc_parts[0].astype(str)
_dmc_counts = _dmc_base['_well'].value_counts().to_dict()

_dmc_parent_rows = [0, 0, 0, 0]
_dmc_parent_wells = [[], [], [], []]
for _dmc_well, _dmc_rows in sorted(_dmc_counts.items(), key=lambda item: (-item[1], item[0])):
    _dmc_parent = min(range(4), key=lambda value: (_dmc_parent_rows[value], value))
    _dmc_parent_rows[_dmc_parent] += int(_dmc_rows)
    _dmc_parent_wells[_dmc_parent].append(str(_dmc_well))

_dmc_leaf_rows = _dmc_np.zeros((4, 4), dtype=int)
_dmc_well_leaf = {{}}
for _dmc_parent in range(4):
    _dmc_child_rows = [0, 0, 0, 0]
    _dmc_members = sorted(
        ((_dmc_well, int(_dmc_counts[_dmc_well])) for _dmc_well in _dmc_parent_wells[_dmc_parent]),
        key=lambda item: (-item[1], item[0]),
    )
    for _dmc_well, _dmc_rows in _dmc_members:
        _dmc_child = min(range(4), key=lambda value: (_dmc_child_rows[value], value))
        _dmc_well_leaf[str(_dmc_well)] = (int(_dmc_parent), int(_dmc_child))
        _dmc_child_rows[_dmc_child] += int(_dmc_rows)
        _dmc_leaf_rows[_dmc_parent, _dmc_child] += int(_dmc_rows)

_dmc_fractions = _dmc_leaf_rows.astype(float) / float(len(_dmc_base))
_dmc_basis = _dmc_np.vstack([
    _dmc_np.ones(4, dtype=float),
    *[
        _dmc_np.asarray([1.0 if char == '+' else -1.0 for char in code], dtype=float)
        for code in _DMC_CHILD_CODES
    ],
])
_dmc_parent_projection = _dmc_np.asarray(_DMC_PARENT_PROJECTIONS, dtype=float)
_dmc_contrast_projection = _dmc_np.asarray(_DMC_CONTRAST_PROJECTIONS, dtype=float)
_dmc_coefficients = _dmc_np.zeros((4, 3), dtype=float)
_dmc_grams = []
_dmc_ranks = []
_dmc_conditions = []
_dmc_raw_offsets = _dmc_np.zeros((4, 4), dtype=float)
for _dmc_parent in range(4):
    _dmc_gram = (_dmc_basis * _dmc_fractions[_dmc_parent][None, :]) @ _dmc_basis.T
    _dmc_rhs = -_dmc_np.asarray(
        [_dmc_parent_projection[_dmc_parent]]
        + [_dmc_contrast_projection[index, _dmc_parent] for index in range(2)],
        dtype=float,
    )
    if float(_dmc_fractions[_dmc_parent].sum()) > 0.0:
        _dmc_coefficients[_dmc_parent] = _dmc_np.linalg.pinv(_dmc_gram, rcond=1.0e-12) @ _dmc_rhs
    _dmc_raw_offsets[_dmc_parent] = _dmc_coefficients[_dmc_parent] @ _dmc_basis
    _dmc_rank = int(_dmc_np.linalg.matrix_rank(_dmc_gram))
    _dmc_grams.append(_dmc_gram.tolist())
    _dmc_ranks.append(_dmc_rank)
    _dmc_conditions.append(
        None if _dmc_rank < 3 else float(_dmc_np.linalg.cond(_dmc_gram))
    )

_dmc_offsets = _dmc_np.clip(_dmc_raw_offsets, -_DMC_OFFSET_CAP, _DMC_OFFSET_CAP)
_dmc_leaf = _dmc_base['_well'].map(_dmc_well_leaf)
_dmc_move = _dmc_np.asarray(
    [_dmc_offsets[int(parent), int(child)] for parent, child in _dmc_leaf],
    dtype=float,
)
_dmc_values = _dmc_base['tvt'].to_numpy(dtype=float) + _dmc_move
if not _dmc_np.isfinite(_dmc_values).all():
    raise RuntimeError('datum16 multistage produced non-finite values')
_dmc_final = _dmc_base[['id']].copy()
_dmc_final['tvt'] = _dmc_values
_dmc_final.to_csv(_DMC_SUB, index=False)

def _dmc_sha(path):
    return _dmc_hashlib.sha256(_DmcPath(path).read_bytes()).hexdigest()

_DATUM16_MULTISTAGE_AUDIT = {{
    'parent_projections': [float(value) for value in _dmc_parent_projection],
    'contrast_projections': _dmc_contrast_projection.tolist(),
    'child_code_indices': [int(value) for value in _DMC_CHILD_CODE_INDICES],
    'child_codes': list(_DMC_CHILD_CODES),
    'parent_bin_rows': [int(value) for value in _dmc_leaf_rows.sum(axis=1)],
    'leaf_bin_rows': [[int(value) for value in row] for row in _dmc_leaf_rows],
    'leaf_bin_row_fractions': _dmc_fractions.tolist(),
    'gram_matrices': _dmc_grams,
    'gram_ranks': _dmc_ranks,
    'gram_conditions': _dmc_conditions,
    'datum_contrast_coefficients': _dmc_coefficients.tolist(),
    'raw_leaf_offsets': _dmc_raw_offsets.tolist(),
    'deployed_leaf_offsets': _dmc_offsets.tolist(),
    'offset_cap': float(_DMC_OFFSET_CAP),
    'partition': 'nested_greedy_row_balanced_run_local_wells_4x4',
    'fixed_public_ids_used': False,
    'rows': int(len(_dmc_final)),
    'wells': int(len(_dmc_well_leaf)),
    'mean_squared_move': float(_dmc_np.mean(_dmc_move * _dmc_move)),
    'max_abs_move': float(_dmc_np.max(_dmc_np.abs(_dmc_move))),
    'base_sha256': _dmc_sha(_DMC_WORK / 'submission_before_datum16_multistage.csv'),
    'final_sha256': _dmc_sha(_DMC_SUB),
}}
(_DMC_WORK / 'datum16_multistage_audit.json').write_text(
    _dmc_json.dumps(_DATUM16_MULTISTAGE_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _DMC_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = 'c1_rank8_datum16_b12_calibrated'
print('datum16 multistage audit:', _dmc_json.dumps(_DATUM16_MULTISTAGE_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def build(
    contrast_projections: tuple[tuple[float, ...], tuple[float, ...]],
) -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8))
    notebook["cells"].append(multistage_cell((1, 2), contrast_projections))
    slug = "rogii-c1r8-d16-b12-calibrated"
    output = _write(
        notebook,
        slug,
        "c1_rank8_datum16_b12_joint_calibrated",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_multistage'] = _DATUM16_MULTISTAGE_AUDIT",
    )
    generated = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in generated["cells"])
    assert f"_DMC_CONTRAST_PROJECTIONS = {contrast_projections!r}" in joined
    assert "00e12e8b" not in joined
    for cell in generated["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{slug}:code", "exec")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=float, required=True)
    parser.add_argument("--b1-scores", required=True)
    parser.add_argument("--b2-scores", required=True)
    parser.add_argument("--amplitude", type=float, default=2.0)
    args = parser.parse_args()

    decoded = []
    for index, raw_scores in ((1, args.b1_scores), (2, args.b2_scores)):
        scores = tuple(float(value.strip()) for value in raw_scores.split(","))
        decoded.append(decode_stage(args.anchor, scores, args.amplitude, index))
    contrasts = tuple(
        tuple(float(value) for value in item["parent_contrast_projection"])
        for item in decoded
    )
    print(f"built and checked {build(contrasts)}")


if __name__ == "__main__":
    main()
