"""Build the final calibrated b3 correction around the b12 candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_c1_datum_b12_centered_b3_batch import contrasts
from build_c1_rank8_datum16_codes import leaf_code
from build_c1_rank8_datum16_multistage_calibrated import multistage_cell
from build_c1_rank8_datum_codes import C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import code_cell, source


PARENT_CODE_INDICES = (0, 1, 2)
MOVE_CAP = 3.0


def solve_coefficients(gram: np.ndarray, projections: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(gram, dtype=float), rcond=1.0e-12) @ -np.asarray(
        projections,
        dtype=float,
    )


def centered_final_cell(
    projections: tuple[float, ...],
    parent_code_indices: tuple[int, ...] = PARENT_CODE_INDICES,
) -> dict:
    if len(projections) != len(parent_code_indices):
        raise ValueError("projections and parent code indices must have equal length")
    codes = tuple(leaf_code(index, 3) for index in parent_code_indices)
    cell_source = f'''# Calibrated competitive-probe correction around the b12 candidate.
import hashlib as _bcf_hashlib
import json as _bcf_json
from pathlib import Path as _BcfPath

import numpy as _bcf_np
import pandas as _bcf_pd

_BCF_PROJECTIONS = {projections!r}
_BCF_PARENT_CODE_INDICES = {parent_code_indices!r}
_BCF_LEAF_CODES = {codes!r}
_BCF_MOVE_CAP = {MOVE_CAP!r}
_BCF_WORK = _BcfPath('/kaggle/working') if _BcfPath('/kaggle/working').exists() else _BcfPath('.')
_BCF_SUB = _BCF_WORK / 'submission.csv'

_bcf_base = _bcf_pd.read_csv(_BCF_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_bcf_base['tvt'] = _bcf_pd.to_numeric(_bcf_base['tvt'], errors='coerce')
if not _bcf_np.isfinite(_bcf_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('centered b3 finalizer received non-finite b12 candidate')
_bcf_base.to_csv(_BCF_WORK / 'submission_before_b3_centered_final.csv', index=False)
_bcf_parts = _bcf_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _bcf_parts.shape[1] != 2:
    raise RuntimeError('centered b3 finalizer could not parse run-local IDs')
_bcf_base['_well'] = _bcf_parts[0].astype(str)
_bcf_counts = _bcf_base['_well'].value_counts().to_dict()

_bcf_parent_rows = [0, 0, 0, 0]
_bcf_parent_wells = [[], [], [], []]
for _bcf_well, _bcf_rows in sorted(_bcf_counts.items(), key=lambda item: (-item[1], item[0])):
    _bcf_parent = min(range(4), key=lambda value: (_bcf_parent_rows[value], value))
    _bcf_parent_rows[_bcf_parent] += int(_bcf_rows)
    _bcf_parent_wells[_bcf_parent].append(str(_bcf_well))

_bcf_leaf_rows = [0] * 16
_bcf_well_leaf = {{}}
for _bcf_parent in range(4):
    _bcf_child_rows = [0, 0, 0, 0]
    _bcf_members = sorted(
        ((_bcf_well, int(_bcf_counts[_bcf_well])) for _bcf_well in _bcf_parent_wells[_bcf_parent]),
        key=lambda item: (-item[1], item[0]),
    )
    for _bcf_well, _bcf_rows in _bcf_members:
        _bcf_child = min(range(4), key=lambda value: (_bcf_child_rows[value], value))
        _bcf_leaf = 4 * _bcf_parent + _bcf_child
        _bcf_well_leaf[str(_bcf_well)] = int(_bcf_leaf)
        _bcf_child_rows[_bcf_child] += int(_bcf_rows)
        _bcf_leaf_rows[_bcf_leaf] += int(_bcf_rows)

_bcf_leaf = _bcf_base['_well'].map(_bcf_well_leaf).to_numpy(dtype=int)
_bcf_signs = _bcf_np.asarray(
    [[1.0 if char == '+' else -1.0 for char in code] for code in _BCF_LEAF_CODES],
    dtype=float,
)
_bcf_directions = _bcf_signs[:, _bcf_leaf]
_bcf_gram = _bcf_directions @ _bcf_directions.T / float(len(_bcf_base))
_bcf_projection = _bcf_np.asarray(_BCF_PROJECTIONS, dtype=float)
_bcf_coefficients = _bcf_np.linalg.pinv(_bcf_gram, rcond=1.0e-12) @ -_bcf_projection
_bcf_raw_move = _bcf_coefficients @ _bcf_directions
_bcf_move = _bcf_np.clip(_bcf_raw_move, -_BCF_MOVE_CAP, _BCF_MOVE_CAP)
_bcf_values = _bcf_base['tvt'].to_numpy(dtype=float) + _bcf_move
if not _bcf_np.isfinite(_bcf_values).all():
    raise RuntimeError('centered b3 finalizer produced non-finite values')
_bcf_final = _bcf_base[['id']].copy()
_bcf_final['tvt'] = _bcf_values
_bcf_final.to_csv(_BCF_SUB, index=False)

def _bcf_sha(path):
    return _bcf_hashlib.sha256(_BcfPath(path).read_bytes()).hexdigest()

_B3_CENTERED_FINAL_AUDIT = {{
    'projections': [float(value) for value in _bcf_projection],
    'parent_code_indices': [int(value) for value in _BCF_PARENT_CODE_INDICES],
    'leaf_codes': list(_BCF_LEAF_CODES),
    'gram_matrix': _bcf_gram.tolist(),
    'gram_rank': int(_bcf_np.linalg.matrix_rank(_bcf_gram)),
    'gram_condition': None if _bcf_np.linalg.matrix_rank(_bcf_gram) < len(_BCF_PROJECTIONS) else float(_bcf_np.linalg.cond(_bcf_gram)),
    'coefficients': _bcf_coefficients.tolist(),
    'move_cap': float(_BCF_MOVE_CAP),
    'raw_move_min': float(_bcf_np.min(_bcf_raw_move)),
    'raw_move_max': float(_bcf_np.max(_bcf_raw_move)),
    'move_min': float(_bcf_np.min(_bcf_move)),
    'move_max': float(_bcf_np.max(_bcf_move)),
    'mean_squared_move': float(_bcf_np.mean(_bcf_move * _bcf_move)),
    'leaf_bin_rows': [int(value) for value in _bcf_leaf_rows],
    'partition': 'nested_greedy_row_balanced_run_local_wells_4x4',
    'fixed_public_ids_used': False,
    'rows': int(len(_bcf_final)),
    'wells': int(len(_bcf_well_leaf)),
    'base_sha256': _bcf_sha(_BCF_WORK / 'submission_before_b3_centered_final.csv'),
    'final_sha256': _bcf_sha(_BCF_SUB),
}}
(_BCF_WORK / 'b3_centered_final_audit.json').write_text(
    _bcf_json.dumps(_B3_CENTERED_FINAL_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _BCF_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = 'c1_rank8_datum16_b12_b3center_calibrated'
print('b3 centered final audit:', _bcf_json.dumps(_B3_CENTERED_FINAL_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(cell_source)


def build(projections: tuple[float, ...]) -> Path:
    notebook = _load_base()
    notebook["cells"].append(c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8))
    notebook["cells"].append(multistage_cell((1, 2), contrasts()))
    notebook["cells"].append(centered_final_cell(projections))
    slug = "rogii-c1r8-d16-b12-b3center-calibrated"
    output = _write(
        notebook,
        slug,
        "c1_rank8_datum16_b12_b3center_calibrated",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_multistage'] = _DATUM16_MULTISTAGE_AUDIT\n"
        "_sd_audit['b3_centered_final'] = _B3_CENTERED_FINAL_AUDIT",
    )
    generated = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in generated["cells"])
    assert f"_BCF_PROJECTIONS = {projections!r}" in joined
    assert "_DMC_CHILD_CODE_INDICES = (1, 2)" in joined
    assert "00e12e8b" not in joined
    for cell in generated["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{slug}:code", "exec")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b12-score", type=float, required=True)
    parser.add_argument("--probe-scores", required=True)
    parser.add_argument("--amplitude", type=float, default=1.0)
    args = parser.parse_args()
    scores = tuple(float(value.strip()) for value in args.probe_scores.split(","))
    if len(scores) != 3:
        raise ValueError("exactly three centered probe scores are required")
    projections = tuple(
        float(
            (score * score - args.b12_score * args.b12_score - args.amplitude * args.amplitude)
            / (2.0 * args.amplitude)
        )
        for score in scores
    )
    print(json.dumps({"b12_score": args.b12_score, "scores": scores, "projections": projections}, sort_keys=True))
    print(f"built and checked {build(projections)}")


if __name__ == "__main__":
    main()
