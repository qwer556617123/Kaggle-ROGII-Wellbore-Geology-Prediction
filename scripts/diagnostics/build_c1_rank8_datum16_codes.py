"""Build the remaining 12 codes of a nested 16-bin datum Hadamard system."""
from __future__ import annotations

import json
from pathlib import Path

from build_c1_rank8_datum_codes import AMPLITUDE, C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import code_cell, source


H4 = ("++++", "++--", "+-+-", "+--+")
CODE_INDICES = tuple((parent_code, child_code) for parent_code in range(4) for child_code in range(1, 4))


def leaf_code(parent_code: int, child_code: int) -> str:
    signs = []
    for parent in range(4):
        for child in range(4):
            parent_sign = 1 if H4[parent_code][parent] == "+" else -1
            child_sign = 1 if H4[child_code][child] == "+" else -1
            signs.append("+" if parent_sign * child_sign > 0 else "-")
    return "".join(signs)


def datum16_cell(parent_code: int, child_code: int) -> dict:
    code = leaf_code(parent_code, child_code)
    source_code = f'''# Nested 16-bin row-balanced per-well datum Hadamard probe.
import hashlib as _d16_hashlib
import json as _d16_json
from pathlib import Path as _D16Path

import numpy as _d16_np
import pandas as _d16_pd

_D16_PARENT_CODE_INDEX = {parent_code!r}
_D16_CHILD_CODE_INDEX = {child_code!r}
_D16_PARENT_CODE = {H4[parent_code]!r}
_D16_CHILD_CODE = {H4[child_code]!r}
_D16_LEAF_CODE = {code!r}
_D16_AMPLITUDE = {AMPLITUDE!r}
_D16_WORK = _D16Path('/kaggle/working') if _D16Path('/kaggle/working').exists() else _D16Path('.')
_D16_SUB = _D16_WORK / 'submission.csv'

_d16_base = _d16_pd.read_csv(_D16_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_d16_base['tvt'] = _d16_pd.to_numeric(_d16_base['tvt'], errors='coerce')
if not _d16_np.isfinite(_d16_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('datum16 received non-finite rank-8 anchor')
_d16_base.to_csv(_D16_WORK / 'submission_before_datum16_code.csv', index=False)
_d16_parts = _d16_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _d16_parts.shape[1] != 2:
    raise RuntimeError('datum16 could not parse run-local submission IDs')
_d16_base['_well'] = _d16_parts[0].astype(str)
_d16_counts = _d16_base['_well'].value_counts().to_dict()

# Reproduce the exact four parent groups used by the already-scored codes.
_d16_parent_rows = [0, 0, 0, 0]
_d16_parent_wells = [[], [], [], []]
_d16_well_parent = {{}}
for _d16_well, _d16_rows in sorted(_d16_counts.items(), key=lambda item: (-item[1], item[0])):
    _d16_parent = min(range(4), key=lambda value: (_d16_parent_rows[value], value))
    _d16_well_parent[str(_d16_well)] = int(_d16_parent)
    _d16_parent_rows[_d16_parent] += int(_d16_rows)
    _d16_parent_wells[_d16_parent].append(str(_d16_well))

# Split each parent independently into four row-balanced child groups.
_d16_leaf_rows = [0] * 16
_d16_leaf_wells = [[] for _ in range(16)]
_d16_well_leaf = {{}}
for _d16_parent in range(4):
    _d16_child_rows = [0, 0, 0, 0]
    _d16_members = sorted(
        ((_d16_well, int(_d16_counts[_d16_well])) for _d16_well in _d16_parent_wells[_d16_parent]),
        key=lambda item: (-item[1], item[0]),
    )
    for _d16_well, _d16_rows in _d16_members:
        _d16_child = min(range(4), key=lambda value: (_d16_child_rows[value], value))
        _d16_leaf = 4 * _d16_parent + _d16_child
        _d16_well_leaf[str(_d16_well)] = int(_d16_leaf)
        _d16_child_rows[_d16_child] += int(_d16_rows)
        _d16_leaf_rows[_d16_leaf] += int(_d16_rows)
        _d16_leaf_wells[_d16_leaf].append(str(_d16_well))

_d16_leaf = _d16_base['_well'].map(_d16_well_leaf).to_numpy(dtype=int)
_d16_signs = _d16_np.asarray([1.0 if char == '+' else -1.0 for char in _D16_LEAF_CODE])
_d16_move = _D16_AMPLITUDE * _d16_signs[_d16_leaf]
_d16_values = _d16_base['tvt'].to_numpy(dtype=float) + _d16_move
if not _d16_np.isfinite(_d16_values).all():
    raise RuntimeError('datum16 produced non-finite values')
_d16_final = _d16_base[['id']].copy()
_d16_final['tvt'] = _d16_values
_d16_final.to_csv(_D16_SUB, index=False)

def _d16_sha(path):
    return _d16_hashlib.sha256(_D16Path(path).read_bytes()).hexdigest()

_DATUM16_CODE_AUDIT = {{
    'parent_code_index': int(_D16_PARENT_CODE_INDEX),
    'child_code_index': int(_D16_CHILD_CODE_INDEX),
    'parent_code': _D16_PARENT_CODE,
    'child_code': _D16_CHILD_CODE,
    'leaf_code': _D16_LEAF_CODE,
    'amplitude': float(_D16_AMPLITUDE),
    'partition': 'nested_greedy_row_balanced_run_local_wells_4x4',
    'parent_bin_rows': [int(value) for value in _d16_parent_rows],
    'leaf_bin_rows': [int(value) for value in _d16_leaf_rows],
    'leaf_bin_row_fractions': [float(value / len(_d16_final)) for value in _d16_leaf_rows],
    'leaf_bin_well_counts': [int(len(value)) for value in _d16_leaf_wells],
    'fixed_public_ids_used': False,
    'rows': int(len(_d16_final)),
    'wells': int(len(_d16_well_leaf)),
    'mean_squared_move': float(_d16_np.mean(_d16_move * _d16_move)),
    'min_move': float(_d16_np.min(_d16_move)),
    'max_move': float(_d16_np.max(_d16_move)),
    'base_sha256': _d16_sha(_D16_WORK / 'submission_before_datum16_code.csv'),
    'final_sha256': _d16_sha(_D16_SUB),
}}
(_D16_WORK / 'datum16_code_audit.json').write_text(
    _d16_json.dumps(_DATUM16_CODE_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _D16_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = f'c1_rank8_datum16_a{{_D16_PARENT_CODE_INDEX}}b{{_D16_CHILD_CODE_INDEX}}'
print('datum16 code audit:', _d16_json.dumps(_DATUM16_CODE_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(source_code)


def build(parent_code: int, child_code: int) -> Path:
    notebook = _load_base()
    notebook["cells"].append(
        c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8)
    )
    notebook["cells"].append(datum16_cell(parent_code, child_code))
    slug = f"rogii-c1r8-d16-a{parent_code}b{child_code}"
    return _write(
        notebook,
        slug,
        f"c1_rank8_datum16_hadamard_a{parent_code}_b{child_code}",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum16_code'] = _DATUM16_CODE_AUDIT",
    )


def main() -> None:
    for parent_code, child_code in CODE_INDICES:
        output = build(parent_code, child_code)
        notebook = json.loads(output.read_text(encoding="utf-8"))
        joined = "\n".join(source(cell) for cell in notebook["cells"])
        slug = output.parent.name
        metadata = json.loads(
            (Path("kaggle") / slug / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["id"] == f"qwer556617123/{slug}"
        assert f"_D16_LEAF_CODE = {leaf_code(parent_code, child_code)!r}" in joined
        assert f"_C1_BIN_ALPHAS = {C1_ALPHAS!r}" in joined
        assert "nested_greedy_row_balanced_run_local_wells_4x4" in joined
        assert "00e12e8b" not in joined
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile(source(cell), f"{slug}:code", "exec")
        print(f"built and checked {output}")


if __name__ == "__main__":
    main()
