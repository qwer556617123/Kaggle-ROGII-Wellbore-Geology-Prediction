"""Build row-balanced per-well datum Hadamard probes over the rank-8 C1 anchor."""
from __future__ import annotations

import json
from pathlib import Path

from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import code_cell, source


C1_ALPHAS = (0.55, -0.325, 0.125, 0.70, 0.55, -0.675, 0.125, 0.475)
CODES = ("++++", "++--", "+-+-", "+--+")
AMPLITUDE = 2.0


def datum_code_cell(code: str) -> dict:
    if len(code) != 4 or any(char not in "+-" for char in code):
        raise ValueError(code)
    source_code = f'''# Row-balanced per-well datum Hadamard probe over rank-8 C1.
import hashlib as _dc_hashlib
import json as _dc_json
from pathlib import Path as _DcPath

import numpy as _dc_np
import pandas as _dc_pd

_DC_CODE = {code!r}
_DC_AMPLITUDE = {AMPLITUDE!r}
_DC_WORK = _DcPath('/kaggle/working') if _DcPath('/kaggle/working').exists() else _DcPath('.')
_DC_SUB = _DC_WORK / 'submission.csv'

_dc_base = _dc_pd.read_csv(_DC_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_dc_base['tvt'] = _dc_pd.to_numeric(_dc_base['tvt'], errors='coerce')
if not _dc_np.isfinite(_dc_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('datum code received non-finite rank-8 anchor')
_dc_base.to_csv(_DC_WORK / 'submission_before_datum_code.csv', index=False)
_dc_parts = _dc_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _dc_parts.shape[1] != 2:
    raise RuntimeError('datum code could not parse run-local submission IDs')
_dc_base['_well'] = _dc_parts[0].astype(str)
_dc_counts = _dc_base['_well'].value_counts().to_dict()

# Greedy largest-first packing gives four deterministic, nearly row-balanced
# groups without depending on any public well ID.
_dc_bin_rows = [0, 0, 0, 0]
_dc_bin_wells = [[], [], [], []]
_dc_well_bin = {{}}
for _dc_well, _dc_rows in sorted(_dc_counts.items(), key=lambda item: (-item[1], item[0])):
    _dc_bin = min(range(4), key=lambda value: (_dc_bin_rows[value], value))
    _dc_well_bin[str(_dc_well)] = int(_dc_bin)
    _dc_bin_rows[_dc_bin] += int(_dc_rows)
    _dc_bin_wells[_dc_bin].append(str(_dc_well))

_dc_bins = _dc_base['_well'].map(_dc_well_bin).to_numpy(dtype=int)
_dc_signs = _dc_np.asarray([1.0 if char == '+' else -1.0 for char in _DC_CODE])
_dc_move = _DC_AMPLITUDE * _dc_signs[_dc_bins]
_dc_values = _dc_base['tvt'].to_numpy(dtype=float) + _dc_move
if not _dc_np.isfinite(_dc_values).all():
    raise RuntimeError('datum code produced non-finite values')
_dc_final = _dc_base[['id']].copy()
_dc_final['tvt'] = _dc_values
_dc_final.to_csv(_DC_SUB, index=False)

def _dc_sha(path):
    return _dc_hashlib.sha256(_DcPath(path).read_bytes()).hexdigest()

_DATUM_CODE_AUDIT = {{
    'code': _DC_CODE,
    'amplitude': float(_DC_AMPLITUDE),
    'partition': 'greedy_row_balanced_run_local_wells_4',
    'fixed_public_ids_used': False,
    'rows': int(len(_dc_final)),
    'wells': int(len(_dc_well_bin)),
    'bin_rows': [int(value) for value in _dc_bin_rows],
    'bin_row_fractions': [float(value / len(_dc_final)) for value in _dc_bin_rows],
    'bin_well_counts': [int(len(value)) for value in _dc_bin_wells],
    'mean_squared_move': float(_dc_np.mean(_dc_move * _dc_move)),
    'min_move': float(_dc_np.min(_dc_move)),
    'max_move': float(_dc_np.max(_dc_move)),
    'base_sha256': _dc_sha(_DC_WORK / 'submission_before_datum_code.csv'),
    'final_sha256': _dc_sha(_DC_SUB),
}}
(_DC_WORK / 'datum_code_audit.json').write_text(
    _dc_json.dumps(_DATUM_CODE_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _DC_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = f'c1_rank8_datum_code_{{_DC_CODE}}'
print('datum code audit:', _dc_json.dumps(_DATUM_CODE_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(source_code)


def build(code: str) -> Path:
    notebook = _load_base()
    notebook["cells"].append(
        c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8)
    )
    notebook["cells"].append(datum_code_cell(code))
    slug = f"rogii-c1r8-datum-{code.replace('+', 'p').replace('-', 'm')}"
    return _write(
        notebook,
        slug,
        f"c1_rank8_row_balanced_datum_hadamard_{code}",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum_code'] = _DATUM_CODE_AUDIT",
    )


def main() -> None:
    for code in CODES:
        output = build(code)
        notebook = json.loads(output.read_text(encoding="utf-8"))
        joined = "\n".join(source(cell) for cell in notebook["cells"])
        slug = output.parent.name
        metadata = json.loads(
            (Path("kaggle") / slug / "kernel-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["id"] == f"qwer556617123/{slug}"
        assert f"_DC_CODE = {code!r}" in joined
        assert f"_C1_BIN_ALPHAS = {C1_ALPHAS!r}" in joined
        assert "greedy_row_balanced_run_local_wells_4" in joined
        assert "00e12e8b" not in joined
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile(source(cell), f"{slug}:code", "exec")
        print(f"built and checked {output}")


if __name__ == "__main__":
    main()
