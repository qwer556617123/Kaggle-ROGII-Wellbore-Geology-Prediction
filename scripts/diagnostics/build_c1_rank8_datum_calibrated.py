"""Build the LB-decoded four-group datum calibration over rank-8 C1."""
from __future__ import annotations

import json
from pathlib import Path

from build_c1_rank8_datum_codes import C1_ALPHAS
from build_hmm010_structural_tomography_batch import _load_base, _write, c1_cell
from build_stopdose_breakthrough_notebooks import code_cell, source


SLUG = "rogii-c1r8-datum-calibrated"
BIN_PROJECTIONS = (-0.0535985625, 0.1933401875, -0.2104801875, 0.1641448125)
OFFSET_CAP = 2.0


def calibrated_datum_cell() -> dict:
    code = f'''# LB-decoded row-balanced per-well datum calibration.
import hashlib as _dcal_hashlib
import json as _dcal_json
from pathlib import Path as _DcalPath

import numpy as _dcal_np
import pandas as _dcal_pd

_DCAL_PROJECTIONS = {BIN_PROJECTIONS!r}
_DCAL_OFFSET_CAP = {OFFSET_CAP!r}
_DCAL_WORK = _DcalPath('/kaggle/working') if _DcalPath('/kaggle/working').exists() else _DcalPath('.')
_DCAL_SUB = _DCAL_WORK / 'submission.csv'

_dcal_base = _dcal_pd.read_csv(_DCAL_SUB, dtype={{'id': 'string'}})[['id', 'tvt']]
_dcal_base['tvt'] = _dcal_pd.to_numeric(_dcal_base['tvt'], errors='coerce')
if not _dcal_np.isfinite(_dcal_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('datum calibration received non-finite rank-8 anchor')
_dcal_base.to_csv(_DCAL_WORK / 'submission_before_datum_calibration.csv', index=False)
_dcal_parts = _dcal_base['id'].astype(str).str.rsplit('_', n=1, expand=True)
if _dcal_parts.shape[1] != 2:
    raise RuntimeError('datum calibration could not parse run-local submission IDs')
_dcal_base['_well'] = _dcal_parts[0].astype(str)
_dcal_counts = _dcal_base['_well'].value_counts().to_dict()

_dcal_bin_rows = [0, 0, 0, 0]
_dcal_bin_wells = [[], [], [], []]
_dcal_well_bin = {{}}
for _dcal_well, _dcal_rows in sorted(_dcal_counts.items(), key=lambda item: (-item[1], item[0])):
    _dcal_bin = min(range(4), key=lambda value: (_dcal_bin_rows[value], value))
    _dcal_well_bin[str(_dcal_well)] = int(_dcal_bin)
    _dcal_bin_rows[_dcal_bin] += int(_dcal_rows)
    _dcal_bin_wells[_dcal_bin].append(str(_dcal_well))

_dcal_fractions = _dcal_np.asarray(_dcal_bin_rows, dtype=float) / float(len(_dcal_base))
_dcal_projection = _dcal_np.asarray(_DCAL_PROJECTIONS, dtype=float)
_dcal_raw_offsets = _dcal_np.divide(
    -_dcal_projection,
    _dcal_fractions,
    out=_dcal_np.zeros(4, dtype=float),
    where=_dcal_fractions > 0.0,
)
_dcal_offsets = _dcal_np.clip(_dcal_raw_offsets, -_DCAL_OFFSET_CAP, _DCAL_OFFSET_CAP)
_dcal_bins = _dcal_base['_well'].map(_dcal_well_bin).to_numpy(dtype=int)
_dcal_move = _dcal_offsets[_dcal_bins]
_dcal_values = _dcal_base['tvt'].to_numpy(dtype=float) + _dcal_move
if not _dcal_np.isfinite(_dcal_values).all():
    raise RuntimeError('datum calibration produced non-finite values')
_dcal_final = _dcal_base[['id']].copy()
_dcal_final['tvt'] = _dcal_values
_dcal_final.to_csv(_DCAL_SUB, index=False)

def _dcal_sha(path):
    return _dcal_hashlib.sha256(_DcalPath(path).read_bytes()).hexdigest()

_DATUM_CALIBRATION_AUDIT = {{
    'bin_projection': [float(value) for value in _dcal_projection],
    'bin_rows': [int(value) for value in _dcal_bin_rows],
    'bin_row_fractions': [float(value) for value in _dcal_fractions],
    'bin_well_counts': [int(len(value)) for value in _dcal_bin_wells],
    'raw_bin_offsets': [float(value) for value in _dcal_raw_offsets],
    'deployed_bin_offsets': [float(value) for value in _dcal_offsets],
    'offset_cap': float(_DCAL_OFFSET_CAP),
    'partition': 'greedy_row_balanced_run_local_wells_4',
    'fixed_public_ids_used': False,
    'rows': int(len(_dcal_final)),
    'wells': int(len(_dcal_well_bin)),
    'mean_squared_move': float(_dcal_np.mean(_dcal_move * _dcal_move)),
    'max_abs_move': float(_dcal_np.max(_dcal_np.abs(_dcal_move))),
    'base_sha256': _dcal_sha(_DCAL_WORK / 'submission_before_datum_calibration.csv'),
    'final_sha256': _dcal_sha(_DCAL_SUB),
}}
(_DCAL_WORK / 'datum_calibration_audit.json').write_text(
    _dcal_json.dumps(_DATUM_CALIBRATION_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
globals()['FINAL_SELECTED_BASE_SOURCE'] = _DCAL_SUB
globals()['FINAL_BASE_SOURCE_LABEL'] = 'c1_rank8_row_balanced_datum_calibrated'
print('datum calibration audit:', _dcal_json.dumps(_DATUM_CALIBRATION_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def main() -> None:
    notebook = _load_base()
    notebook["cells"].append(
        c1_cell(1.0, bin_alphas=C1_ALPHAS, partition_mod=8)
    )
    notebook["cells"].append(calibrated_datum_cell())
    output = _write(
        notebook,
        SLUG,
        "c1_rank8_row_balanced_datum_calibrated",
        "_sd_audit['hmm'] = _HMM_AUDIT\n"
        "_sd_audit['c1_heel'] = _C1_AUDIT\n"
        "_sd_audit['datum_calibration'] = _DATUM_CALIBRATION_AUDIT",
    )
    notebook = json.loads(output.read_text(encoding="utf-8"))
    joined = "\n".join(source(cell) for cell in notebook["cells"])
    metadata = json.loads(
        (Path("kaggle") / SLUG / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["id"] == f"qwer556617123/{SLUG}"
    assert f"_DCAL_PROJECTIONS = {BIN_PROJECTIONS!r}" in joined
    assert f"_C1_BIN_ALPHAS = {C1_ALPHAS!r}" in joined
    assert "greedy_row_balanced_run_local_wells_4" in joined
    assert "00e12e8b" not in joined
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{SLUG}:code", "exec")
    print(f"built and checked {output}")


if __name__ == "__main__":
    main()
