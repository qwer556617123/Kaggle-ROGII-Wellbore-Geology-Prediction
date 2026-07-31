"""Build HMM + neighboring structural path + Prefix-GR RF well-bias fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


FINAL_AUDIT_CELL = r'''# Final orthogonal structure/well-bias fusion audit.
import hashlib as _fw_hashlib
import json as _fw_json
import time as _fw_time
from pathlib import Path as _FwPath
import numpy as _fw_np
import pandas as _fw_pd

_fw_work = _FwPath('/kaggle/working') if _FwPath('/kaggle/working').exists() else _FwPath('.')
_fw_data = _FwPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_fw_data / 'sample_submission.csv').exists():
    _fw_data = _FwPath('/kaggle/input/rogii-wellbore-geology-prediction')
_fw_anchor_path = _fw_work / 'submission_before_neighbor_structure.csv'
_fw_structure_path = _fw_work / 'submission_before_well_bias.csv'
_fw_final_path = _fw_work / 'submission.csv'
_fw_sample = _fw_pd.read_csv(_fw_data / 'sample_submission.csv')[['id']]
_fw_anchor = _fw_pd.read_csv(_fw_anchor_path)[['id', 'tvt']]
_fw_structure = _fw_pd.read_csv(_fw_structure_path)[['id', 'tvt']]
_fw_final = _fw_pd.read_csv(_fw_final_path)[['id', 'tvt']]
for _fw_frame in (_fw_sample, _fw_anchor, _fw_structure, _fw_final):
    _fw_frame['id'] = _fw_frame['id'].astype(str)
if not all(_fw_frame['id'].equals(_fw_sample['id']) for _fw_frame in (_fw_anchor, _fw_structure, _fw_final)):
    raise RuntimeError('fusion component/sample id mismatch')
for _fw_frame in (_fw_anchor, _fw_structure, _fw_final):
    if not _fw_np.isfinite(_fw_frame['tvt'].to_numpy(dtype=float)).all():
        raise RuntimeError('fusion component contains non-finite TVT')

def _fw_sha(path):
    return _fw_hashlib.sha256(_FwPath(path).read_bytes()).hexdigest()

_fw_anchor_values = _fw_anchor['tvt'].to_numpy(dtype=float)
_fw_structure_values = _fw_structure['tvt'].to_numpy(dtype=float)
_fw_final_values = _fw_final['tvt'].to_numpy(dtype=float)
_fw_structure_move = _fw_structure_values - _fw_anchor_values
_fw_bias_move = _fw_final_values - _fw_structure_values
_fw_total_move = _fw_final_values - _fw_anchor_values
_fw_corr = float(_fw_np.corrcoef(_fw_structure_move, _fw_bias_move)[0, 1])
_fw_neighbor_audit = _fw_json.loads(
    (_fw_work / 'neighbor_structure_audit.json').read_text(encoding='utf-8')
)
_fw_audit = {
    'strategy': 'pkadopt_wellcal_hmm_neighbor_structure010_plus_prefix_gr_rf_well_bias',
    'rows': int(len(_fw_final)),
    'id_order_matches_sample': True,
    'runtime_sec': float(_fw_time.time() - _SD_STARTED),
    'structure_bias_direction_correlation': _fw_corr,
    'structure_move_rms_ft': float(_fw_np.sqrt(_fw_np.mean(_fw_structure_move ** 2))),
    'well_bias_move_rms_ft': float(_fw_np.sqrt(_fw_np.mean(_fw_bias_move ** 2))),
    'total_move_rms_ft': float(_fw_np.sqrt(_fw_np.mean(_fw_total_move ** 2))),
    'total_move_max_abs_ft': float(_fw_np.max(_fw_np.abs(_fw_total_move))),
    'anchor_sha256': _fw_sha(_fw_anchor_path),
    'structure_sha256': _fw_sha(_fw_structure_path),
    'final_sha256': _fw_sha(_fw_final_path),
    'same_id_excluded': bool(_fw_neighbor_audit['native_mask_gate']['same_id_excluded']),
    'neighbor_wells': _fw_neighbor_audit['wells'],
    'well_bias_rows': _wb_audit,
}
(_fw_work / 'hmm_structure_wellbias_fusion_audit.json').write_text(
    _fw_json.dumps(_fw_audit, indent=2, sort_keys=True), encoding='utf-8'
)
print('Structure/well-bias fusion audit:', _fw_json.dumps(_fw_audit, sort_keys=True), flush=True)
'''


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-name", default="rogii-pkadopt-hmm-neighbor-structure010"
    )
    parser.add_argument(
        "--external-notebook",
        type=Path,
        default=ROOT
        / "kaggle"
        / "external_reviews"
        / "frontier_20260719"
        / "wellbias_rf"
        / "rogii-det-wellbias-prefix-rf-r1.ipynb",
    )
    parser.add_argument(
        "--target-name", default="rogii-pkadopt-hmm-structure-wellbias-fusion"
    )
    args = parser.parse_args()

    source_dir = ROOT / "kaggle" / args.source_name
    source_notebook = source_dir / f"{args.source_name}.ipynb"
    notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
    external = json.loads(args.external_notebook.read_text(encoding="utf-8"))
    matches = [
        cell
        for cell in external.get("cells", [])
        if "Predictor-driven per-well datum correction" in _cell_source(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one well-bias cell, found {len(matches)}")
    well_bias_source = _cell_source(matches[0])
    old_loop = "for _wb_path in sorted(_wb_glob.glob(str(_WB_WORK / '*.csv'))):"
    if old_loop not in well_bias_source:
        raise RuntimeError("could not locate broad well-bias CSV loop")
    well_bias_source = well_bias_source.replace(
        old_loop,
        "_wb_anchor_path = _WB_WORK / 'submission_before_well_bias.csv'\n"
        "    _wb_base[['id', 'tvt']].to_csv(_wb_anchor_path, index=False)\n"
        "    for _wb_path in [_WB_WORK / 'submission.csv']:",
        1,
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": well_bias_source,
        }
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": FINAL_AUDIT_CELL,
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
