"""Build a runtime-safe HMM anchor plus public OOF well-bias RF overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FINAL_AUDIT_CELL = r'''# Final well-bias rerun audit.
import hashlib as _wba_hashlib
import json as _wba_json
import time as _wba_time
from pathlib import Path as _WbaPath
import numpy as _wba_np
import pandas as _wba_pd

_wba_work = _WbaPath('/kaggle/working') if _WbaPath('/kaggle/working').exists() else _WbaPath('.')
_wba_data = _WbaPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_wba_data / 'sample_submission.csv').exists():
    _wba_data = _WbaPath('/kaggle/input/rogii-wellbore-geology-prediction')
_wba_path = _wba_work / 'submission.csv'
_wba_before_path = _wba_work / 'submission_before_well_bias.csv'
_wba_sample = _wba_pd.read_csv(_wba_data / 'sample_submission.csv')[['id']]
_wba_final = _wba_pd.read_csv(_wba_path)[['id', 'tvt']]
_wba_before = _wba_pd.read_csv(_wba_before_path)[['id', 'tvt']]
for _wba_frame in (_wba_sample, _wba_final, _wba_before):
    _wba_frame['id'] = _wba_frame['id'].astype(str)
if not _wba_final['id'].equals(_wba_sample['id']):
    raise RuntimeError('well-bias submission/sample id mismatch')
if not _wba_before['id'].equals(_wba_sample['id']):
    raise RuntimeError('well-bias anchor/sample id mismatch')
_wba_values = _wba_final['tvt'].to_numpy(dtype=float)
_wba_base_values = _wba_before['tvt'].to_numpy(dtype=float)
if not _wba_np.isfinite(_wba_values).all():
    raise RuntimeError('well-bias output contains non-finite TVT')

def _wba_sha(path):
    return _wba_hashlib.sha256(_WbaPath(path).read_bytes()).hexdigest()

_wba_move = _wba_values - _wba_base_values
_wba_audit = {
    'strategy': 'pkadopt_wellcal_hmm_plus_prefix_gr_rf_well_bias',
    'source': 'chiekhalloul/rogii-det-wellbias-prefix-rf-r1',
    'rows': int(len(_wba_final)),
    'id_order_matches_sample': True,
    'runtime_sec': float(_wba_time.time() - _SD_STARTED),
    'mean_move_ft': float(_wba_np.mean(_wba_move)),
    'mean_abs_move_ft': float(_wba_np.mean(_wba_np.abs(_wba_move))),
    'rms_move_ft': float(_wba_np.sqrt(_wba_np.mean(_wba_move ** 2))),
    'max_abs_move_ft': float(_wba_np.max(_wba_np.abs(_wba_move))),
    'anchor_sha256': _wba_sha(_wba_before_path),
    'final_sha256': _wba_sha(_wba_path),
    'well_bias_rows': _wb_audit,
}
(_wba_work / 'hmm_well_bias_audit.json').write_text(
    _wba_json.dumps(_wba_audit, indent=2, sort_keys=True), encoding='utf-8'
)
print('HMM well-bias final audit:', _wba_json.dumps(_wba_audit, sort_keys=True), flush=True)
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
        "--target-name", default="rogii-pkadopt-hmm-wellbias-rf"
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
    if old_loop in well_bias_source:
        raise RuntimeError("failed to narrow the well-bias CSV writer")
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
    metadata = json.loads((source_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
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
