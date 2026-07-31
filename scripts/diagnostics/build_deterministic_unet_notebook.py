"""Add a native-prefix U-Net datum nudge to the verified deterministic 7.166 anchor."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


LEAKAGE_BLOCK = """            if wid in tw_ids:
                htr = _upd.read_csv(trd / f'{wid}__horizontal_well.csv'); hw = hw.copy()
                hw['TVT_input'] = htr['TVT_input'].values
                tp = trd / f'{wid}__typewell.csv'; td = _utemp.mkdtemp()
                hp = _uo.path.join(td, f'{wid}__horizontal_well.csv'); hw.to_csv(hp, index=False)
"""


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build(anchor_path: Path, frontier_path: Path, output_dir: Path) -> Path:
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    unet_source = None
    for cell in frontier["cells"]:
        source = _source(cell)
        if "U2Net LIGHT inference" in source:
            unet_source = source
            break
    if unet_source is None:
        raise RuntimeError("U2Net inference cell not found")
    if LEAKAGE_BLOCK not in unet_source:
        raise RuntimeError("same-ID train TVT_input replacement block not found")
    unet_source = unet_source.replace(LEAKAGE_BLOCK, "")
    unet_source = unet_source.replace("    _rows = []\n", "    _rows = []\n    _path_rows = []\n")
    unet_source = unet_source.replace(
        "        _rows.append((wd['wid'], float(_un.mean(pr))))   # per-well U2Net datum\n",
        "        _rows.append((wd['wid'], float(_un.mean(pr))))   # per-well U2Net datum\n"
        "        _path_rows.extend((f\"{wd['wid']}_{int(ri)}\", float(pv)) for ri, pv in zip(wd['ev'], pr))\n",
    )
    unet_source = unet_source.replace(
        "    _udf.to_csv(_pw / 'unet_datum.csv', index=False)\n",
        "    _udf.to_csv(_pw / 'unet_datum.csv', index=False)\n"
        "    _upd.DataFrame(_path_rows, columns=['id', 'unet_tvt']).to_csv(_pw / 'unet_path.csv', index=False)\n",
    )

    cells = []
    for cell in anchor["cells"]:
        source = _source(cell)
        if "Final submission audit. This cell never changes predictions." in source:
            continue
        cleaned = copy.deepcopy(cell)
        cleaned["execution_count"] = None
        cleaned["outputs"] = []
        cells.append(cleaned)
    cells.append(_code_cell(unet_source))

    nudge = r'''# Conservative zero-mean native-prefix U-Net shape nudge.
import json as _du_json
from pathlib import Path as _DuPath
import numpy as _du_np
import pandas as _du_pd
_du_work = _DuPath('/kaggle/working') if _DuPath('/kaggle/working').exists() else _DuPath('.')
_du_sub_path = _du_work / 'submission.csv'
_du_sub = _du_pd.read_csv(_du_sub_path)[['id', 'tvt']]
_du_parts = _du_sub['id'].astype(str).str.rsplit('_', n=1, expand=True)
_du_sub['well'] = _du_parts[0]
_du_sub['row_idx'] = _du_parts[1].astype(int)
_du_rows = []
_du_unet_path = _du_work / 'unet_path.csv'
if not _du_unet_path.exists():
    raise RuntimeError('native-prefix U-Net did not produce unet_path.csv')
_du_path = _du_pd.read_csv(_du_unet_path)
_du_map = dict(zip(_du_path['id'].astype(str), _du_path['unet_tvt'].astype(float)))
for _du_well, _du_group in _du_sub.groupby('well', sort=False):
    _du_idx = _du_group.index.to_numpy()
    _du_base = _du_sub.loc[_du_idx, 'tvt'].to_numpy(dtype=float)
    _du_ids = _du_sub.loc[_du_idx, 'id'].astype(str).tolist()
    _du_target = _du_np.array([_du_map.get(rid, _du_np.nan) for rid in _du_ids], dtype=float)
    _du_finite = _du_np.isfinite(_du_target)
    _du_delta = _du_target - _du_base
    _du_basis = _du_delta - float(_du_np.nanmean(_du_delta)) if _du_finite.all() else _du_np.zeros_like(_du_base)
    _du_mean_abs = float(_du_np.mean(_du_np.abs(_du_basis)))
    _du_p95 = float(_du_np.quantile(_du_np.abs(_du_basis), 0.95))
    _du_gate = bool(_du_finite.all() and 1.0 <= _du_mean_abs <= 15.0 and _du_p95 <= 25.0)
    _du_ramp = 1.0 - _du_np.exp(-_du_np.arange(len(_du_base), dtype=float) / max(80.0, 0.12 * max(1, len(_du_base))))
    _du_move = _du_np.clip(0.25 * _du_basis, -6.0, 6.0) * _du_ramp if _du_gate else _du_np.zeros_like(_du_base)
    _du_sub.loc[_du_idx, 'tvt'] = _du_base + _du_move
    _du_rows.append({
        'well': str(_du_well), 'rows': int(len(_du_idx)),
        'path_rows': int(_du_finite.sum()), 'basis_mean': float(_du_np.mean(_du_basis)),
        'basis_mean_abs': _du_mean_abs, 'basis_p95_abs': _du_p95, 'gate': _du_gate,
        'mean_abs_move': float(_du_np.mean(_du_np.abs(_du_move))),
        'max_abs_move': float(_du_np.max(_du_np.abs(_du_move))),
    })
_du_sub[['id', 'tvt']].to_csv(_du_sub_path, index=False)
_du_report = _du_pd.DataFrame(_du_rows)
_du_report.to_csv(_du_work / 'deterministic_unet_nudge_report.csv', index=False)
print('deterministic U-Net nudge:', {
    'unet_path_rows': int(len(_du_path)),
    'gated_wells': int(_du_report['gate'].sum()),
    'moved_rows': int(_du_report.loc[_du_report['gate'], 'rows'].sum()),
}, flush=True)
'''
    cells.append(_code_cell(nudge))

    audit = r'''# Final output audit; predictions are not modified.
import hashlib as _dua_hashlib
import json as _dua_json
import time as _dua_time
from pathlib import Path as _DuaPath
import numpy as _dua_np
import pandas as _dua_pd
_dua_work = _DuaPath('/kaggle/working') if _DuaPath('/kaggle/working').exists() else _DuaPath('.')
_dua_data = _DuaPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
_dua_path = _dua_work / 'submission.csv'
_dua_sub = _dua_pd.read_csv(_dua_path)[['id', 'tvt']]
_dua_sample = _dua_pd.read_csv(_dua_data / 'sample_submission.csv')[['id']]
_dua_sub['id'] = _dua_sub['id'].astype(str); _dua_sample['id'] = _dua_sample['id'].astype(str)
_dua_values = _dua_sub['tvt'].to_numpy(dtype=float)
if len(_dua_sub) != len(_dua_sample) or not _dua_sub['id'].equals(_dua_sample['id']):
    raise RuntimeError('deterministic U-Net output is not sample aligned')
if not _dua_np.isfinite(_dua_values).all():
    raise RuntimeError('deterministic U-Net output contains non-finite values')
def _dua_sha(path):
    digest = _dua_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()
_dua_report = _dua_pd.read_csv(_dua_work / 'deterministic_unet_nudge_report.csv')
_dua_audit = {
    'strategy': 'deterministic7166_native_unet_zero_mean_shape',
    'same_id_train_tvt_input_used': False,
    'gold_profile': 'conservative',
    'global_bias_shift_ft': -0.40,
    'unet_weight': 0.25,
    'unet_min_mean_abs_basis': 1.0,
    'unet_max_mean_abs_basis': 15.0,
    'unet_max_p95_abs_basis': 25.0,
    'unet_max_abs_move': 6.0,
    'unet_path_rows': int(_dua_report['path_rows'].sum()),
    'gated_wells': int(_dua_report['gate'].sum()),
    'rows': int(len(_dua_sub)),
    'id_order_matches_sample': True,
    'submission_sha256': _dua_sha(_dua_path),
    'runtime_sec': float(_dua_time.time() - _DETERMINISTIC7166_START),
}
with open(_dua_work / 'deterministic_unet_audit.json', 'w', encoding='utf-8') as handle:
    _dua_json.dump(_dua_audit, handle, indent=2, sort_keys=True)
print('deterministic U-Net final audit:', _dua_audit, flush=True)
'''
    cells.append(_code_cell(audit))
    anchor["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "rogii-deterministic-unet.ipynb"
    notebook_path.write_text(json.dumps(anchor, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "qwer556617123/rogii-deterministic-unet",
        "title": "rogii-deterministic-unet",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "phongnguyn23021656/koolbox-offline",
            "fleongg/rogii-claude-models-pub",
            "sangrampatil5150/unet-rogii",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {notebook_path} with {len(cells)} cells")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path("kaggle/rogii-deterministic7166-rebuild/rogii-deterministic7166-rebuild.ipynb"),
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path(os.environ.get("TEMP", ".")) / "rogii_survey" / "frontierfull" / "rogii-frontier-full.ipynb",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("kaggle/rogii-deterministic-unet"))
    args = parser.parse_args()
    build(args.anchor, args.frontier, args.output_dir)


if __name__ == "__main__":
    main()
