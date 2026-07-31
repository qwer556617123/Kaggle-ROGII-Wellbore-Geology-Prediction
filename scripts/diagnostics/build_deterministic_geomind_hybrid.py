"""Build a 75/25 hybrid of the verified deterministic anchor and GeoMind routing."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


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


def build(anchor_path: Path, geomind_path: Path, output_dir: Path) -> Path:
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    geomind = json.loads(geomind_path.read_text(encoding="utf-8"))
    cells = []
    for original in anchor["cells"]:
        source = _source(original)
        if "Final submission audit. This cell never changes predictions." in source:
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        cells.append(cell)
    cells.append(
        _code_cell(
            "from pathlib import Path as _DgPath\n"
            "import pandas as _dg_pd\n"
            "_dg_work = _DgPath('/kaggle/working') if _DgPath('/kaggle/working').exists() else _DgPath('.')\n"
            "_dg_pd.read_csv(_dg_work / 'submission.csv')[['id', 'tvt']].to_csv(_dg_work / 'deterministic7166_anchor.csv', index=False)\n"
            "print('saved deterministic7166_anchor.csv before GeoMind routing', flush=True)\n"
        )
    )
    for original in geomind["cells"]:
        if original.get("cell_type") != "code" or not _source(original).strip():
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        cells.append(cell)

    blend = r'''# Conservative deterministic/GeoMind hybrid.
import hashlib as _dgh_hashlib
import json as _dgh_json
import time as _dgh_time
from pathlib import Path as _DghPath
import numpy as _dgh_np
import pandas as _dgh_pd
_dgh_work = _DghPath('/kaggle/working') if _DghPath('/kaggle/working').exists() else _DghPath('.')
_dgh_base = _dgh_pd.read_csv(_dgh_work / 'deterministic7166_anchor.csv')[['id', 'tvt']]
_dgh_geo = _dgh_pd.read_csv(_dgh_work / 'submission.csv')[['id', 'tvt']]
_dgh_base['id'] = _dgh_base['id'].astype(str); _dgh_geo['id'] = _dgh_geo['id'].astype(str)
_dgh_merged = _dgh_base.rename(columns={'tvt': 'base'}).merge(
    _dgh_geo.rename(columns={'tvt': 'geo'}), on='id', how='inner', validate='one_to_one')
if len(_dgh_merged) != len(_dgh_base) or len(_dgh_merged) != len(_dgh_geo):
    raise RuntimeError('deterministic/GeoMind id mismatch')
_dgh_weight = 0.25
_dgh_out = _dgh_merged[['id']].copy()
_dgh_out['tvt'] = (1.0 - _dgh_weight) * _dgh_merged['base'] + _dgh_weight * _dgh_merged['geo']
_dgh_path = _dgh_work / 'submission.csv'
_dgh_out.to_csv(_dgh_path, index=False)
_dgh_sample = _dgh_pd.read_csv('/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv')[['id']]
_dgh_sample['id'] = _dgh_sample['id'].astype(str)
_dgh_values = _dgh_out['tvt'].to_numpy(dtype=float)
if not _dgh_out['id'].equals(_dgh_sample['id']) or not _dgh_np.isfinite(_dgh_values).all():
    raise RuntimeError('hybrid output audit failed')
_dgh_diff = _dgh_merged['geo'].to_numpy(dtype=float) - _dgh_merged['base'].to_numpy(dtype=float)
def _dgh_sha(path):
    digest = _dgh_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()
_dgh_audit = {
    'strategy': 'deterministic7166_geomind_hybrid',
    'geomind_weight': _dgh_weight,
    'rows': int(len(_dgh_out)),
    'id_order_matches_sample': True,
    'rmse_geo_vs_anchor': float(_dgh_np.sqrt(_dgh_np.mean(_dgh_diff ** 2))),
    'mean_abs_geo_vs_anchor': float(_dgh_np.mean(_dgh_np.abs(_dgh_diff))),
    'p95_abs_geo_vs_anchor': float(_dgh_np.quantile(_dgh_np.abs(_dgh_diff), 0.95)),
    'submission_sha256': _dgh_sha(_dgh_path),
    'runtime_sec': float(_dgh_time.time() - _DETERMINISTIC7166_START),
}
with open(_dgh_work / 'deterministic_geomind_hybrid_audit.json', 'w', encoding='utf-8') as handle:
    _dgh_json.dump(_dgh_audit, handle, indent=2, sort_keys=True)
print('deterministic/GeoMind hybrid audit:', _dgh_audit, flush=True)
'''
    cells.append(_code_cell(blend))
    anchor["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "rogii-deterministic-geomind-hybrid.ipynb"
    notebook_path.write_text(json.dumps(anchor, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "qwer556617123/rogii-deterministic-geomind-hybrid",
        "title": "rogii-deterministic-geomind-hybrid",
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
        "--geomind",
        type=Path,
        default=Path(os.environ.get("TEMP", ".")) / "rogii_survey" / "geomind" / "geomind-trajectory-intelligence-core.ipynb",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("kaggle/rogii-deterministic-geomind-hybrid")
    )
    args = parser.parse_args()
    build(args.anchor, args.geomind, args.output_dir)


if __name__ == "__main__":
    main()
