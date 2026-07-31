"""Build symmetric zero-mean hierarchical-shape probes on the GeoMind -0.40 anchor."""
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


def _metadata(kernel_id: str, code_file: str) -> dict:
    return {
        "id": f"qwer556617123/{kernel_id}",
        "title": kernel_id,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_one(hier_path: Path, geomind_path: Path, output_dir: Path, sign: float) -> Path:
    hierarchy = json.loads(hier_path.read_text(encoding="utf-8"))
    geomind = json.loads(geomind_path.read_text(encoding="utf-8"))
    cells = []
    for original in hierarchy["cells"]:
        source = _source(original)
        if "Wrapper audit; predictions are not modified." in source:
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        cells.append(cell)
    cells.append(
        _code_cell(
            "from pathlib import Path as _GhPath\n"
            "import pandas as _gh_pd\n"
            "_gh_work = _GhPath('/kaggle/working') if _GhPath('/kaggle/working').exists() else _GhPath('.')\n"
            "_gh_pd.read_csv(_gh_work / 'submission.csv')[['id', 'tvt']].to_csv(_gh_work / 'hierarchical_candidate.csv', index=False)\n"
            "print('saved hierarchical_candidate.csv before GeoMind routing', flush=True)\n"
        )
    )
    for original in geomind["cells"]:
        if original.get("cell_type") != "code" or not _source(original).strip():
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        cells.append(cell)

    strategy = "plus" if sign > 0 else "minus"
    blend = f'''# GeoMind -0.40 datum plus a zero-mean hierarchical shape probe.
import hashlib as _ghs_hashlib
import json as _ghs_json
import time as _ghs_time
from pathlib import Path as _GhsPath
import numpy as _ghs_np
import pandas as _ghs_pd
_ghs_work = _GhsPath('/kaggle/working') if _GhsPath('/kaggle/working').exists() else _GhsPath('.')
_ghs_hier = _ghs_pd.read_csv(_ghs_work / 'hierarchical_candidate.csv')[['id', 'tvt']]
_ghs_geo = _ghs_pd.read_csv(_ghs_work / 'submission.csv')[['id', 'tvt']]
_ghs_hier['id'] = _ghs_hier['id'].astype(str); _ghs_geo['id'] = _ghs_geo['id'].astype(str)
_ghs = _ghs_geo.rename(columns={{'tvt': 'geo'}}).merge(
    _ghs_hier.rename(columns={{'tvt': 'hier'}}), on='id', how='inner', validate='one_to_one')
if len(_ghs) != len(_ghs_geo) or len(_ghs) != len(_ghs_hier):
    raise RuntimeError('GeoMind/hierarchical id mismatch')
_ghs_parts = _ghs['id'].str.rsplit('_', n=1, expand=True)
_ghs['well'] = _ghs_parts[0]
_ghs['base'] = _ghs['geo'].astype(float) - 0.40
_ghs['tvt'] = _ghs['base']
_ghs_rows = []
_ghs_sign = {sign!r}
_ghs_weight = 0.10
for _ghs_well, _ghs_group in _ghs.groupby('well', sort=False):
    _ghs_idx = _ghs_group.index.to_numpy()
    _ghs_basis = (_ghs.loc[_ghs_idx, 'hier'] - _ghs.loc[_ghs_idx, 'geo']).to_numpy(dtype=float)
    _ghs_basis = _ghs_basis - float(_ghs_np.mean(_ghs_basis))
    _ghs_mean_abs = float(_ghs_np.mean(_ghs_np.abs(_ghs_basis)))
    _ghs_p95 = float(_ghs_np.quantile(_ghs_np.abs(_ghs_basis), 0.95))
    _ghs_gate = bool(1.0 <= _ghs_mean_abs <= 20.0 and _ghs_p95 <= 40.0)
    _ghs_move = _ghs_np.clip(_ghs_sign * _ghs_weight * _ghs_basis, -4.0, 4.0) if _ghs_gate else _ghs_np.zeros_like(_ghs_basis)
    _ghs.loc[_ghs_idx, 'tvt'] = _ghs.loc[_ghs_idx, 'base'].to_numpy(dtype=float) + _ghs_move
    _ghs_rows.append({{
        'well': str(_ghs_well), 'rows': int(len(_ghs_idx)), 'gate': _ghs_gate,
        'basis_mean': float(_ghs_np.mean(_ghs_basis)), 'basis_mean_abs': _ghs_mean_abs,
        'basis_p95_abs': _ghs_p95, 'mean_abs_move': float(_ghs_np.mean(_ghs_np.abs(_ghs_move))),
        'max_abs_move': float(_ghs_np.max(_ghs_np.abs(_ghs_move))),
    }})
_ghs_out = _ghs[['id', 'tvt']]
_ghs_path = _ghs_work / 'submission.csv'
_ghs_out.to_csv(_ghs_path, index=False)
_ghs_report = _ghs_pd.DataFrame(_ghs_rows)
_ghs_report.to_csv(_ghs_work / 'geomind_hierarchical_shape_report.csv', index=False)
_ghs_sample = _ghs_pd.read_csv('/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv')[['id']]
_ghs_sample['id'] = _ghs_sample['id'].astype(str)
_ghs_values = _ghs_out['tvt'].to_numpy(dtype=float)
if not _ghs_out['id'].equals(_ghs_sample['id']) or not _ghs_np.isfinite(_ghs_values).all():
    raise RuntimeError('GeoMind/hierarchical shape output audit failed')
def _ghs_sha(path):
    digest = _ghs_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()
_ghs_audit = {{
    'strategy': 'geomind_bias_minus_0p40_hierarchical_shape_{strategy}',
    'shape_sign': _ghs_sign, 'shape_weight': _ghs_weight,
    'gated_wells': int(_ghs_report['gate'].sum()),
    'rows': int(len(_ghs_out)), 'id_order_matches_sample': True,
    'submission_sha256': _ghs_sha(_ghs_path),
    'runtime_sec': float(_ghs_time.time() - _HIERARCHICAL_OFFICIAL_START),
}}
with open(_ghs_work / 'geomind_hierarchical_shape_audit.json', 'w', encoding='utf-8') as handle:
    _ghs_json.dump(_ghs_audit, handle, indent=2, sort_keys=True)
print('GeoMind/hierarchical shape audit:', _ghs_audit, flush=True)
'''
    cells.append(_code_cell(blend))
    hierarchy["cells"] = cells

    kernel_id = f"rogii-geomind-hier-shape-{strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f"{kernel_id}.ipynb"
    notebook_path.write_text(json.dumps(hierarchy, ensure_ascii=False), encoding="utf-8")
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(_metadata(kernel_id, notebook_path.name), indent=2), encoding="utf-8"
    )
    print(f"wrote {notebook_path}")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hierarchical",
        type=Path,
        default=Path("kaggle/rogii-hierarchical-official/rogii-hierarchical-official.ipynb"),
    )
    parser.add_argument(
        "--geomind",
        type=Path,
        default=Path(os.environ.get("TEMP", ".")) / "rogii_survey" / "geomind" / "geomind-trajectory-intelligence-core.ipynb",
    )
    args = parser.parse_args()
    build_one(args.hierarchical, args.geomind, Path("kaggle/rogii-geomind-hier-shape-plus"), 1.0)
    build_one(args.hierarchical, args.geomind, Path("kaggle/rogii-geomind-hier-shape-minus"), -1.0)


if __name__ == "__main__":
    main()
