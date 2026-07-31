"""Build slim audited rebuilds of official-data-only public candidates."""
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


def _load_code_cells(path: Path) -> tuple[dict, list[dict]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for original in notebook["cells"]:
        if original.get("cell_type") != "code" or not _source(original).strip():
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        cells.append(cell)
    return notebook, cells


def _metadata(kernel_id: str, title: str, code_file: str) -> dict:
    return {
        "id": f"qwer556617123/{kernel_id}",
        "title": title,
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


def build_candidate(source: Path, output_dir: Path, kernel_id: str, strategy: str) -> Path:
    notebook, source_cells = _load_code_cells(source)
    timer_name = f"_{strategy.upper()}_START"
    cells = [_code_cell(f"import time as _official_time\n{timer_name} = _official_time.time()\n")]
    cells.extend(source_cells)
    audit = f'''# Wrapper audit; predictions are not modified.
import hashlib as _official_hashlib
import json as _official_json
from pathlib import Path as _OfficialPath
import numpy as _official_np
import pandas as _official_pd
_official_work = _OfficialPath('/kaggle/working') if _OfficialPath('/kaggle/working').exists() else _OfficialPath('.')
_official_data = _OfficialPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
_official_path = _official_work / 'submission.csv'
_official_sub = _official_pd.read_csv(_official_path)[['id', 'tvt']]
_official_sample = _official_pd.read_csv(_official_data / 'sample_submission.csv')[['id']]
_official_sub['id'] = _official_sub['id'].astype(str); _official_sample['id'] = _official_sample['id'].astype(str)
_official_values = _official_sub['tvt'].to_numpy(dtype=float)
if len(_official_sub) != len(_official_sample) or not _official_sub['id'].equals(_official_sample['id']):
    raise RuntimeError('official candidate output is not sample aligned')
if not _official_np.isfinite(_official_values).all():
    raise RuntimeError('official candidate output contains non-finite values')
def _official_sha(path):
    digest = _official_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()
_official_audit = {{
    'strategy': '{strategy}',
    'official_data_only': True,
    'rows': int(len(_official_sub)),
    'id_order_matches_sample': True,
    'submission_sha256': _official_sha(_official_path),
    'runtime_sec': float(_official_time.time() - {timer_name}),
}}
with open(_official_work / '{strategy}_wrapper_audit.json', 'w', encoding='utf-8') as handle:
    _official_json.dump(_official_audit, handle, indent=2, sort_keys=True)
print('official candidate wrapper audit:', _official_audit, flush=True)
'''
    cells.append(_code_cell(audit))
    notebook["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f"{kernel_id}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(_metadata(kernel_id, kernel_id, notebook_path.name), indent=2), encoding="utf-8"
    )
    print(f"wrote {notebook_path} with {len(cells)} code cells")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey-root", type=Path, default=Path(os.environ.get("TEMP", ".")) / "rogii_survey")
    args = parser.parse_args()
    build_candidate(
        next((args.survey_root / "hierarchical").glob("*.ipynb")),
        Path("kaggle/rogii-hierarchical-official"),
        "rogii-hierarchical-official",
        "hierarchical_official",
    )
    build_candidate(
        next((args.survey_root / "geomind").glob("*.ipynb")),
        Path("kaggle/rogii-geomind-official"),
        "rogii-geomind-official",
        "geomind_official",
    )


if __name__ == "__main__":
    main()
