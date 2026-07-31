"""Build a safe, slim rebuild of the public deterministic LB 7.166 notebook."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


DROP_CELLS = {24, 25, 28, 34}
LAST_REQUIRED_CELL = 39


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


def build(source_path: Path, output_dir: Path) -> Path:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    cells = []
    for index, original in enumerate(notebook["cells"]):
        if index > LAST_REQUIRED_CELL or index in DROP_CELLS:
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        if index == 0:
            cell = _code_cell(
                "import os, random, time\n"
                "_DETERMINISTIC7166_START = time.time()\n"
                "os.environ['PYTHONHASHSEED'] = '42'\n"
                "os.environ['ROGII_GOLD_PROFILE'] = 'conservative'\n"
                "os.environ['ROGII_GOLD_PREFIX_CAL'] = '1'\n"
                "os.environ['ROGII_PROBE'] = '0'\n"
                "random.seed(42)\n"
            )
        cells.append(cell)

    audit = r'''# Final submission audit. This cell never changes predictions.
import hashlib as _d7166_hashlib
import json as _d7166_json
import time as _d7166_time
from pathlib import Path as _D7166Path
import numpy as _d7166_np
import pandas as _d7166_pd

_d7166_work = _D7166Path('/kaggle/working') if _D7166Path('/kaggle/working').exists() else _D7166Path('.')
_d7166_data = _D7166Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
_d7166_sub_path = _d7166_work / 'submission.csv'
_d7166_sub = _d7166_pd.read_csv(_d7166_sub_path)[['id', 'tvt']]
_d7166_sample = _d7166_pd.read_csv(_d7166_data / 'sample_submission.csv')[['id']]
_d7166_sub['id'] = _d7166_sub['id'].astype(str)
_d7166_sample['id'] = _d7166_sample['id'].astype(str)
_d7166_values = _d7166_sub['tvt'].to_numpy(dtype=float)
if len(_d7166_sub) != len(_d7166_sample):
    raise RuntimeError('submission row count mismatch')
if not _d7166_sub['id'].equals(_d7166_sample['id']):
    raise RuntimeError('submission id order mismatch')
if not _d7166_np.isfinite(_d7166_values).all():
    raise RuntimeError('submission contains non-finite tvt')

def _d7166_sha256(path):
    digest = _d7166_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

_d7166_audit = {
    'strategy': 'deterministic7166_safe_rebuild',
    'probe_enabled': False,
    'gold_profile': 'conservative',
    'global_bias_shift_ft': -0.40,
    'rows': int(len(_d7166_sub)),
    'id_order_matches_sample': True,
    'tvt_min': float(_d7166_values.min()),
    'tvt_max': float(_d7166_values.max()),
    'tvt_mean': float(_d7166_values.mean()),
    'tvt_std': float(_d7166_values.std()),
    'submission_sha256': _d7166_sha256(_d7166_sub_path),
    'runtime_sec': float(_d7166_time.time() - _DETERMINISTIC7166_START),
}
with open(_d7166_work / 'deterministic7166_audit.json', 'w', encoding='utf-8') as handle:
    _d7166_json.dump(_d7166_audit, handle, indent=2, sort_keys=True)
print('deterministic7166 final audit:', _d7166_audit, flush=True)
'''
    cells.append(_code_cell(audit))
    notebook["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "rogii-deterministic7166-rebuild.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "id": "qwer556617123/rogii-deterministic7166-rebuild",
        "title": "rogii-deterministic7166-rebuild",
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
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {notebook_path} with {len(cells)} cells")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("TEMP", "."))
        / "rogii_survey"
        / "det7166"
        / "rogii-determenistic-solution-stable-lb-7-166.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/rogii-deterministic7166-rebuild"),
    )
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
