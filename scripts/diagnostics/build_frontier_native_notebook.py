"""Build a native-prefix, audited Frontier Full exploration notebook."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


DROP_CELLS = {24, 25, 28, 34}


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
    leakage_block = """            if wid in tw_ids:
                htr = _upd.read_csv(trd / f'{wid}__horizontal_well.csv'); hw = hw.copy()
                hw['TVT_input'] = htr['TVT_input'].values
                tp = trd / f'{wid}__typewell.csv'; td = _utemp.mkdtemp()
                hp = _uo.path.join(td, f'{wid}__horizontal_well.csv'); hw.to_csv(hp, index=False)
"""
    for index, original in enumerate(notebook["cells"]):
        if original.get("cell_type") != "code" or index in DROP_CELLS:
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        source = _source(cell)
        if index == 0:
            source = (
                "import os, random, time\n"
                "_FRONTIER_NATIVE_START = time.time()\n"
                "os.environ['PYTHONHASHSEED'] = '42'\n"
                "os.environ['ROGII_GOLD_PROFILE'] = 'conservative'\n"
                "os.environ['ROGII_GOLD_PREFIX_CAL'] = '0'\n"
                "random.seed(42)\n"
            )
        if index == 36:
            if leakage_block not in source:
                raise RuntimeError("expected same-ID TVT_input replacement block was not found")
            source = source.replace(leakage_block, "")
        cell["source"] = source.splitlines(keepends=True)
        cells.append(cell)

    audit = r'''# Native-prefix Frontier audit; predictions are not modified.
import hashlib as _fn_hashlib
import json as _fn_json
from pathlib import Path as _FnPath
_fn_work = _FnPath('/kaggle/working') if _FnPath('/kaggle/working').exists() else _FnPath('.')
_fn_data = _FnPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
_fn_sub_path = _fn_work / 'submission.csv'
_fn_sub = _final_pd.read_csv(_fn_sub_path)[['id', 'tvt']]
_fn_sample = _final_pd.read_csv(_fn_data / 'sample_submission.csv')[['id']]
_fn_sub['id'] = _fn_sub['id'].astype(str)
_fn_sample['id'] = _fn_sample['id'].astype(str)
_fn_values = _fn_sub['tvt'].to_numpy(dtype=float)
if len(_fn_sub) != len(_fn_sample) or not _fn_sub['id'].equals(_fn_sample['id']):
    raise RuntimeError('Frontier output is not sample aligned')
if not _final_np.isfinite(_fn_values).all():
    raise RuntimeError('Frontier output contains non-finite values')

def _fn_sha(path):
    digest = _fn_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

_fn_pkw = _fn_work / 'pilkwang_submission.csv'
_fn_unet = _fn_work / 'unet_datum.csv'
_fn_audit = {
    'strategy': 'frontier_native_prefix_pilkwang_unet',
    'same_id_train_tvt_input_used': False,
    'gold_overlay_enabled': False,
    'pilkwang_branch_available': bool(_fn_pkw.exists()),
    'unet_branch_available': bool(_fn_unet.exists()),
    'pilkwang_rows': int(len(_final_pd.read_csv(_fn_pkw))) if _fn_pkw.exists() else 0,
    'unet_wells': int(len(_final_pd.read_csv(_fn_unet))) if _fn_unet.exists() else 0,
    'rows': int(len(_fn_sub)),
    'submission_sha256': _fn_sha(_fn_sub_path),
    'runtime_sec': float(time.time() - _FRONTIER_NATIVE_START),
}
with open(_fn_work / 'frontier_native_audit.json', 'w', encoding='utf-8') as handle:
    _fn_json.dump(_fn_audit, handle, indent=2, sort_keys=True)
print('frontier native final audit:', _fn_audit, flush=True)
'''
    cells.append(_code_cell(audit))
    notebook["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "rogii-frontier-native.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "qwer556617123/rogii-frontier-native",
        "title": "rogii-frontier-native",
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
            "pilkwang/rogii-model-package",
            "sangrampatil5150/unet-rogii",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": ["packagemanager/pm-121894185-at-06-13-2026-22-17-53"],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {notebook_path} with {len(cells)} code cells")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("TEMP", "."))
        / "rogii_survey"
        / "frontierfull"
        / "rogii-frontier-full.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/rogii-frontier-native"),
    )
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
