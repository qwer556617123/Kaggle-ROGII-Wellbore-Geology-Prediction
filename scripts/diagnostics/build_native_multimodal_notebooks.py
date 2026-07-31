"""Build slim Kaggle notebooks for the native-mask multimodal strategy."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PLOT_ONLY_CELLS = {21, 24, 25, 28, 34}
KOOLBOX_BOOTSTRAP = """import glob, subprocess, sys
_kb_wheels = sorted(set(
    glob.glob('/kaggle/input/**/koolbox-*.whl', recursive=True)
))
if not _kb_wheels:
    raise FileNotFoundError('koolbox wheel not found in Kaggle inputs')
print('installing koolbox wheel:', _kb_wheels[0], flush=True)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', _kb_wheels[0]],
    check=True,
)
import koolbox
print('koolbox OK:', koolbox.__file__, flush=True)
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def clear_cell(cell: dict) -> dict:
    out = copy.deepcopy(cell)
    if out.get("cell_type") == "code":
        out["execution_count"] = None
        out["outputs"] = []
    return out


def build_anchor(source_path: Path, output_path: Path) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    kept = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if not source.strip() or index in PLOT_ONLY_CELLS:
            continue
        cleaned = clear_cell(cell)
        if index == 1:
            cleaned["source"] = KOOLBOX_BOOTSTRAP.splitlines(keepends=True)
        kept.append(cleaned)

    bootstrap = code_cell(
        "import os, random, time\n"
        "os.environ['PYTHONHASHSEED'] = '42'\n"
        "os.environ['ROGII_GOLD_PROFILE'] = 'conservative'\n"
        "random.seed(42)\n"
        "_AMGED_ANCHOR_T0 = time.time()\n"
    )
    final_audit = code_cell(
        "# Rerun-safe final anchor audit.\n"
        "import hashlib as _aa_hashlib\n"
        "import json as _aa_json\n"
        "import time as _aa_time\n"
        "from pathlib import Path as _AAPath\n"
        "import numpy as _aa_np\n"
        "import pandas as _aa_pd\n\n"
        "_aa_work = _AAPath('/kaggle/working') if _AAPath('/kaggle/working').exists() else _AAPath('.')\n"
        "_aa_sub_path = _aa_work / 'submission.csv'\n"
        "_aa_sub = _aa_pd.read_csv(_aa_sub_path)\n"
        "_aa_data = _AAPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')\n"
        "if not (_aa_data / 'sample_submission.csv').exists():\n"
        "    _aa_data = _AAPath('/kaggle/input/rogii-wellbore-geology-prediction')\n"
        "_aa_sample = _aa_pd.read_csv(_aa_data / 'sample_submission.csv')\n"
        "assert list(_aa_sub.columns) == ['id', 'tvt']\n"
        "assert len(_aa_sub) == len(_aa_sample)\n"
        "assert _aa_sub['id'].astype(str).equals(_aa_sample['id'].astype(str))\n"
        "assert _aa_np.isfinite(_aa_sub['tvt'].to_numpy(float)).all()\n"
        "_aa_sha = _aa_hashlib.sha256(_aa_sub_path.read_bytes()).hexdigest()\n"
        "_aa_audit = {\n"
        "    'strategy': 'amged_7091_public_source_slim_anchor',\n"
        "    'rows': int(len(_aa_sub)),\n"
        "    'runtime_sec': float(_aa_time.time() - _AMGED_ANCHOR_T0),\n"
        "    'sha256': _aa_sha,\n"
        "    'tvt_min': float(_aa_sub['tvt'].min()),\n"
        "    'tvt_max': float(_aa_sub['tvt'].max()),\n"
        "    'tvt_mean': float(_aa_sub['tvt'].mean()),\n"
        "    'tvt_std': float(_aa_sub['tvt'].std()),\n"
        "    'gold_profile': os.environ.get('ROGII_GOLD_PROFILE'),\n"
        "}\n"
        "(_aa_work / 'amged7091_anchor_audit.json').write_text(\n"
        "    _aa_json.dumps(_aa_audit, indent=2, sort_keys=True), encoding='utf-8'\n"
        ")\n"
        "print('Amged 7.091 slim anchor audit:', _aa_audit, flush=True)\n"
    )
    notebook["cells"] = [bootstrap, *kept, final_audit]
    for index, cell in enumerate(notebook["cells"]):
        cell.get("metadata", {}).pop("execution", None)
        cell.get("metadata", {}).pop("papermill", None)
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"anchor_cell_{index}", "exec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")


def write_metadata(source_metadata: Path, output_path: Path) -> None:
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata.update(
        {
            "id": "qwer556617123/rogii-amged7091-rebuild",
            "title": "rogii-amged7091-rebuild",
            "code_file": "rogii-amged7091-rebuild.ipynb",
            "is_private": True,
        }
    )
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def posterior_runner_cell() -> str:
    return r'''# Heel-calibrated multimodal posterior layer over the rerun-safe anchor.
import hashlib as _pt_hashlib
import json as _pt_json
import os as _pt_os
import time as _pt_time
from pathlib import Path as _PTPath
import numpy as _pt_np
import pandas as _pt_pd

_pt_started = _pt_time.time()
_pt_work = _PTPath('/kaggle/working') if _PTPath('/kaggle/working').exists() else _PTPath('.')
_pt_data = _PTPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_pt_data / 'sample_submission.csv').exists():
    _pt_data = _PTPath('/kaggle/input/rogii-wellbore-geology-prediction')
_pt_mode = _pt_os.environ.get('ROGII_POSTERIOR_OUTPUT_MODE', 'anchor').strip().lower()
if _pt_mode not in {'anchor', 'posterior', 'hybrid'}:
    raise ValueError('ROGII_POSTERIOR_OUTPUT_MODE must be anchor, posterior, or hybrid')
_pt_config = PosteriorConfig(
    max_shift=float(_pt_os.environ.get('ROGII_POSTERIOR_MAX_SHIFT', '60')),
    grid_step=float(_pt_os.environ.get('ROGII_POSTERIOR_GRID_STEP', '0.5')),
    temperature=float(_pt_os.environ.get('ROGII_POSTERIOR_TEMPERATURE', '0.08')),
)
_pt_hybrid_cap = float(_pt_os.environ.get('ROGII_POSTERIOR_HYBRID_CAP', '0.25'))
_pt_anchor_path = _pt_work / 'submission.csv'
_pt_anchor = _pt_pd.read_csv(_pt_anchor_path)[['id', 'tvt']].copy()
_pt_sample = _pt_pd.read_csv(_pt_data / 'sample_submission.csv')[['id']].copy()
_pt_anchor['id'] = _pt_anchor['id'].astype(str)
_pt_sample['id'] = _pt_sample['id'].astype(str)
assert _pt_anchor['id'].equals(_pt_sample['id'])
_pt_anchor.to_csv(_pt_work / 'submission_anchor.csv', index=False)
_pt_anchor_map = dict(zip(_pt_anchor['id'], _pt_anchor['tvt'].astype(float)))
_pt_posterior_map = {}
_pt_hybrid_map = {}
_pt_well_audits = []

for _pt_well in _pt_anchor['id'].str.rsplit('_', n=1).str[0].drop_duplicates():
    _pt_hw = _pt_pd.read_csv(_pt_data / 'test' / f'{_pt_well}__horizontal_well.csv')
    _pt_tw = _pt_pd.read_csv(_pt_data / 'test' / f'{_pt_well}__typewell.csv')
    _pt_full_anchor = _pt_hw['TVT_input'].to_numpy(dtype=float).copy()
    _pt_group = _pt_anchor[_pt_anchor['id'].str.startswith(_pt_well + '_')]
    _pt_rows = _pt_group['id'].str.rsplit('_', n=1).str[1].astype(int).to_numpy()
    for _pt_id, _pt_row in zip(_pt_group['id'], _pt_rows):
        _pt_full_anchor[int(_pt_row)] = _pt_anchor_map[_pt_id]
    _pt_result = predict_multimodal(
        _pt_hw,
        _pt_tw,
        anchor=_pt_full_anchor,
        config=_pt_config,
        alpha=None,
        search_from_anchor=True,
    )
    _pt_alpha = min(float(_pt_result.metadata.get('alpha', 0.0)), _pt_hybrid_cap)
    _pt_hybrid = _pt_full_anchor + _pt_alpha * (_pt_result.posterior - _pt_full_anchor)
    for _pt_id, _pt_row in zip(_pt_group['id'], _pt_rows):
        _pt_posterior_map[_pt_id] = float(_pt_result.posterior[int(_pt_row)])
        _pt_hybrid_map[_pt_id] = float(_pt_hybrid[int(_pt_row)])
    _pt_meta = dict(_pt_result.metadata)
    _pt_meta.update({'well': _pt_well, 'hybrid_alpha_capped': _pt_alpha})
    _pt_well_audits.append(_pt_meta)

_pt_posterior = _pt_sample.copy()
_pt_posterior['tvt'] = _pt_posterior['id'].map(_pt_posterior_map).astype(float)
_pt_hybrid = _pt_sample.copy()
_pt_hybrid['tvt'] = _pt_hybrid['id'].map(_pt_hybrid_map).astype(float)
_pt_posterior.to_csv(_pt_work / 'submission_posterior.csv', index=False)
_pt_hybrid.to_csv(_pt_work / 'submission_hybrid.csv', index=False)
_pt_selected = {'anchor': _pt_anchor, 'posterior': _pt_posterior, 'hybrid': _pt_hybrid}[_pt_mode]
assert len(_pt_selected) == len(_pt_sample)
assert _pt_selected['id'].equals(_pt_sample['id'])
assert _pt_np.isfinite(_pt_selected['tvt'].to_numpy(float)).all()
_pt_selected.to_csv(_pt_work / 'submission.csv', index=False)

def _pt_sha(path):
    return _pt_hashlib.sha256(_PTPath(path).read_bytes()).hexdigest()

_pt_audit = {
    'strategy': 'native_mask_multimodal_geosteering',
    'output_mode': _pt_mode,
    'rows': int(len(_pt_selected)),
    'config': _pt_config.__dict__,
    'hybrid_cap': _pt_hybrid_cap,
    'runtime_sec_posterior_layer': float(_pt_time.time() - _pt_started),
    'anchor_sha256': _pt_sha(_pt_work / 'submission_anchor.csv'),
    'posterior_sha256': _pt_sha(_pt_work / 'submission_posterior.csv'),
    'hybrid_sha256': _pt_sha(_pt_work / 'submission_hybrid.csv'),
    'final_sha256': _pt_sha(_pt_work / 'submission.csv'),
    'wells': _pt_well_audits,
}
(_pt_work / 'posterior_tracker_audit.json').write_text(
    _pt_json.dumps(_pt_audit, indent=2, sort_keys=True), encoding='utf-8'
)
print('Posterior tracker audit:', _pt_audit, flush=True)
'''


def build_posterior(anchor_path: Path, module_path: Path, output_path: Path) -> None:
    notebook = json.loads(anchor_path.read_text(encoding="utf-8"))
    module_source = module_path.read_text(encoding="utf-8")
    notebook["cells"].extend([code_cell(module_source), code_cell(posterior_runner_cell())])
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"posterior_cell_{index}", "exec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")


def write_posterior_metadata(anchor_metadata: Path, output_path: Path) -> None:
    metadata = json.loads(anchor_metadata.read_text(encoding="utf-8"))
    metadata.update(
        {
            "id": "qwer556617123/rogii-heel-posterior-tracker",
            "title": "rogii-heel-posterior-tracker",
            "code_file": "rogii-heel-posterior-tracker.ipynb",
            "is_private": True,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("kaggle/rogii-amged7091-rebuild"))
    parser.add_argument("--posterior-output-dir", type=Path, default=Path("kaggle/rogii-heel-posterior-tracker"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_anchor(
        args.source_dir / "7-091-public.ipynb",
        args.output_dir / "rogii-amged7091-rebuild.ipynb",
    )
    write_metadata(
        args.source_dir / "kernel-metadata.json",
        args.output_dir / "kernel-metadata.json",
    )
    build_posterior(
        args.output_dir / "rogii-amged7091-rebuild.ipynb",
        Path("scripts/diagnostics/multimodal_geosteering.py"),
        args.posterior_output_dir / "rogii-heel-posterior-tracker.ipynb",
    )
    write_posterior_metadata(
        args.output_dir / "kernel-metadata.json",
        args.posterior_output_dir / "kernel-metadata.json",
    )
    print(f"Built {args.output_dir}")
    print(f"Built {args.posterior_output_dir}")


if __name__ == "__main__":
    main()
