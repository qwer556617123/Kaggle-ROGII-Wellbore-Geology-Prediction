"""Build audited, output-free rebuilds of the public 7.016 and 7.061 notebooks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCES = {
    "targetfree7016": {
        "source": Path(
            "kaggle/external_reviews/target-free-geosteering-7016/"
            "target-free-tvt-geosteering-7-016.ipynb"
        ),
        "output": Path("kaggle/rogii-targetfree7016-rebuild"),
        "kernel_id": "rogii-targetfree7016-rebuild",
        "strategy": "target_free_geosteering_7016_exact_rebuild",
        "expected_profile": "vp_balanced_final",
    },
    "pkadopt7061": {
        "source": Path(
            "kaggle/external_reviews/rogii-pk-adopt-lb7061/"
            "rogii-pk-adopt-pb-lb-7-061.ipynb"
        ),
        "output": Path("kaggle/rogii-pkadopt7061-rebuild"),
        "kernel_id": "rogii-pkadopt7061-rebuild",
        "strategy": "pk_adopt_7061_modelpkg_010_exact_rebuild",
        "expected_profile": "vp_balanced_modelpkg_010",
    },
    "mpkg015": {
        "source": Path(
            "kaggle/external_reviews/rogii-mpkg015/"
            "rogii-exp085-yusuke-a023-mpkg015.ipynb"
        ),
        "output": Path("kaggle/rogii-mpkg015-rebuild"),
        "kernel_id": "rogii-mpkg015-rebuild",
        "strategy": "yusuke_a023_modelpkg_015_exact_rebuild",
        "expected_profile": "vp_balanced_modelpkg_0150",
    },
    "mpkg020": {
        "source": Path(
            "kaggle/external_reviews/rogii-mpkg020/"
            "rogii-exp086-yusuke-a023-mpkg020.ipynb"
        ),
        "output": Path("kaggle/rogii-mpkg020-rebuild"),
        "kernel_id": "rogii-mpkg020-rebuild",
        "strategy": "yusuke_a023_modelpkg_020_exact_rebuild",
        "expected_profile": "vp_balanced_modelpkg_0200",
    },
    "mpkg0125": {
        "source": Path(
            "kaggle/external_reviews/rogii-kim-mpkg0125/"
            "rogii-kim-mpkg0125.ipynb"
        ),
        "output": Path("kaggle/rogii-kim-mpkg0125-rebuild"),
        "kernel_id": "rogii-kim-mpkg0125-rebuild",
        "strategy": "kim_modelpkg_0125_guarded_exact_rebuild",
        "expected_profile": "vp_balanced_modelpkg_010",
        "effective_profile": "vp_balanced_modelpkg_0125",
        "required_fragments": [
            "'model_package_gated_max_weight': 0.0125",
            "MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0",
        ],
    },
    "dualtrack": {
        "source": Path(
            "kaggle/external_reviews/rogii-dual-track/"
            "rogii-dual-track-prefix-calibrated-geosteering.ipynb"
        ),
        "output": Path("kaggle/rogii-pilkwang-dualtrack-rebuild"),
        "kernel_id": "rogii-pilkwang-dualtrack-rebuild",
        "strategy": "dual_track_prefix_modelpkg_010_exact_rebuild",
        "expected_profile": "dual_track_prefix_modelpkg_010",
        "required_fragments": [
            "SP45_BLEND_WEIGHT = float(_profile['sp45_blend_weight'])",
            "MODEL_PACKAGE_DIFF_P95_DISABLE = None",
        ],
    },
    "dualtrack_latest": {
        "source": Path(
            "kaggle/external_reviews/frontier_20260720/dual_track/"
            "rogii-dual-track-prefix-calibrated-geosteering.ipynb"
        ),
        "output": Path("kaggle/rogii-pilkwang-dualtrack-latest-rebuild"),
        "kernel_id": "rogii-pilkwang-dualtrack-latest-rebuild",
        "strategy": "dual_track_prefix_modelpkg_latest_exact_rebuild",
        "expected_profile": "dual_track_prefix_modelpkg",
        "required_fragments": [
            "GLOBAL_TVT_BIAS = -0.40",
            "SP45_BLEND_WEIGHT = 0.55",
            "RUN_VISIBLE_PREFIX_CALIBRATION = True",
            "RUN_GUARDED_OVERLAP_OVERRIDE = True",
            "MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.0100",
        ],
        "source_replacements": {
            "device = torch.device('cuda' if torch.cuda.is_available() and str(entry.get('device', 'auto')).lower() != 'cpu' else 'cpu')":
            "device = torch.device('cpu')  # Kaggle CUDA image lacks a compatible TCN kernel.",
        },
    },
}


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


def build_one(config: dict) -> Path:
    source_path = config["source"]
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    code_cells = []
    for original in notebook["cells"]:
        if original.get("cell_type") != "code" or not _source(original).strip():
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        code_cells.append(cell)

    for old, new in config.get("source_replacements", {}).items():
        matches = sum(_source(cell).count(old) for cell in code_cells)
        if matches != 1:
            raise RuntimeError(
                f"expected one source replacement match, found {matches}: {old}"
            )
        for cell in code_cells:
            source = _source(cell)
            if old in source:
                cell["source"] = source.replace(old, new).splitlines(keepends=True)

    joined = "\n".join(_source(cell) for cell in code_cells)
    profile_line = f"SUBMISSION_PROFILE = '{config['expected_profile']}'"
    if profile_line not in joined:
        raise RuntimeError(f"expected source profile not found: {profile_line}")
    for fragment in config.get("required_fragments", []):
        if fragment not in joined:
            raise RuntimeError(f"required source fragment not found: {fragment}")

    effective_profile = config.get("effective_profile", config["expected_profile"])

    start = _code_cell(
        "import time as _public_rebuild_time\n"
        "_PUBLIC_REBUILD_START = _public_rebuild_time.time()\n"
    )
    audit = _code_cell(
        f'''# Rebuild audit; predictions are not modified.
import hashlib as _pra_hashlib
import json as _pra_json
from pathlib import Path as _PraPath
import numpy as _pra_np
import pandas as _pra_pd

_pra_work = _PraPath('/kaggle/working') if _PraPath('/kaggle/working').exists() else _PraPath('.')
_pra_data = _PraPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
_pra_path = _pra_work / 'submission.csv'
_pra_sub = _pra_pd.read_csv(_pra_path)[['id', 'tvt']]
_pra_sample = _pra_pd.read_csv(_pra_data / 'sample_submission.csv')[['id']]
_pra_sub['id'] = _pra_sub['id'].astype(str)
_pra_sample['id'] = _pra_sample['id'].astype(str)
_pra_values = _pra_sub['tvt'].to_numpy(dtype=float)
if len(_pra_sub) != len(_pra_sample):
    raise RuntimeError('submission row count mismatch')
if not _pra_sub['id'].equals(_pra_sample['id']):
    raise RuntimeError('submission id order mismatch')
if not _pra_np.isfinite(_pra_values).all():
    raise RuntimeError('submission contains non-finite tvt')

def _pra_sha256(path):
    digest = _pra_hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

_pra_audit = {{
    'strategy': {config['strategy']!r},
    'source_notebook': {source_path.as_posix()!r},
    'submission_profile': {effective_profile!r},
    'compatibility_patches': {list(config.get('source_replacements', {}).values())!r},
    'rows': int(len(_pra_sub)),
    'id_order_matches_sample': True,
    'tvt_min': float(_pra_values.min()),
    'tvt_max': float(_pra_values.max()),
    'tvt_mean': float(_pra_values.mean()),
    'tvt_std': float(_pra_values.std()),
    'submission_sha256': _pra_sha256(_pra_path),
    'runtime_sec': float(_public_rebuild_time.time() - _PUBLIC_REBUILD_START),
}}
with open(_pra_work / 'public_anchor_rebuild_audit.json', 'w', encoding='utf-8') as handle:
    _pra_json.dump(_pra_audit, handle, indent=2, sort_keys=True)
print('public anchor rebuild audit:', _pra_audit, flush=True)
'''
    )
    notebook["cells"] = [start, *code_cells, audit]

    output_dir = config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f"{config['kernel_id']}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")

    source_metadata = json.loads(
        (source_path.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    metadata = {
        "id": f"qwer556617123/{config['kernel_id']}",
        "title": config["kernel_id"],
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": source_metadata.get("dataset_sources", []),
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {notebook_path} with {len(notebook['cells'])} code cells")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="TARGET",
    )
    args = parser.parse_args()
    targets = args.targets or list(SOURCES)
    unknown = sorted(set(targets) - set(SOURCES))
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}")
    for target in targets:
        build_one(SOURCES[target])


if __name__ == "__main__":
    main()
