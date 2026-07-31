"""Build an audited hidden-rerun rebuild of the public dual-pipeline LB 7.159 notebook."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


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
    cells = [
        _code_cell(
            "import glob as _kb_glob, subprocess as _kb_subprocess, sys as _kb_sys\n"
            "_kb_wheels = sorted(_kb_glob.glob('/kaggle/input/**/koolbox*.whl', recursive=True))\n"
            "if not _kb_wheels:\n"
            "    raise FileNotFoundError('koolbox wheel not found in mounted datasets')\n"
            "_kb_subprocess.run([_kb_sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', _kb_wheels[0]], check=True)\n"
            "import koolbox\n"
            "print('koolbox artifact compatibility:', koolbox.__file__, flush=True)\n"
        )
    ]
    for original in notebook["cells"]:
        if original.get("cell_type") != "code":
            continue
        cell = copy.deepcopy(original)
        cell["execution_count"] = None
        cell["outputs"] = []
        source = _cell_source(cell)
        if "GLOBAL SWITCHES" in source:
            source = source.replace(
                "import os\n", "import os, random\n_DEG7159_START = __import__('time').time()\n"
            )
            source = source.replace(
                "ENABLE_GOLD_OVERLAY = False",
                "ENABLE_GOLD_OVERLAY = False\nos.environ['PYTHONHASHSEED'] = '42'\nrandom.seed(42)",
            )
            cell["source"] = source.splitlines(keepends=True)
        if "X_test_A = test_df_A[features_A]" in source:
            source = source.replace(
                "X_test_A = test_df_A[features_A]",
                "_missing_features_A = [c for c in features_A if c not in test_df_A.columns]\n"
                "for _missing_col_A in _missing_features_A:\n"
                "    test_df_A[_missing_col_A] = 0.0\n"
                "print('Pipeline A zero-filled missing test features:', _missing_features_A)\n"
                "X_test_A = test_df_A[features_A]",
            )
            cell["source"] = source.splitlines(keepends=True)
        cells.append(cell)

    audit = r'''# Rebuild lineage and runtime audit; predictions are not modified.
_deg7159_path = CFG.OUT / 'submission.csv'
_deg7159_sub = pd.read_csv(_deg7159_path)[['id', 'tvt']]
_deg7159_sample = pd.read_csv(CFG.DATA / 'sample_submission.csv')[['id']]
_deg7159_sub['id'] = _deg7159_sub['id'].astype(str)
_deg7159_sample['id'] = _deg7159_sample['id'].astype(str)
_deg7159_values = _deg7159_sub['tvt'].to_numpy(dtype=float)
if len(_deg7159_sub) != len(_deg7159_sample):
    raise RuntimeError('dual-pipeline row count mismatch')
if not _deg7159_sub['id'].equals(_deg7159_sample['id']):
    raise RuntimeError('dual-pipeline id order mismatch')
if not np.isfinite(_deg7159_values).all():
    raise RuntimeError('dual-pipeline non-finite output')
_deg7159_audit = {
    'strategy': 'degnonguidi7159_dual_pipeline_rebuild',
    'gold_overlay_enabled': bool(ENABLE_GOLD_OVERLAY),
    'sp45_weight': float(SP45_WEIGHT),
    'pipeline_a_zero_filled_features': list(_missing_features_A),
    'rows': int(len(_deg7159_sub)),
    'id_order_matches_sample': True,
    'submission_sha256': sha256_file(_deg7159_path),
    'runtime_sec': float(time.time() - _DEG7159_START),
}
with open(CFG.OUT / 'degnonguidi7159_rebuild_audit.json', 'w', encoding='utf-8') as handle:
    json.dump(_deg7159_audit, handle, indent=2, sort_keys=True)
print('degnonguidi7159 rebuild audit:', _deg7159_audit, flush=True)
'''
    cells.append(_code_cell(audit))
    notebook["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "rogii-degnonguidi7159-rebuild.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "qwer556617123/rogii-degnonguidi7159-rebuild",
        "title": "rogii-degnonguidi7159-rebuild",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "phongnguyn23021656/koolbox-offline",
            "nina2025/rogii-03",
            "thbdh5765/rogii-v10-fresh-artifacts",
            "fleongg/rogii-claude-models-pub",
            "needless090/rogii-tabicl-mirror",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
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
        / "degnonguidi7159"
        / "public-score-rogii-lb-7-159.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/rogii-degnonguidi7159-rebuild"),
    )
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
