"""Build a slim rerun-safe PRVS grouped-OOF meta-residual notebook."""
from __future__ import annotations

import copy
import json
from pathlib import Path


SOURCE_DIR = Path("kaggle/external_reviews/municef-prvs-meta-offset")
SOURCE_NOTEBOOK = SOURCE_DIR / "rogii-prvs-meta-offset.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
TARGET_SLUG = "rogii-prvs-oof-meta-rebuild"
TARGET_DIR = Path("kaggle") / TARGET_SLUG


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


AUDIT_CELL = r'''# Final grouped-OOF meta-residual integrity audit.
import hashlib as _om_hashlib
import json as _om_json
import time as _om_time
from pathlib import Path as _OmPath
import numpy as _om_np
import pandas as _om_pd

_om_work = _OmPath('/kaggle/working') if _OmPath('/kaggle/working').exists() else _OmPath('.')
_om_data = _OmPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_om_data / 'sample_submission.csv').exists():
    _om_data = _OmPath('/kaggle/input/rogii-wellbore-geology-prediction')
_om_path = _om_work / 'submission.csv'
_om_submission = _om_pd.read_csv(_om_path)
_om_sample = _om_pd.read_csv(_om_data / 'sample_submission.csv')
_om_submission['id'] = _om_submission['id'].astype(str)
_om_sample['id'] = _om_sample['id'].astype(str)
if list(_om_submission.columns) != ['id', 'tvt']:
    raise RuntimeError(f'unexpected OOF-meta columns: {list(_om_submission.columns)}')
if len(_om_submission) != len(_om_sample):
    raise RuntimeError('OOF-meta row count mismatch')
if not _om_submission['id'].equals(_om_sample['id']):
    raise RuntimeError('OOF-meta id order mismatch')
_om_values = _om_submission['tvt'].to_numpy(dtype=float)
if not _om_np.isfinite(_om_values).all():
    raise RuntimeError('OOF-meta output contains non-finite values')
_om_summary_path = _om_work / 'oof_meta_residual_audit.json'
if not _om_summary_path.exists():
    raise RuntimeError('OOF meta residual branch did not produce its audit')
_om_summary = _om_json.loads(_om_summary_path.read_text(encoding='utf-8'))
_om_features = list(_om_summary.get('features', []))
if 'model_package_postprocessed' not in _om_features or 'public_pf_delta' not in _om_features:
    raise RuntimeError(f'OOF meta branch missing required features: {_om_features}')

_OM_AUDIT = {
    'strategy': 'prvs_grouped_oof_meta_residual_slim',
    'source': 'municef1/rogii-prvs-meta-offset',
    'source_profile': str(globals().get('SUBMISSION_PROFILE', 'unknown')),
    'rows': int(len(_om_submission)),
    'id_order_matches_sample': True,
    'finite': True,
    'runtime_sec': float(_om_time.time() - _OOF_META_STARTED_AT),
    'oof_rows': int(_om_summary.get('rows', 0)),
    'oof_wells': int(_om_summary.get('wells', 0)),
    'oof_features': _om_features,
    'oof_ramped_rmse': float(_om_summary.get('meta_group_oof_ramped_rmse', float('nan'))),
    'submission_sha256': _om_hashlib.sha256(_om_path.read_bytes()).hexdigest(),
}
(_om_work / 'prvs_oof_meta_final_audit.json').write_text(
    _om_json.dumps(_OM_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('PRVS OOF-meta final audit:', _OM_AUDIT, flush=True)
'''


def validate_notebook(notebook: dict) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        compile(source, f"{TARGET_SLUG}:cell-{index}", "exec")


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    # Cell 68 is the final audit. Cells 69-84 are plots; 85 needs an unpublished lookup.
    notebook["cells"] = [copy.deepcopy(cell) for cell in notebook["cells"][:69]]
    notebook["cells"].insert(0, code_cell("import time as _oof_meta_time\n_OOF_META_STARTED_AT = _oof_meta_time.time()\n"))
    quote_fix_old = "print(f'  PF {int(globals().get('SELECTOR_PF_SEEDS', SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK scales={SELECTOR_SCALES}')"
    quote_fix_new = 'print(f"  PF {int(globals().get(\'SELECTOR_PF_SEEDS\', SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK scales={SELECTOR_SCALES}")'
    quote_fixes = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if quote_fix_old in source:
                source = source.replace(quote_fix_old, quote_fix_new)
                quote_fixes += 1
            cell["source"] = source
            cell["execution_count"] = None
            cell["outputs"] = []
    if quote_fixes != 1:
        raise RuntimeError(f"expected one source quote fix, found {quote_fixes}")
    notebook["cells"].append(code_cell(AUDIT_CELL))
    notebook.setdefault("metadata", {}).pop("papermill", None)
    validate_notebook(notebook)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = TARGET_DIR / f"{TARGET_SLUG}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata["id"] = f"qwer556617123/{TARGET_SLUG}"
    metadata["title"] = TARGET_SLUG
    metadata["code_file"] = notebook_path.name
    metadata["dataset_sources"] = [
        source for source in metadata.get("dataset_sources", []) if str(source).strip()
    ]
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"built {notebook_path}")


if __name__ == "__main__":
    main()
