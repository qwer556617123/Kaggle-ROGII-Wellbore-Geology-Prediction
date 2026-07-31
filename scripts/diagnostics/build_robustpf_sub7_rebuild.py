"""Build a slim, rerun-safe copy of the public robust-PF sub-7 notebook."""
from __future__ import annotations

import copy
import json
from pathlib import Path


SOURCE_DIR = Path("kaggle/external_reviews/shanyiming-robust-pf-sub7")
SOURCE_NOTEBOOK = SOURCE_DIR / "rogii-robust-pf-sub-7-rebuild.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
TARGET_SLUG = "rogii-robustpf-sub7-rebuild"
TARGET_DIR = Path("kaggle") / TARGET_SLUG


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


AUDIT_CELL = r'''# Final hidden-rerun integrity audit.
import hashlib as _rp_hashlib
import json as _rp_json
import time as _rp_time
from pathlib import Path as _RpPath
import numpy as _rp_np
import pandas as _rp_pd

_rp_work = _RpPath('/kaggle/working') if _RpPath('/kaggle/working').exists() else _RpPath('.')
_rp_data = _RpPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_rp_data / 'sample_submission.csv').exists():
    _rp_data = _RpPath('/kaggle/input/rogii-wellbore-geology-prediction')
_rp_submission_path = _rp_work / 'submission.csv'
_rp_submission = _rp_pd.read_csv(_rp_submission_path)
_rp_sample = _rp_pd.read_csv(_rp_data / 'sample_submission.csv')
_rp_submission['id'] = _rp_submission['id'].astype(str)
_rp_sample['id'] = _rp_sample['id'].astype(str)
if list(_rp_submission.columns) != ['id', 'tvt']:
    raise RuntimeError(f'unexpected submission columns: {list(_rp_submission.columns)}')
if len(_rp_submission) != len(_rp_sample):
    raise RuntimeError('robust-PF row count mismatch')
if not _rp_submission['id'].equals(_rp_sample['id']):
    raise RuntimeError('robust-PF id order mismatch')
_rp_values = _rp_submission['tvt'].to_numpy(dtype=float)
if not _rp_np.isfinite(_rp_values).all():
    raise RuntimeError('robust-PF output contains non-finite values')

_RP_AUDIT = {
    'strategy': 'public_robust_pf_raw128_smooth32_vp_balanced_exact',
    'source': 'shanyiming/rogii-robust-pf-sub-7-rebuild',
    'source_profile': str(globals().get('SUBMISSION_PROFILE', 'unknown')),
    'rows': int(len(_rp_submission)),
    'id_order_matches_sample': True,
    'finite': True,
    'mean': float(_rp_np.mean(_rp_values)),
    'std': float(_rp_np.std(_rp_values)),
    'min': float(_rp_np.min(_rp_values)),
    'max': float(_rp_np.max(_rp_values)),
    'hidden_tvt_ever_read': bool(globals().get('_VP_HIDDEN_TVT_EVER_READ', False)),
    'submission_sha256': _rp_hashlib.sha256(_rp_submission_path.read_bytes()).hexdigest(),
    'audit_unix_time': float(_rp_time.time()),
}
(_rp_work / 'robustpf_sub7_audit.json').write_text(
    _rp_json.dumps(_RP_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('Robust-PF final audit:', _RP_AUDIT, flush=True)
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
    # Cell 58 writes the final calibrated submission. Cells 59+ are read-only plots.
    notebook["cells"] = [copy.deepcopy(cell) for cell in notebook["cells"][:59]]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"].append(code_cell(AUDIT_CELL))
    notebook.setdefault("metadata", {}).pop("papermill", None)
    validate_notebook(notebook)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = TARGET_DIR / f"{TARGET_SLUG}.ipynb"
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=True), encoding="utf-8"
    )

    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata["id"] = f"qwer556617123/{TARGET_SLUG}"
    metadata["title"] = TARGET_SLUG
    metadata["code_file"] = notebook_path.name
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"built {notebook_path}")


if __name__ == "__main__":
    main()
