"""Build C1-anchor notebooks for the orientation-field experiment family."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from build_stopdose_breakthrough_notebooks import code_cell, source, validate_notebook


BASE_DIR = Path("kaggle/rogii-hmm010-c1cal035")
BASE_NOTEBOOK = BASE_DIR / "rogii-hmm010-c1cal035.ipynb"
CORE_PATH = Path("scripts/diagnostics/orientation_field_hmm.py")


@dataclass(frozen=True)
class NotebookVariant:
    slug: str
    strategy: str
    mode: str
    tool_response: str
    output: str
    field_weight: float
    posterior_weight: float
    push_eligible: bool


VARIANTS = (
    NotebookVariant(
        slug="rogii-c1cal035-orientation-antifield015",
        strategy="c1cal035_minus_orientation_field_path_015",
        mode="transition",
        tool_response="off",
        output="field_bridge",
        field_weight=-0.15,
        posterior_weight=0.0,
        push_eligible=True,
    ),
    NotebookVariant(
        slug="rogii-c1cal035-orientation-field015",
        strategy="c1cal035_plus_orientation_field_path_015",
        mode="transition",
        tool_response="off",
        output="field_bridge",
        field_weight=0.15,
        posterior_weight=0.0,
        push_eligible=True,
    ),
    NotebookVariant(
        slug="rogii-c1cal035-orientation-hmm010",
        strategy="c1cal035_plus_orientation_transition_hmm010",
        mode="transition",
        tool_response="off",
        output="anchor_hybrid",
        field_weight=0.0,
        posterior_weight=0.10,
        push_eligible=False,
    ),
    NotebookVariant(
        slug="rogii-c1cal035-orientation-fault010",
        strategy="c1cal035_plus_fault_aware_orientation_hmm010",
        mode="fault_aware",
        tool_response="off",
        output="anchor_hybrid",
        field_weight=0.0,
        posterior_weight=0.10,
        push_eligible=False,
    ),
    NotebookVariant(
        slug="rogii-c1cal035-orientation-tool010",
        strategy="c1cal035_plus_orientation_tool_response_hmm010",
        mode="transition",
        tool_response="prefix_mixture",
        output="anchor_hybrid",
        field_weight=0.0,
        posterior_weight=0.10,
        push_eligible=False,
    ),
    NotebookVariant(
        slug="rogii-c1cal035-orientation-combined010",
        strategy="c1cal035_plus_fault_orientation_tool_response_hmm010",
        mode="fault_aware",
        tool_response="prefix_mixture",
        output="anchor_hybrid",
        field_weight=0.0,
        posterior_weight=0.10,
        push_eligible=False,
    ),
)


def _joined(notebook: dict) -> str:
    return "\n".join(source(cell) for cell in notebook["cells"])


def _load_base() -> dict:
    notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    joined = _joined(notebook)
    required = (
        "_C1_ALPHA = 0.35",
        "mha400_continuity_plus_student_t_hmm010",
        "c1_heel_audit.json",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"C1 calibrated anchor fragments missing: {missing}")
    return notebook


def _core_cell() -> dict:
    core = CORE_PATH.read_text(encoding="utf-8")
    if "00e12e8b" in core:
        raise RuntimeError("orientation core contains a fixed public well ID")
    return code_cell(core)


def _execution_cell(variant: NotebookVariant) -> dict:
    code = f'''# Orientation-field layer over the scored C1 alpha-0.35 anchor.
import hashlib as _of_hashlib
import json as _of_json
import os as _of_os
import time as _of_time
from pathlib import Path as _OfPath

import numpy as _of_np
import pandas as _of_pd

_OF_STARTED = _of_time.time()
_OF_STRATEGY = {variant.strategy!r}
_OF_DEFAULT_MODE = {variant.mode!r}
_OF_DEFAULT_TOOL_RESPONSE = {variant.tool_response!r}
_OF_DEFAULT_OUTPUT = {variant.output!r}
_OF_DEFAULT_FIELD_WEIGHT = {variant.field_weight!r}
_OF_DEFAULT_POSTERIOR_WEIGHT = {variant.posterior_weight!r}
_OF_MODE = _of_os.environ.get('ROGII_ORIENTATION_MODE', _OF_DEFAULT_MODE).strip().lower()
_OF_TOOL_RESPONSE = _of_os.environ.get(
    'ROGII_TOOL_RESPONSE', _OF_DEFAULT_TOOL_RESPONSE
).strip().lower()
_OF_OUTPUT = _of_os.environ.get('ROGII_ORIENTATION_OUTPUT', _OF_DEFAULT_OUTPUT).strip().lower()
_OF_K = int(_of_os.environ.get('ROGII_ORIENTATION_K', '48'))
_OF_SEGMENT_ROWS = int(_of_os.environ.get('ROGII_ORIENTATION_SEGMENT_ROWS', '128'))
_OF_DRIFT_SCALE = float(_of_os.environ.get('ROGII_ORIENTATION_DRIFT_SCALE', '1.0'))
_OF_FIELD_WEIGHT = float(
    _of_os.environ.get('ROGII_ORIENTATION_FIELD_WEIGHT', str(_OF_DEFAULT_FIELD_WEIGHT))
)
_OF_POSTERIOR_WEIGHT = float(
    _of_os.environ.get('ROGII_ORIENTATION_POSTERIOR_WEIGHT', str(_OF_DEFAULT_POSTERIOR_WEIGHT))
)
if _OF_MODE not in {{'off', 'transition', 'fault_aware'}}:
    raise ValueError(f'invalid ROGII_ORIENTATION_MODE={{_OF_MODE!r}}')
if _OF_TOOL_RESPONSE not in {{'off', 'prefix_mixture'}}:
    raise ValueError(f'invalid ROGII_TOOL_RESPONSE={{_OF_TOOL_RESPONSE!r}}')
if _OF_OUTPUT not in {{'field_bridge', 'posterior', 'anchor_hybrid'}}:
    raise ValueError(f'invalid ROGII_ORIENTATION_OUTPUT={{_OF_OUTPUT!r}}')
if not -1.0 <= _OF_FIELD_WEIGHT <= 1.0 or not 0.0 <= _OF_POSTERIOR_WEIGHT <= 1.0:
    raise ValueError('orientation field weight must lie in [-1, 1] and posterior weight in [0, 1]')

_of_work = _OfPath('/kaggle/working') if _OfPath('/kaggle/working').exists() else _OfPath('.')
_of_data = _OfPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_of_data / 'sample_submission.csv').exists():
    _of_data = _OfPath('/kaggle/input/rogii-wellbore-geology-prediction')
_of_submission_path = _of_work / 'submission.csv'
_of_anchor = _of_pd.read_csv(_of_submission_path, dtype={{'id': 'string'}})[['id', 'tvt']]
_of_anchor['id'] = _of_anchor['id'].astype(str)
_of_anchor['tvt'] = _of_pd.to_numeric(_of_anchor['tvt'], errors='coerce')
if not _of_np.isfinite(_of_anchor['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('orientation layer received non-finite anchor')
_of_anchor.to_csv(_of_work / 'submission_before_orientation_field.csv', index=False)
_of_parts = _of_anchor['id'].str.rsplit('_', n=1, expand=True)
if _of_parts.shape[1] != 2:
    raise RuntimeError('orientation layer could not parse submission IDs')
_of_anchor['_well'] = _of_parts[0].astype(str)
_of_anchor['_row'] = _of_pd.to_numeric(_of_parts[1], errors='raise').astype(int)

_of_config = OrientationFieldConfig(
    segment_rows=_OF_SEGMENT_ROWS,
    neighbor_count=_OF_K,
    drift_scale=_OF_DRIFT_SCALE,
)
_of_model = build_orientation_field(_of_data / 'train', _of_config)
_of_values = _of_anchor['tvt'].to_numpy(dtype=float).copy()
_of_well_audits = []

for _of_well, _of_group in _of_anchor.groupby('_well', sort=True):
    _of_positions = _of_group.sort_values('_row').index.to_numpy(dtype=int)
    _of_rows = _of_anchor.loc[_of_positions, '_row'].to_numpy(dtype=int)
    _of_hw = _of_pd.read_csv(_of_data / 'test' / f'{{_of_well}}__horizontal_well.csv')
    if (_of_rows < 0).any() or (_of_rows >= len(_of_hw)).any():
        raise RuntimeError(f'orientation row index outside target well {{_of_well}}')
    _of_tvt_input = _of_pd.to_numeric(_of_hw['TVT_input'], errors='coerce').to_numpy(dtype=float)
    _of_eval = _of_np.flatnonzero(~_of_np.isfinite(_of_tvt_input))
    if not _of_np.array_equal(_of_rows, _of_eval):
        raise RuntimeError(f'orientation native suffix contract failed for {{_of_well}}')
    _of_full_anchor = _of_tvt_input.copy()
    _of_full_anchor[_of_rows] = _of_values[_of_positions]
    if not _of_np.isfinite(_of_full_anchor).all():
        raise RuntimeError(f'orientation anchor does not cover {{_of_well}}')

    _of_path = _of_model.predict_path(
        _of_hw,
        _of_full_anchor,
        target_well=str(_of_well),
        use_faults=_OF_MODE == 'fault_aware',
    )
    _of_base_eval = _of_full_anchor[_of_rows]
    if _OF_OUTPUT == 'field_bridge':
        _of_candidate = _of_path.field_path[_of_rows]
        _of_final = (1.0 - _OF_FIELD_WEIGHT) * _of_base_eval + _OF_FIELD_WEIGHT * _of_candidate
        _of_hmm_metadata = None
    else:
        _of_tw = _of_pd.read_csv(
            _of_data / 'test' / f'{{_of_well}}__typewell.csv', usecols=['TVT', 'GR']
        )
        _of_result = run_orientation_hmm(
            _of_hw,
            _of_tw,
            _of_full_anchor,
            _of_model,
            target_well=str(_of_well),
            use_faults=_OF_MODE == 'fault_aware',
            use_tool_response=_OF_TOOL_RESPONSE == 'prefix_mixture',
        )
        _of_candidate = _of_result.posterior[_of_rows]
        _of_final = (
            _of_candidate
            if _OF_OUTPUT == 'posterior'
            else (1.0 - _OF_POSTERIOR_WEIGHT) * _of_base_eval
            + _OF_POSTERIOR_WEIGHT * _of_candidate
        )
        _of_hmm_metadata = _of_result.metadata
    if not _of_np.isfinite(_of_final).all():
        raise RuntimeError(f'orientation produced non-finite values for {{_of_well}}')
    _of_values[_of_positions] = _of_final
    _of_move = _of_final - _of_base_eval
    _of_well_audits.append({{
        'well': str(_of_well),
        'rows': int(len(_of_rows)),
        'path': _of_path.metadata,
        'hmm': _of_hmm_metadata,
        'mean_move': float(_of_np.mean(_of_move)),
        'mean_abs_move': float(_of_np.mean(_of_np.abs(_of_move))),
        'p95_abs_move': float(_of_np.quantile(_of_np.abs(_of_move), 0.95)),
        'max_abs_move': float(_of_np.max(_of_np.abs(_of_move))),
    }})

_of_final_frame = _of_anchor[['id']].copy()
_of_final_frame['tvt'] = _of_values
_of_sample = _of_pd.read_csv(_of_data / 'sample_submission.csv', dtype={{'id': 'string'}})[['id']]
_of_sample['id'] = _of_sample['id'].astype(str)
if len(_of_final_frame) != len(_of_sample) or not _of_final_frame['id'].equals(_of_sample['id']):
    raise RuntimeError('orientation submission/sample alignment mismatch')
_of_final_frame.to_csv(_of_submission_path, index=False)

def _of_sha(path):
    return _of_hashlib.sha256(_OfPath(path).read_bytes()).hexdigest()

_of_component_digest = _of_hashlib.sha256()
for _of_array in (
    _of_model.observations,
    _of_model.fault_observations,
):
    _of_component_digest.update(_of_np.ascontiguousarray(_of_array).tobytes())
_ORIENTATION_AUDIT = {{
    'strategy': _OF_STRATEGY,
    'mode': _OF_MODE,
    'tool_response': _OF_TOOL_RESPONSE,
    'output': _OF_OUTPUT,
    'field_weight': float(_OF_FIELD_WEIGHT),
    'posterior_weight': float(_OF_POSTERIOR_WEIGHT),
    'config': asdict(_of_config),
    'rows': int(len(_of_final_frame)),
    'wells': int(len(_of_well_audits)),
    'train_orientation_observations': int(len(_of_model.observations)),
    'train_fault_observations': int(len(_of_model.fault_observations)),
    'orientation_component_sha256': _of_component_digest.hexdigest(),
    'anchor_sha256': _of_sha(_of_work / 'submission_before_orientation_field.csv'),
    'final_sha256': _of_sha(_of_submission_path),
    'runtime_sec': float(_of_time.time() - _OF_STARTED),
    'same_id_train_contacts_excluded': True,
    'target_tail_tvt_used': False,
    'fixed_public_ids_used': False,
    'well_audits': _of_well_audits,
}}
(_of_work / 'orientation_field_audit.json').write_text(
    _of_json.dumps(_ORIENTATION_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('orientation field audit:', _of_json.dumps(_ORIENTATION_AUDIT, indent=2, sort_keys=True), flush=True)
'''
    return code_cell(code)


def _final_audit_cell(variant: NotebookVariant) -> dict:
    code = f'''# Final contract after the orientation layer.
import hashlib as _ofa_hashlib
import json as _ofa_json
from pathlib import Path as _OfaPath
import numpy as _ofa_np
import pandas as _ofa_pd

_ofa_work = _OfaPath('/kaggle/working') if _OfaPath('/kaggle/working').exists() else _OfaPath('.')
_ofa_data = _OfaPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
if not (_ofa_data / 'sample_submission.csv').exists():
    _ofa_data = _OfaPath('/kaggle/input/rogii-wellbore-geology-prediction')
_ofa_path = _ofa_work / 'submission.csv'
_ofa_sub = _ofa_pd.read_csv(_ofa_path, dtype={{'id': 'string'}})[['id', 'tvt']]
_ofa_sample = _ofa_pd.read_csv(_ofa_data / 'sample_submission.csv', dtype={{'id': 'string'}})[['id']]
_ofa_sub['id'] = _ofa_sub['id'].astype(str)
_ofa_sample['id'] = _ofa_sample['id'].astype(str)
_ofa_values = _ofa_pd.to_numeric(_ofa_sub['tvt'], errors='coerce').to_numpy(dtype=float)
if len(_ofa_sub) != len(_ofa_sample) or not _ofa_sub['id'].equals(_ofa_sample['id']):
    raise RuntimeError('final orientation ID order mismatch')
if not _ofa_np.isfinite(_ofa_values).all():
    raise RuntimeError('final orientation values are non-finite')
_OFA_AUDIT = {{
    'strategy': {variant.strategy!r},
    'rows': int(len(_ofa_sub)),
    'id_order_matches_sample': True,
    'submission_sha256': _ofa_hashlib.sha256(_ofa_path.read_bytes()).hexdigest(),
    'tvt_min': float(_ofa_values.min()),
    'tvt_max': float(_ofa_values.max()),
    'tvt_mean': float(_ofa_values.mean()),
    'tvt_std': float(_ofa_values.std()),
    'orientation': _ORIENTATION_AUDIT,
}}
(_ofa_work / 'orientation_submission_audit.json').write_text(
    _ofa_json.dumps(_OFA_AUDIT, indent=2, sort_keys=True), encoding='utf-8'
)
print('final orientation audit:', _OFA_AUDIT, flush=True)
'''
    return code_cell(code)


def build_variant(variant: NotebookVariant) -> Path:
    notebook = _load_base()
    notebook["cells"].extend([_core_cell(), _execution_cell(variant), _final_audit_cell(variant)])
    validate_notebook(notebook, variant.slug)
    joined = _joined(notebook)
    if "00e12e8b" in joined:
        raise RuntimeError(f"{variant.slug} contains a fixed public well ID")
    output = Path("kaggle") / variant.slug
    output.mkdir(parents=True, exist_ok=True)
    notebook_path = output / f"{variant.slug}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    metadata = json.loads((BASE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata.update(
        {
            "id": f"qwer556617123/{variant.slug}",
            "title": variant.slug,
            "code_file": notebook_path.name,
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
        }
    )
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return notebook_path


def main() -> None:
    for variant in VARIANTS:
        path = build_variant(variant)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "strategy": variant.strategy,
                    "push_eligible": variant.push_eligible,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
