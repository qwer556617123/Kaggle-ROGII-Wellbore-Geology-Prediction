"""Build geometry/reference PF candidates on the verified MHA140SEP4 contract."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_stopdose_breakthrough_notebooks import (
    clean_code_cells,
    code_cell,
    hmm_runner,
    integrity_audit,
    source,
    validate_notebook,
    write_metadata,
)


SOURCE = Path(
    "kaggle/external_reviews/frontier_20260720c/refblend_geom25/"
    "rogii-det-pf-refblend-w25-geometry-r1.ipynb"
)
HMM_SOURCE = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")
META_PATH = Path(
    "kaggle/outputs/rogii-geometry-meta-source-exact-v1/"
    "ridge_meta_coefficients.json"
)

GEOMETRY_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry025")
HMM_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry-hmm025")
FAST_GEOMETRY_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry-fast")
FAST_HMM_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry-hmm-fast")
PARALLEL_GEOMETRY_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry-parallel")
PARALLEL_HMM_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry-hmm-parallel")
LEAN_HMM_OUTPUT = Path("kaggle/rogii-pf-refblend-geometry-hmm-lean")
EXACT_META_OUTPUT = Path("kaggle/rogii-geometry-meta-source-exact")

# Source cells 43-46 are read-only diagnostics and 47 is an inactive destructive probe.
SKIPPED = {43, 44, 45, 46, 47}
FAST_SKIPPED = SKIPPED | {14, 21}


def _replace_once(cells: list[dict], old: str, new: str) -> None:
    matches = []
    for index, cell in enumerate(cells):
        text = source(cell)
        if old in text:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"expected one source replacement for {old!r}, found {matches}")
    cell = cells[matches[0]]
    text = source(cell).replace(old, new)
    cell["source"] = text.splitlines(keepends=True)


def _replace_cell(cells: list[dict], prefix: str, new: str) -> None:
    matches = [index for index, cell in enumerate(cells) if source(cell).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell starting with {prefix!r}, found {matches}")
    cells[matches[0]]["source"] = new.splitlines(keepends=True)


def _prepare_anchor() -> tuple[dict, list[dict]]:
    notebook, cells = clean_code_cells(SOURCE, skipped=SKIPPED)
    _replace_once(
        cells,
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "1.4, 0.22, 6.0, 40.0, 4.0",
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "1.4, 0.22, 4.0, 40.0, 4.0",
    )
    joined = "\n".join(source(cell) for cell in cells)
    required = (
        "def beam_search_geometry",
        "def empirical_lwd_reference",
        "_PF_REFBLEND_WEIGHT = 0.25000000",
        "1.4, 0.22, 4.0, 40.0, 4.0",
        "_BC_SHIFT = -0.40",
        'os.environ["ROGII_GOLD_PROFILE"] = "conservative"',
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"geometry/reference source fragments missing: {missing}")
    forbidden = (
        "Legal heel-cal predictor v2",
        "Smoother SALVAGE sweep",
        "Smoother-vs-STACK OOF blend test",
        "LB PROBE v4",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"diagnostic/probe cells leaked into rebuild: {leaked}")
    return notebook, cells


def _prepare_fast_anchor() -> tuple[dict, list[dict], dict]:
    if not META_PATH.exists():
        raise RuntimeError(f"runtime-safe Ridge manifest is missing: {META_PATH}")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    expected_order = [
        "lightgbm-1",
        "lightgbm-2",
        "lightgbm-3",
        "catboost-1",
        "catboost-2",
    ]
    if meta.get("model_order") != expected_order:
        raise RuntimeError(f"unexpected Ridge model order: {meta.get('model_order')}")
    if len(meta.get("coef", [])) != 5:
        raise RuntimeError("runtime-safe Ridge manifest must contain five coefficients")

    notebook, cells = clean_code_cells(SOURCE, skipped=FAST_SKIPPED)
    _replace_once(
        cells,
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "1.4, 0.22, 6.0, 40.0, 4.0",
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "1.4, 0.22, 4.0, 40.0, 4.0",
    )
    _replace_cell(
        cells,
        'if (CFG.artifacts_path / "data" / "train.csv").exists():',
        """# Runtime-safe inference: every fitted model is loaded from the public artifact.
if not (CFG.artifacts_path / "data" / "train.csv").exists():
    raise RuntimeError("required public model artifact dataset is missing")

test_paths = sorted((CFG.dataset_path / "test").glob('*__horizontal_well.csv'))
test_df = build_dataset(test_paths, is_train=False, label="test")
features = None
X_test = None
""",
    )
    _replace_once(
        cells,
        '    oof_preds[f"lightgbm-{i+1}"] = trainer.oof_preds\n'
        '    test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)',
        '    if X_test is None:\n'
        '        features = list(trainer.estimators[0].feature_name_)\n'
        '        missing = sorted(set(features) - set(test_df.columns))\n'
        '        if missing:\n'
        '            raise RuntimeError(f"test features missing: {missing}")\n'
        '        X_test = test_df[features]\n'
        '    test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)',
    )
    _replace_once(
        cells,
        '    oof_preds[f"catboost-{i+1}"] = trainer.oof_preds\n'
        '    test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)',
        '    test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)',
    )
    _replace_cell(
        cells,
        "oof_preds = pd.DataFrame(oof_preds)",
        "test_preds = pd.DataFrame(test_preds)\n",
    )
    coef = [float(value) for value in meta["coef"]]
    intercept = float(meta["intercept"])
    _replace_cell(
        cells,
        "ridge_trainer = Trainer(",
        "# Exact mean of the five fold Ridge predictions extracted from public OOF.\n"
        f"_FAST_META_ORDER = {expected_order!r}\n"
        f"_FAST_META_COEF = np.asarray({coef!r}, dtype=float)\n"
        f"_FAST_META_INTERCEPT = {intercept!r}\n"
        "test_preds = test_preds[_FAST_META_ORDER]\n"
        "ridge_test_preds = (\n"
        "    test_preds.to_numpy(dtype=float) @ _FAST_META_COEF\n"
        "    + _FAST_META_INTERCEPT\n"
        ")\n",
    )
    _replace_cell(
        cells,
        'sample_sub = pd.read_csv(CFG.dataset_path / "sample_submission.csv")',
        """sample_sub = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sub_1 = sample_sub[['id']].merge(
    test_df2[['id', 'pred']].rename(columns={'pred': 'tvt'}),
    on='id',
    how='left',
)
fallback = float(test_df2['pred'].median())
sub_1['tvt'] = sub_1['tvt'].fillna(fallback)
sub_1
""",
    )

    joined = "\n".join(source(cell) for cell in cells)
    required = (
        "def beam_search_geometry",
        "def empirical_lwd_reference",
        "_PF_REFBLEND_WEIGHT = 0.25000000",
        "n_particles=500, n_seeds=128, stats_out=_bstats",
        "_FAST_META_COEF",
        "1.4, 0.22, 4.0, 40.0, 4.0",
        "Gold visible-prefix calibration overlay",
        "Guarded contact override v2",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"runtime-safe source fragments missing: {missing}")
    forbidden = (
        'pd.read_csv(CFG.artifacts_path / "data" / "train.csv"',
        "ridge_trainer.fit",
        "Legal heel-cal predictor v2",
        "Smoother SALVAGE sweep",
        "Smoother-vs-STACK OOF blend test",
        "LB PROBE v4",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"slow/diagnostic source leaked into fast rebuild: {leaked}")
    return notebook, cells, meta


def _parallelize_reference_loop(cells: list[dict]) -> None:
    marker = (
        "rows = []\n"
        "PF_BIMODAL_STATS = {}  # DELTA midhedge\n"
        "for i, wid in enumerate(test_wells):\n"
    )
    matches = [index for index, cell in enumerate(cells) if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one sequential reference loop, found {matches}")
    cell = cells[matches[0]]
    prefix, body = source(cell).split(marker, 1)
    if body.count("PF_BIMODAL_STATS[wid] = _bstats") != 1:
        raise RuntimeError("reference loop bimodal stats assignment changed")
    if body.count("rows.append(") != 1:
        raise RuntimeError("reference loop row append changed")
    body = body.replace("PF_BIMODAL_STATS[wid] = _bstats", "_well_bstats = _bstats")
    body = body.replace("rows.append(", "_well_rows.append(")
    parallel = (
        "def _run_reference_well(_job):\n"
        "    i, wid = _job\n"
        "    _well_rows = []\n"
        "    _well_bstats = {}\n"
        + body
        + "\n    return _well_rows, wid, _well_bstats\n\n"
        "_REFERENCE_N_JOBS = max(1, min(3, len(test_wells)))\n"
        "_reference_results = Parallel(\n"
        "    n_jobs=_REFERENCE_N_JOBS, backend='loky', verbose=5\n"
        ")(delayed(_run_reference_well)(job) for job in enumerate(test_wells))\n"
        "rows = []\n"
        "PF_BIMODAL_STATS = {}  # DELTA midhedge\n"
        "for _well_rows, _wid, _well_bstats in _reference_results:\n"
        "    rows.extend(_well_rows)\n"
        "    PF_BIMODAL_STATS[_wid] = _well_bstats\n"
        "print(f'parallel paired PF complete: jobs={_REFERENCE_N_JOBS} wells={len(test_wells)}')\n"
    )
    cell["source"] = (prefix + parallel).splitlines(keepends=True)


def _parallelize_gold_loop(cells: list[dict]) -> None:
    start = "    _gold_reports = []\n"
    end = "    _gold_report_df = _gold_pd.DataFrame(_gold_reports)\n"
    matches = [
        index
        for index, cell in enumerate(cells)
        if start in source(cell) and end in source(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Gold calibration loop, found {matches}")
    cell = cells[matches[0]]
    text = source(cell)
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    replacement = r'''    _gold_reports = []
    _gold_cut_reports = []
    _gold_candidate_by_id = {}
    _gold_wells = list(_gold_base['well'].drop_duplicates())[:_GOLD_MAX_WELLS]

    def _gold_process_well(_job):
        _wi, _wid = _job
        _local_candidates = {}
        try:
            _hw_path = _GOLD_DATA / 'test' / f'{_wid}__horizontal_well.csv'
            _tw_path = _GOLD_DATA / 'test' / f'{_wid}__typewell.csv'
            if not _hw_path.exists() or not _tw_path.exists():
                return dict(well=_wid, status='skip_missing_files'), [], _local_candidates
            _hw = _gold_pd.read_csv(_hw_path)
            _tw = _gold_pd.read_csv(_tw_path)
            print('[gold %d/%d] calibrating %s' % (_wi, len(_gold_wells), _wid), flush=True)
            _rep = _gold_calibrate_well(_wid, _hw, _tw, _GOLD_DATA, _gold_variants)
            if _rep is None:
                _rep = dict(well=_wid, status='skip_none')
            _cut_rows = _rep.pop('cut_rows', []) if isinstance(_rep, dict) else []
            if _rep.get('status') == 'ok':
                _best_name = _rep['best_name']
                _need_pf_final = str(_best_name).startswith('pf|')
                _pool_final = _gold_candidate_pool(
                    _wid, _hw, _tw, _GOLD_DATA, _gold_variants,
                    include_pf=_need_pf_final,
                    n_seeds=_GOLD_FINAL_SEEDS,
                    n_particles=_GOLD_PARTICLES,
                )
                if _best_name not in _pool_final and _need_pf_final:
                    _pool_final = _gold_candidate_pool(
                        _wid, _hw, _tw, _GOLD_DATA, _gold_variants,
                        include_pf=False,
                        n_seeds=0,
                        n_particles=_GOLD_PARTICLES,
                    )
                if _best_name in _pool_final:
                    _g = _gold_base[_gold_base['well'] == _wid]
                    _arr = _pool_final[_best_name]
                    for _rid, _ri in zip(_g['id'].astype(str).values, _g['row_idx'].astype(int).values):
                        if 0 <= int(_ri) < len(_arr) and _gold_np.isfinite(_arr[int(_ri)]):
                            _local_candidates[_rid] = float(_arr[int(_ri)])
                    _rep['final_candidate_available'] = True
                else:
                    _rep['final_candidate_available'] = False
                    _rep['status'] = 'skip_no_final_candidate'
            print('  report:', {k: _rep.get(k) for k in ['status', 'best_name', 'best_score', 'default_score', 'gain', 'consistency']}, flush=True)
            return _rep, _cut_rows, _local_candidates
        except Exception as _e:
            print('gold calibration fallback', _wid, _e)
            return dict(well=_wid, status='error', error=str(_e)), [], _local_candidates

    _GOLD_N_JOBS = max(1, min(3, len(_gold_wells)))
    _gold_results = Parallel(
        n_jobs=_GOLD_N_JOBS, backend='loky', verbose=5
    )(delayed(_gold_process_well)(job) for job in enumerate(_gold_wells, 1))
    for _rep, _cut_rows, _local_candidates in _gold_results:
        _gold_reports.append(_rep)
        _gold_cut_reports.extend(_cut_rows)
        _gold_candidate_by_id.update(_local_candidates)
    print('parallel Gold calibration complete: jobs=%d wells=%d' % (_GOLD_N_JOBS, len(_gold_wells)), flush=True)

    _gold_report_df = _gold_pd.DataFrame(_gold_reports)
'''
    cell["source"] = (before + replacement + after).splitlines(keepends=True)


def _prepare_parallel_anchor() -> tuple[dict, list[dict], dict]:
    notebook, cells, meta = _prepare_fast_anchor()
    _parallelize_reference_loop(cells)
    _parallelize_gold_loop(cells)
    joined = "\n".join(source(cell) for cell in cells)
    required = (
        "def _run_reference_well",
        "parallel paired PF complete",
        "def _gold_process_well",
        "parallel Gold calibration complete",
        "backend='loky'",
        "n_particles=500, n_seeds=128, stats_out=_bstats",
        "_GOLD_CAL_SEEDS",
        "_GOLD_FINAL_SEEDS",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"parallel runtime source fragments missing: {missing}")
    forbidden = (
        "for i, wid in enumerate(test_wells):",
        "for _wi, _wid in enumerate(_gold_wells, 1):",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"sequential per-well loop leaked into parallel rebuild: {leaked}")
    return notebook, cells, meta


def _prepare_lean_anchor() -> tuple[dict, list[dict], dict]:
    notebook, cells, meta = _prepare_parallel_anchor()
    _replace_once(
        cells,
        "n_particles=500, n_seeds=128, stats_out=_bstats",
        "n_particles=500, n_seeds=64, stats_out=_bstats",
    )
    _replace_once(
        cells,
        "_GOLD_CAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_CAL_SEEDS', '24'))",
        "_GOLD_CAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_CAL_SEEDS', '12'))",
    )
    _replace_once(
        cells,
        "_GOLD_FINAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_FINAL_SEEDS', '48'))",
        "_GOLD_FINAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_FINAL_SEEDS', '24'))",
    )
    joined = "\n".join(source(cell) for cell in cells)
    required = (
        "n_particles=500, n_seeds=64, stats_out=_bstats",
        "ROGII_GOLD_CAL_SEEDS', '12'",
        "ROGII_GOLD_FINAL_SEEDS', '24'",
        "backend='loky'",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"lean runtime source fragments missing: {missing}")
    return notebook, cells, meta


def _start_cell() -> dict:
    return code_cell("import time as _sd_time\n_SD_STARTED = _sd_time.time()\n")


def _write(
    notebook: dict,
    cells: list[dict],
    output: Path,
    slug: str,
    strategy: str,
    audit_extra: str,
) -> Path:
    notebook["cells"] = [
        _start_cell(),
        *cells,
        code_cell(integrity_audit(strategy, "_SD_STARTED", audit_extra)),
    ]
    validate_notebook(notebook, slug)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(SOURCE, output, slug)
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_sources"] = [
        item for item in metadata.get("dataset_sources", []) if item
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def build_geometry() -> Path:
    notebook, cells = _prepare_anchor()
    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'\n"
        "_sd_audit['typewell_reference'] = 'raw_plus_prefix_affine_empirical_lwd'"
    )
    return _write(
        notebook,
        cells,
        GEOMETRY_OUTPUT,
        "rogii-pf-refblend-geometry025",
        "mha140sep4_geometry_beam_and_paired_reference_pf025",
        extra,
    )


def build_hmm() -> Path:
    notebook, cells = _prepare_anchor()
    hmm_notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    hmm_core = copy.deepcopy(hmm_notebook["cells"][58])
    hmm_core["execution_count"] = None
    hmm_core["outputs"] = []
    hmm_core.get("metadata", {}).pop("execution", None)
    hmm_core.get("metadata", {}).pop("papermill", None)
    core_source = source(hmm_core)
    if "class HMMParams" not in core_source or "def run_hmm2" not in core_source:
        raise RuntimeError("exact HMM core was not found in EXP087 source cell 58")

    runner = hmm_runner()
    replacements = {
        "_HMM_WEIGHT = 0.15": "_HMM_WEIGHT = 0.25",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "mha140sep4_geometry_refblend_plus_student_t_hmm"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t second-order HMM posterior-mean blend over the "
            "geometry/reference PF candidate."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    cells.extend([hmm_core, code_cell(runner)])

    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'\n"
        "_sd_audit['hmm'] = _HMM_AUDIT"
    )
    return _write(
        notebook,
        cells,
        HMM_OUTPUT,
        "rogii-pf-refblend-geometry-hmm025",
        "mha140sep4_geometry_reference_pf025_plus_student_t_hmm025",
        extra,
    )


def build_fast_geometry() -> Path:
    notebook, cells, meta = _prepare_fast_anchor()
    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['runtime_safe_meta'] = True\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'"
    )
    return _write(
        notebook,
        cells,
        FAST_GEOMETRY_OUTPUT,
        "rogii-pf-refblend-geometry-fast",
        "runtime_safe_geometry_reference_pf025_embedded_ridge",
        extra,
    )


def build_fast_hmm() -> Path:
    notebook, cells, meta = _prepare_fast_anchor()
    hmm_notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    hmm_core = copy.deepcopy(hmm_notebook["cells"][58])
    hmm_core["execution_count"] = None
    hmm_core["outputs"] = []
    hmm_core.get("metadata", {}).pop("execution", None)
    hmm_core.get("metadata", {}).pop("papermill", None)
    core_source = source(hmm_core)
    if "class HMMParams" not in core_source or "def run_hmm2" not in core_source:
        raise RuntimeError("exact HMM core was not found in EXP087 source cell 58")

    runner = hmm_runner()
    replacements = {
        "_HMM_WEIGHT = 0.15": "_HMM_WEIGHT = 0.25",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "runtime_safe_geometry_refblend_plus_student_t_hmm"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t second-order HMM posterior mean over the runtime-safe "
            "geometry/reference PF candidate."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one fast HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    cells.extend([hmm_core, code_cell(runner)])

    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['runtime_safe_meta'] = True\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'\n"
        "_sd_audit['hmm'] = _HMM_AUDIT"
    )
    return _write(
        notebook,
        cells,
        FAST_HMM_OUTPUT,
        "rogii-pf-refblend-geometry-hmm-fast",
        "runtime_safe_geometry_reference_pf025_plus_student_t_hmm025",
        extra,
    )


def build_parallel_geometry() -> Path:
    notebook, cells, meta = _prepare_parallel_anchor()
    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['runtime_safe_meta'] = True\n"
        "_sd_audit['parallel_reference_jobs'] = int(_REFERENCE_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'"
    )
    return _write(
        notebook,
        cells,
        PARALLEL_GEOMETRY_OUTPUT,
        "rogii-pf-refblend-geometry-parallel",
        "parallel_exact_geometry_reference_pf025",
        extra,
    )


def build_parallel_hmm() -> Path:
    notebook, cells, meta = _prepare_parallel_anchor()
    hmm_notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    hmm_core = copy.deepcopy(hmm_notebook["cells"][58])
    hmm_core["execution_count"] = None
    hmm_core["outputs"] = []
    hmm_core.get("metadata", {}).pop("execution", None)
    hmm_core.get("metadata", {}).pop("papermill", None)
    core_source = source(hmm_core)
    if "class HMMParams" not in core_source or "def run_hmm2" not in core_source:
        raise RuntimeError("exact HMM core was not found in EXP087 source cell 58")

    runner = hmm_runner()
    replacements = {
        "_HMM_WEIGHT = 0.15": "_HMM_WEIGHT = 0.25",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "parallel_exact_geometry_refblend_plus_student_t_hmm"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t second-order HMM posterior mean over the parallel exact "
            "geometry/reference PF candidate."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one parallel HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    cells.extend([hmm_core, code_cell(runner)])

    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['runtime_safe_meta'] = True\n"
        "_sd_audit['parallel_reference_jobs'] = int(_REFERENCE_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'\n"
        "_sd_audit['hmm'] = _HMM_AUDIT"
    )
    return _write(
        notebook,
        cells,
        PARALLEL_HMM_OUTPUT,
        "rogii-pf-refblend-geometry-hmm-parallel",
        "parallel_exact_geometry_reference_pf025_plus_student_t_hmm025",
        extra,
    )


def build_lean_hmm() -> Path:
    notebook, cells, meta = _prepare_lean_anchor()
    hmm_notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    hmm_core = copy.deepcopy(hmm_notebook["cells"][58])
    hmm_core["execution_count"] = None
    hmm_core["outputs"] = []
    hmm_core.get("metadata", {}).pop("execution", None)
    hmm_core.get("metadata", {}).pop("papermill", None)
    core_source = source(hmm_core)
    if "class HMMParams" not in core_source or "def run_hmm2" not in core_source:
        raise RuntimeError("exact HMM core was not found in EXP087 source cell 58")

    runner = hmm_runner()
    replacements = {
        "_HMM_WEIGHT = 0.15": "_HMM_WEIGHT = 0.25",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "parallel_lean_geometry_refblend_plus_student_t_hmm"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t second-order HMM posterior mean over the parallel lean "
            "geometry/reference PF candidate."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one lean HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    cells.extend([hmm_core, code_cell(runner)])

    extra = (
        "_sd_audit['anchor'] = 'verified_mha140sep4_contract'\n"
        "_sd_audit['runtime_safe_meta'] = True\n"
        "_sd_audit['parallel_reference_jobs'] = int(_REFERENCE_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)\n"
        "_sd_audit['main_pf_seeds'] = 64\n"
        "_sd_audit['gold_cal_seeds'] = int(_GOLD_CAL_SEEDS)\n"
        "_sd_audit['gold_final_seeds'] = int(_GOLD_FINAL_SEEDS)\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}\n"
        "_sd_audit['pf_reference_blend_weight'] = float(_PF_REFBLEND_WEIGHT)\n"
        "_sd_audit['beam_transition'] = 'geometry_u_equals_tvt_plus_z'\n"
        "_sd_audit['hmm'] = _HMM_AUDIT"
    )
    return _write(
        notebook,
        cells,
        LEAN_HMM_OUTPUT,
        "rogii-pf-refblend-geometry-hmm-lean",
        "parallel_lean_geometry_reference_pf025_plus_student_t_hmm025",
        extra,
    )


def build_exact_meta() -> Path:
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook, cells = clean_code_cells(
        SOURCE, skipped=set(range(13, len(raw["cells"])))
    )
    extraction = r'''# Save the exact fitted Ridge state before any PF/routing work.
import hashlib as _mx_hashlib
import json as _mx_json
import time as _mx_time
from pathlib import Path as _MxPath
import numpy as _mx_np

_mx_models = list(ridge_trainer.estimators)
if len(_mx_models) != 5:
    raise RuntimeError(f'expected five Ridge folds, found {len(_mx_models)}')
_mx_folds = [
    {
        'fold': int(index),
        'coef': model.coef_.astype(float).tolist(),
        'intercept': float(model.intercept_),
        'rmse': float(ridge_trainer.fold_scores[index]),
    }
    for index, model in enumerate(_mx_models)
]
_mx_manifest = {
    'model_order': list(test_preds.columns),
    'coef': _mx_np.mean(_mx_np.asarray([item['coef'] for item in _mx_folds]), axis=0).tolist(),
    'intercept': float(_mx_np.mean([item['intercept'] for item in _mx_folds])),
    'folds': _mx_folds,
    'overall_oof_rmse': float(ridge_trainer.overall_score),
    'rows': int(len(ridge_trainer.oof_preds)),
    'runtime_sec': float(_mx_time.time() - _SD_STARTED),
}
_mx_path = _MxPath('/kaggle/working/ridge_meta_coefficients.json')
_mx_path.write_text(_mx_json.dumps(_mx_manifest, indent=2, sort_keys=True), encoding='utf-8')
_mx_manifest['sha256'] = _mx_hashlib.sha256(_mx_path.read_bytes()).hexdigest()
print('exact source meta:', _mx_manifest, flush=True)
'''
    notebook["cells"] = [_start_cell(), *cells, code_cell(extraction)]
    validate_notebook(notebook, "geometry_meta_source_exact")
    EXACT_META_OUTPUT.mkdir(parents=True, exist_ok=True)
    slug = "rogii-geometry-meta-source-exact"
    path = EXACT_META_OUTPUT / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(SOURCE, EXACT_META_OUTPUT, slug)
    metadata_path = EXACT_META_OUTPUT / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_sources"] = [
        item for item in metadata.get("dataset_sources", []) if item
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()
    builders = {
        "geometry": build_geometry,
        "hmm": build_hmm,
        "fast_geometry": build_fast_geometry,
        "fast_hmm": build_fast_hmm,
        "parallel_geometry": build_parallel_geometry,
        "parallel_hmm": build_parallel_hmm,
        "lean_hmm": build_lean_hmm,
        "meta_exact": build_exact_meta,
    }
    targets = args.targets or list(builders)
    unknown = sorted(set(targets) - set(builders))
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}")
    for target in targets:
        path = builders[target]()
        print(f"built {path}")


if __name__ == "__main__":
    main()
