"""Build runtime-safe exact Public 6.768 anchor and Student-t HMM fusion."""
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
    "kaggle/external_reviews/frontier_20260721/codex6768/"
    "rogii-codex-exact-public-6-768-v1.ipynb"
)
HMM_SOURCE = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")
META_PATH = Path(
    "kaggle/outputs/rogii-geometry-meta-source-exact-v1/"
    "ridge_meta_coefficients.json"
)

ANCHOR_OUTPUT = Path("kaggle/rogii-codex6768-exact")
HMM_OUTPUT = Path("kaggle/rogii-codex6768-student-hmm025")

# Remove only train-scoring, read-only diagnostics, and visualizations. All
# prediction writers through the hidden-set contract remain in source order.
SKIPPED = {13, 24, 32, 35, 36, 39, 45, 48, 55} | set(range(61, 75))


def _replace_once(cells: list[dict], old: str, new: str) -> None:
    matches = [index for index, cell in enumerate(cells) if old in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one replacement for {old!r}, found {matches}")
    cell = cells[matches[0]]
    cell["source"] = source(cell).replace(old, new).splitlines(keepends=True)


def _replace_cell(cells: list[dict], prefix: str, replacement: str) -> None:
    matches = [
        index for index, cell in enumerate(cells) if source(cell).startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell starting with {prefix!r}, found {matches}")
    cells[matches[0]]["source"] = replacement.splitlines(keepends=True)


def _load_meta() -> dict:
    if not META_PATH.exists():
        raise RuntimeError(f"exact Ridge manifest is missing: {META_PATH}")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    expected = [
        "lightgbm-1",
        "lightgbm-2",
        "lightgbm-3",
        "catboost-1",
        "catboost-2",
    ]
    if meta.get("model_order") != expected or len(meta.get("coef", [])) != 5:
        raise RuntimeError("unexpected exact Ridge manifest contract")
    return meta


def _embed_exact_ridge(cells: list[dict], meta: dict) -> None:
    _replace_cell(
        cells,
        'if (CFG.artifacts_path / "data" / "train.csv").exists():',
        '''# Runtime-safe inference: fitted estimators and exact Ridge state are public artifacts.
if not (CFG.artifacts_path / "data" / "train.csv").exists():
    raise RuntimeError("required public model artifact dataset is missing")

test_paths = sorted((CFG.dataset_path / "test").glob('*__horizontal_well.csv'))
test_df = build_dataset(test_paths, is_train=False, label="test")
features = None
X_test = None
''',
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
    _replace_cell(cells, "oof_preds = pd.DataFrame(oof_preds)", "test_preds = pd.DataFrame(test_preds)\n")

    order = list(meta["model_order"])
    coef = [float(value) for value in meta["coef"]]
    intercept = float(meta["intercept"])
    _replace_cell(
        cells,
        "ridge_trainer = Trainer(",
        "# Exact mean of the five public fold-Ridge predictions.\n"
        f"_FAST_META_ORDER = {order!r}\n"
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
        '''sample_sub = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sub_1 = sample_sub[['id']].merge(
    test_df2[['id', 'pred']].rename(columns={'pred': 'tvt'}),
    on='id',
    how='left',
)
if sub_1['tvt'].isna().any():
    raise RuntimeError('learned inference did not cover every sample row')
sub_1
''',
    )


def _parallelize_main_pf(cells: list[dict]) -> None:
    marker = (
        "rows = []\n"
        "bimodal_report_rows = []\n"
        "PF_SEED_BRANCH_STATS = {}\n"
        "for i, wid in enumerate(test_wells):\n"
    )
    post_marker = "\nif bimodal_report_rows:\n"
    matches = [index for index, cell in enumerate(cells) if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one main PF loop, found {matches}")
    cell = cells[matches[0]]
    prefix, remainder = source(cell).split(marker, 1)
    if post_marker not in remainder:
        raise RuntimeError("main PF post-loop report marker changed")
    body, post = remainder.split(post_marker, 1)
    required = (
        "PF_SEED_BRANCH_STATS[str(wid)] = _seed_branch",
        "bimodal_report_rows.append(",
        "\n        rows.append(",
    )
    missing = [fragment for fragment in required if body.count(fragment) != 1]
    if missing:
        raise RuntimeError(f"main PF body contract changed: {missing}")
    body = body.replace(
        "PF_SEED_BRANCH_STATS[str(wid)] = _seed_branch",
        "_well_seed_stats[str(wid)] = _seed_branch",
    )
    body = body.replace("bimodal_report_rows.append(", "_well_bimodal_rows.append(")
    body = body.replace("\n        rows.append(", "\n        _well_rows.append(")
    parallel = (
        "def _run_main_pf_well(_job):\n"
        "    i, wid = _job\n"
        "    _well_rows = []\n"
        "    _well_bimodal_rows = []\n"
        "    _well_seed_stats = {}\n"
        + body
        + "\n    return _well_rows, _well_bimodal_rows, _well_seed_stats\n\n"
        "_MAIN_PF_N_JOBS = max(1, min(3, len(test_wells)))\n"
        "_main_pf_results = Parallel(\n"
        "    n_jobs=_MAIN_PF_N_JOBS, backend='loky', verbose=5\n"
        ")(delayed(_run_main_pf_well)(job) for job in enumerate(test_wells))\n"
        "rows = []\n"
        "bimodal_report_rows = []\n"
        "PF_SEED_BRANCH_STATS = {}\n"
        "for _well_rows, _well_bimodal_rows, _well_seed_stats in _main_pf_results:\n"
        "    rows.extend(_well_rows)\n"
        "    bimodal_report_rows.extend(_well_bimodal_rows)\n"
        "    PF_SEED_BRANCH_STATS.update(_well_seed_stats)\n"
        "print(f'parallel exact main PF complete: jobs={_MAIN_PF_N_JOBS} wells={len(test_wells)}')\n"
        + post_marker
        + post
    )
    cell["source"] = (prefix + parallel).splitlines(keepends=True)


def _parallelize_gold(cells: list[dict]) -> None:
    start = "    _gold_reports = []\n"
    end = "    _gold_report_df = _gold_pd.DataFrame(_gold_reports)\n"
    matches = [
        index
        for index, cell in enumerate(cells)
        if start in source(cell) and end in source(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Gold loop, found {matches}")
    cell = cells[matches[0]]
    before, remainder = source(cell).split(start, 1)
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
            if _wid in _gold_bimodal_skip_wells:
                print('[gold %d/%d] keeping bimodal hedge for %s' % (_wi, len(_gold_wells), _wid), flush=True)
                return dict(
                    well=_wid,
                    status='skip_bimodal_hedge',
                    bimodal_prefix_guard=True,
                    reason='visible-prefix commit disabled for active bimodal selector well',
                ), [], _local_candidates
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
    print('parallel exact Gold complete: jobs=%d wells=%d' % (_GOLD_N_JOBS, len(_gold_wells)), flush=True)

    _gold_report_df = _gold_pd.DataFrame(_gold_reports)
'''
    cell["source"] = (before + replacement + after).splitlines(keepends=True)


def _prepare_anchor() -> tuple[dict, list[dict], dict]:
    meta = _load_meta()
    notebook, cells = clean_code_cells(SOURCE, skipped=SKIPPED)
    _embed_exact_ridge(cells, meta)
    _parallelize_main_pf(cells)
    _parallelize_gold(cells)

    joined = "\n".join(source(cell) for cell in cells)
    required = (
        "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'",
        "model_package_gated_max_weight=0.00425",
        "SP45_SELECTOR_N_SEEDS = 128",
        "_GOLD_CAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_CAL_SEEDS', '24'))",
        "_GOLD_FINAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_FINAL_SEEDS', '48'))",
        "def _run_main_pf_well",
        "parallel exact main PF complete",
        "def _gold_process_well",
        "parallel exact Gold complete",
        "Guarded PF seed-branch midpoint hedge",
        "Final hidden-set contract transaction",
        "_FAST_META_COEF",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"exact 6.768 fragments missing: {missing}")
    forbidden = (
        'pd.read_csv(CFG.artifacts_path / "data" / "train.csv"',
        "ridge_trainer.fit",
        "for i, wid in enumerate(test_wells):",
        "for _wi, _wid in enumerate(_gold_wells, 1):",
        "Full-stack bimodal CV ablation",
        "def fig_overview",
        "def fig_results",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"slow/diagnostic source leaked into rebuild: {leaked}")
    return notebook, cells, meta


def _hmm_cells() -> list[dict]:
    notebook = json.loads(HMM_SOURCE.read_text(encoding="utf-8"))
    core = copy.deepcopy(notebook["cells"][58])
    core["execution_count"] = None
    core["outputs"] = []
    core.get("metadata", {}).pop("execution", None)
    core.get("metadata", {}).pop("papermill", None)
    if "class HMMParams" not in source(core) or "def run_hmm2" not in source(core):
        raise RuntimeError("EXP087 exact HMM core was not found")

    runner = hmm_runner()
    replacements = {
        "_HMM_WEIGHT = 0.15": "_HMM_WEIGHT = 0.25",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "runtime_safe_public6768_plus_student_t_hmm025"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t second-order HMM posterior mean over the exact Public "
            "6.768 anchor."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    return [core, code_cell(runner)]


def _write(
    notebook: dict,
    cells: list[dict],
    meta: dict,
    output: Path,
    slug: str,
    strategy: str,
    with_hmm: bool,
) -> Path:
    if with_hmm:
        cells.extend(_hmm_cells())
    extra = (
        "_sd_audit['source'] = 'yasut0ra/rogii-codex-exact-public-6-768-v1'\n"
        "_sd_audit['source_public_score'] = 6.768\n"
        "_sd_audit['submission_profile'] = SUBMISSION_PROFILE\n"
        "_sd_audit['runtime_safe_meta'] = True\n"
        "_sd_audit['parallel_main_pf_jobs'] = int(_MAIN_PF_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}"
    )
    if with_hmm:
        extra += "\n_sd_audit['hmm'] = _HMM_AUDIT"
    notebook["cells"] = [
        code_cell("import time as _sd_time\n_SD_STARTED = _sd_time.time()\n"),
        *cells,
        code_cell(integrity_audit(strategy, "_SD_STARTED", extra)),
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


def build_anchor() -> Path:
    notebook, cells, meta = _prepare_anchor()
    return _write(
        notebook,
        cells,
        meta,
        ANCHOR_OUTPUT,
        "rogii-codex6768-exact",
        "runtime_safe_exact_public6768_anchor",
        with_hmm=False,
    )


def build_hmm() -> Path:
    notebook, cells, meta = _prepare_anchor()
    return _write(
        notebook,
        cells,
        meta,
        HMM_OUTPUT,
        "rogii-codex6768-student-hmm025",
        "runtime_safe_public6768_plus_student_t_hmm025",
        with_hmm=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", choices=("anchor", "hmm"))
    args = parser.parse_args()
    builders = {"anchor": build_anchor, "hmm": build_hmm}
    for target in args.targets or ("anchor", "hmm"):
        print(f"built {builders[target]()}")


if __name__ == "__main__":
    main()
