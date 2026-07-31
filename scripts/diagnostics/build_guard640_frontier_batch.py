"""Build the guarded-continuity frontier pair and exact MHA250 control."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import build_codex6768_rebuild as codex
import build_geometry_refblend_hmm as geometry
from build_stopdose_breakthrough_notebooks import (
    clean_code_cells,
    code_cell,
    hmm_runner,
    integrity_audit,
    source,
    validate_notebook,
    write_metadata,
)


GUARD_SOURCE = Path(
    "kaggle/external_reviews/frontier_20260721b/guard640/"
    "rogii-codex-public-6-40-guard.ipynb"
)
MHA_SOURCE = Path(
    "kaggle/external_reviews/frontier_20260721/mha250sep2/"
    "rogii-det-mha250sep2-public6858.ipynb"
)
MHA_FRONTIER_SPECS = {
    "mha300_mm15": {
        "source": Path(
            "kaggle/external_reviews/frontier_20260722/exp082/"
            "rogii-exp082-best-mm15.ipynb"
        ),
        "output": Path("kaggle/rogii-mha300-mm15-exact"),
        "slug": "rogii-mha300-mm15-exact",
        "config": (3.0, 0.15, 1.5, 40.0, 6.0),
        "source_ref": "hirotayusuke/rogii-exp082-best-mm15",
    },
    "mha300_hi60": {
        "source": Path(
            "kaggle/external_reviews/frontier_20260722/exp083/"
            "rogii-exp083-best-hi60.ipynb"
        ),
        "output": Path("kaggle/rogii-mha300-hi60-exact"),
        "slug": "rogii-mha300-hi60-exact",
        "config": (3.0, 0.22, 1.5, 60.0, 6.0),
        "source_ref": "hirotayusuke/rogii-exp083-best-hi60",
    },
    "mha400_sep1": {
        "source": Path(
            "kaggle/external_reviews/frontier_20260722/exp084/"
            "rogii-exp084-mha40sep1cap10.ipynb"
        ),
        "output": Path("kaggle/rogii-mha400-exact"),
        "slug": "rogii-mha400-exact",
        "config": (4.0, 0.22, 1.0, 40.0, 10.0),
        "source_ref": "hirotayusuke/rogii-exp084-mha40sep1cap10",
    },
}
HMM_SOURCE = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")

GUARD_OUTPUT = Path("kaggle/rogii-guard640-exact-cpu")
GUARD_HMM_OUTPUT = Path("kaggle/rogii-guard640-student-hmm015-cpu")
MHA_OUTPUT = Path("kaggle/rogii-mha250sep2-exact")

# Train-score cells, read-only diagnostics, and visualizations only.
GUARD_SKIPPED = {13, 24, 32, 35, 36, 39, 45, 48, 56} | set(range(64, 78))
MHA_SKIPPED = {
    16,
    23,
    26,
    27,
    30,
    36,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
} | set(range(53, 67))


def _joined(cells: list[dict]) -> str:
    return "\n".join(source(cell) for cell in cells)


def _prepare_guard() -> tuple[dict, list[dict], dict]:
    meta = codex._load_meta()
    notebook, cells = clean_code_cells(GUARD_SOURCE, skipped=GUARD_SKIPPED)
    codex._embed_exact_ridge(cells, meta)
    codex._parallelize_main_pf(cells)
    codex._parallelize_gold(cells)
    joined = _joined(cells)
    required = (
        "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'",
        "model_package_gated_max_weight=0.490000",
        "model_package_gated_scale=192.000000",
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9",
        "_UC_CAP = 8.000000",
        "_UC_TAU = 240.000000",
        "Public-test overlap calibration",
        "_OV_WEIGHT = 0.055",
        "def _run_main_pf_well",
        "def _gold_process_well",
        "Final hidden-set contract transaction",
        "_FAST_META_COEF",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"guarded 6.40 source fragments missing: {missing}")
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
        raise RuntimeError(f"slow/diagnostic source leaked into guard rebuild: {leaked}")
    return notebook, cells, meta


def _prepare_mha_source(
    source_path: Path,
    expected_config: tuple[float, float, float, float, float],
    label: str,
) -> tuple[dict, list[dict], dict]:
    meta = codex._load_meta()
    notebook, cells = clean_code_cells(source_path, skipped=MHA_SKIPPED)
    codex._embed_exact_ridge(cells, meta)
    geometry._parallelize_reference_loop(cells)
    geometry._parallelize_gold_loop(cells)
    joined = _joined(cells)
    config_fragment = (
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        + ", ".join(str(value) for value in expected_config)
    )
    required = (
        'os.environ["ROGII_GOLD_PROFILE"] = "conservative"',
        "n_particles=500, n_seeds=128, stats_out=_bstats",
        "_GOLD_CAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_CAL_SEEDS', '24'))",
        "_GOLD_FINAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_FINAL_SEEDS', '48'))",
        "_BC_SHIFT = -0.40",
        config_fragment,
        "def _run_reference_well",
        "def _gold_process_well",
        "Final hidden-set contract transaction",
        "_FAST_META_COEF",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"{label} source fragments missing: {missing}")
    forbidden = (
        'pd.read_csv(CFG.artifacts_path / "data" / "train.csv"',
        "ridge_trainer.fit",
        "for i, wid in enumerate(test_wells):",
        "for _wi, _wid in enumerate(_gold_wells, 1):",
        "Legal heel-cal predictor v2",
        "Smoother SALVAGE sweep",
        "Smoother-vs-STACK OOF blend test",
        "LB PROBE v4",
        "def fig_overview",
        "def fig_results",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"slow/diagnostic source leaked into {label} rebuild: {leaked}")
    return notebook, cells, meta


def _prepare_mha() -> tuple[dict, list[dict], dict]:
    return _prepare_mha_source(
        MHA_SOURCE,
        (2.5, 0.22, 2.0, 40.0, 4.0),
        "MHA250",
    )


def _student_hmm_cells(weight: float) -> list[dict]:
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
        "_HMM_WEIGHT = 0.15": f"_HMM_WEIGHT = {float(weight)!r}",
        "result = run_hmm2(hw[TEST_COLS].copy(), tw)": (
            "result = run_hmm2(hw[TEST_COLS].copy(), tw, "
            "params=HMMParams(emission='t', sigma_mode='std'))"
        ),
        "pkadopt_7053_plus_exact_second_order_hmm": (
            "guard640_continuity_plus_student_t_hmm015_before_overlap"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t HMM over the guarded continuity candidate, before the "
            "exact-overlap terminal calibration."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    return [core, code_cell(runner)]


def _insert_before(cells: list[dict], prefix: str, additions: list[dict]) -> None:
    matches = [index for index, cell in enumerate(cells) if source(cell).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one insertion point {prefix!r}, found {matches}")
    cells[matches[0] : matches[0]] = additions


def _write(
    notebook: dict,
    cells: list[dict],
    meta: dict,
    source_path: Path,
    output: Path,
    slug: str,
    strategy: str,
    extra: str,
    enable_gpu: bool,
) -> Path:
    common = (
        "_sd_audit['runtime_safe_meta'] = True\n"
        f"_sd_audit['meta_oof_rmse'] = {float(meta['overall_oof_rmse'])!r}\n"
        f"_sd_audit['meta_coef'] = {meta['coef']!r}\n"
        f"_sd_audit['meta_intercept'] = {float(meta['intercept'])!r}\n"
    )
    notebook["cells"] = [
        code_cell("import time as _sd_time\n_SD_STARTED = _sd_time.time()\n"),
        *cells,
        code_cell(integrity_audit(strategy, "_SD_STARTED", common + extra)),
    ]
    validate_notebook(notebook, slug)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(source_path, output, slug)
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["enable_gpu"] = bool(enable_gpu)
    metadata["enable_tpu"] = False
    if not enable_gpu:
        metadata.pop("machine_shape", None)
        metadata["keywords"] = [
            item for item in metadata.get("keywords", []) if str(item).lower() != "gpu"
        ]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def build_guard() -> Path:
    notebook, cells, meta = _prepare_guard()
    extra = (
        "_sd_audit['source'] = 'fleongg/rogii-codex-public-6-40-guard'\n"
        "_sd_audit['source_claimed_score'] = 6.40\n"
        "_sd_audit['package_max_weight'] = float(MODEL_PACKAGE_GATED_MAX_WEIGHT)\n"
        "_sd_audit['package_scale'] = float(MODEL_PACKAGE_GATED_SCALE)\n"
        "_sd_audit['continuity_cap'] = float(_UC_CAP)\n"
        "_sd_audit['continuity_tau'] = float(_UC_TAU)\n"
        "_sd_audit['public_overlap_weight'] = float(_OV_WEIGHT)\n"
        "_sd_audit['parallel_main_pf_jobs'] = int(_MAIN_PF_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)"
    )
    return _write(
        notebook,
        cells,
        meta,
        GUARD_SOURCE,
        GUARD_OUTPUT,
        "rogii-guard640-exact-cpu",
        "runtime_safe_exact_guard640_package049_continuity",
        extra,
        enable_gpu=False,
    )


def build_guard_hmm() -> Path:
    notebook, cells, meta = _prepare_guard()
    _insert_before(cells, "# Public-test overlap calibration.", _student_hmm_cells(0.15))
    extra = (
        "_sd_audit['source'] = 'fleongg/rogii-codex-public-6-40-guard'\n"
        "_sd_audit['source_claimed_score'] = 6.40\n"
        "_sd_audit['package_max_weight'] = float(MODEL_PACKAGE_GATED_MAX_WEIGHT)\n"
        "_sd_audit['continuity_cap'] = float(_UC_CAP)\n"
        "_sd_audit['continuity_tau'] = float(_UC_TAU)\n"
        "_sd_audit['public_overlap_weight'] = float(_OV_WEIGHT)\n"
        "_sd_audit['parallel_main_pf_jobs'] = int(_MAIN_PF_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)\n"
        "_sd_audit['hmm_before_overlap'] = _HMM_AUDIT"
    )
    return _write(
        notebook,
        cells,
        meta,
        GUARD_SOURCE,
        GUARD_HMM_OUTPUT,
        "rogii-guard640-student-hmm015-cpu",
        "guard640_continuity_plus_student_t_hmm015_before_overlap",
        extra,
        enable_gpu=False,
    )


def build_mha() -> Path:
    notebook, cells, meta = _prepare_mha()
    extra = (
        "_sd_audit['source'] = 'wbfranci/rogii-det-mha250sep2-public6858'\n"
        "_sd_audit['source_claimed_score'] = 6.858\n"
        "_sd_audit['mha_alpha'] = float(_MH_ALPHA)\n"
        "_sd_audit['mha_min_mass'] = float(_MH_MINMASS)\n"
        "_sd_audit['mha_sep_low'] = float(_MH_SEPLO)\n"
        "_sd_audit['mha_cap'] = float(_MH_CAP)\n"
        "_sd_audit['parallel_main_pf_jobs'] = int(_REFERENCE_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)"
    )
    return _write(
        notebook,
        cells,
        meta,
        MHA_SOURCE,
        MHA_OUTPUT,
        "rogii-mha250sep2-exact",
        "runtime_safe_exact_mha250sep2_public6858_control",
        extra,
        enable_gpu=True,
    )


def build_mha_frontier(spec_name: str) -> Path:
    spec = MHA_FRONTIER_SPECS[spec_name]
    config = spec["config"]
    notebook, cells, meta = _prepare_mha_source(
        spec["source"], config, spec_name
    )
    extra = (
        f"_sd_audit['source'] = {spec['source_ref']!r}\n"
        "_sd_audit['source_claimed_score'] = None\n"
        "_sd_audit['mha_alpha'] = float(_MH_ALPHA)\n"
        "_sd_audit['mha_min_mass'] = float(_MH_MINMASS)\n"
        "_sd_audit['mha_sep_low'] = float(_MH_SEPLO)\n"
        "_sd_audit['mha_sep_high'] = float(_MH_SEPHI)\n"
        "_sd_audit['mha_cap'] = float(_MH_CAP)\n"
        "_sd_audit['parallel_main_pf_jobs'] = int(_REFERENCE_N_JOBS)\n"
        "_sd_audit['parallel_gold_jobs'] = int(_GOLD_N_JOBS)"
    )
    return _write(
        notebook,
        cells,
        meta,
        spec["source"],
        spec["output"],
        spec["slug"],
        f"runtime_safe_exact_{spec_name}_hidden_hedge_frontier",
        extra,
        enable_gpu=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ("guard", "guard_hmm", "mha", *MHA_FRONTIER_SPECS)
    parser.add_argument("targets", nargs="*", choices=choices)
    args = parser.parse_args()
    builders = {
        "guard": build_guard,
        "guard_hmm": build_guard_hmm,
        "mha": build_mha,
        **{
            name: (lambda spec_name=name: build_mha_frontier(spec_name))
            for name in MHA_FRONTIER_SPECS
        },
    }
    for target in args.targets or tuple(builders):
        print(f"built {builders[target]()}")


if __name__ == "__main__":
    main()
