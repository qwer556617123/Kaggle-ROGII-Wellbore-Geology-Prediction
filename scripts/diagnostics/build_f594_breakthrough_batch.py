"""Build the exact dynamic F594 anchor and orthogonal continuation probes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import build_codex6768_rebuild as codex
from build_stopdose_breakthrough_notebooks import (
    clean_code_cells,
    code_cell,
    hmm_runner,
    integrity_audit,
    source,
    validate_notebook,
    write_metadata,
)


F594_SOURCE = Path(
    "kaggle/external_reviews/frontier_20260722b/hahaha6594/"
    "hahaha-nondet-agi.ipynb"
)
HMM_SOURCE = Path("kaggle/external_reviews/exp087-hmm/rogii-exp087-hmm.ipynb")
MHA400_SOURCE = Path(
    "kaggle/rogii-mha400-exact/rogii-mha400-exact.ipynb"
)
CONTINUITY_SOURCE = Path(
    "kaggle/external_reviews/frontier_20260722b/exp086b/"
    "rogii-exp086b-mha076-cont.ipynb"
)

F594_OUTPUT = Path("kaggle/rogii-f594-exact")
F594_HMM_OUTPUT = Path("kaggle/rogii-f594-student-hmm015")
MHA400_CONT_OUTPUT = Path("kaggle/rogii-mha400-continuity")
MHA_DOSE_SPECS = {
    "mha500": (5.0, 12.0, Path("kaggle/rogii-mha500-exact"), "rogii-mha500-exact"),
    "mha600": (6.0, 15.0, Path("kaggle/rogii-mha600-exact"), "rogii-mha600-exact"),
}

# Read-only train diagnostics, plots, and full-stack CV. Every scoring writer
# from the learned branch through the dynamic PF branch remains in source order.
F594_SKIPPED = {12, 23, 31, 34, 35, 38, 44, 47, 54}


def _joined(cells: list[dict]) -> str:
    return "\n".join(source(cell) for cell in cells)


def _prepare_f594() -> tuple[dict, list[dict], dict]:
    meta = codex._load_meta()
    notebook, cells = clean_code_cells(F594_SOURCE, skipped=F594_SKIPPED)
    codex._embed_exact_ridge(cells, meta)
    codex._parallelize_main_pf(cells)
    codex._parallelize_gold(cells)
    joined = _joined(cells)
    required = (
        "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'",
        "model_package_gated_max_weight=0.00425",
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0",
        "SP45_SELECTOR_N_SEEDS = 128",
        "_BH_STRENGTH = 0.60",
        "_BH_MIN_MASS = 0.25",
        "_BH_SEP_LOW = 4.00",
        "_BH_SEP_HIGH = 40.00",
        "_BH_CAP = 2.00",
        "globals().get('PF_SEED_BRANCH_STATS', {})",
        "def _run_main_pf_well",
        "def _gold_process_well",
        "Final submission audit",
        "_FAST_META_COEF",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"F594 source fragments missing: {missing}")
    forbidden = (
        'pd.read_csv(CFG.artifacts_path / "data" / "train.csv"',
        "ridge_trainer.fit",
        "for i, wid in enumerate(test_wells):",
        "for _wi, _wid in enumerate(_gold_wells, 1):",
        "Full-stack bimodal CV ablation",
        "def fig_overview",
        "def fig_results",
        "_EX_EXPECTED_WELL",
        "FRONTIER_TARGET_WELL",
    )
    leaked = [fragment for fragment in forbidden if fragment in joined]
    if leaked:
        raise RuntimeError(f"slow, fixed-ID, or diagnostic F594 source leaked: {leaked}")
    return notebook, cells, meta


def _hmm_cells(weight: float) -> list[dict]:
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
            "dynamic_f594_plus_student_t_hmm015"
        ),
        "Exact second-order HMM posterior-mean blend over the 7.053 anchor.": (
            "Student-t posterior mean over the dynamic F594 PF-branch anchor."
        ),
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise RuntimeError(f"expected one HMM runner replacement: {old}")
        runner = runner.replace(old, new)
    return [core, code_cell(runner)]


def _write_f594(with_hmm: bool) -> Path:
    notebook, cells, meta = _prepare_f594()
    if with_hmm:
        cells.extend(_hmm_cells(0.15))
        output = F594_HMM_OUTPUT
        slug = "rogii-f594-student-hmm015"
        strategy = "dynamic_f594_plus_student_t_hmm015"
    else:
        output = F594_OUTPUT
        slug = "rogii-f594-exact"
        strategy = "runtime_safe_exact_dynamic_f594_anchor"

    extra = (
        "_sd_audit['source'] = 'johnjanson/hahaha-nondet-agi'\n"
        "_sd_audit['source_public_score'] = 6.594\n"
        "_sd_audit['submission_profile'] = SUBMISSION_PROFILE\n"
        "_sd_audit['branch_strength'] = float(_BH_STRENGTH)\n"
        "_sd_audit['branch_min_mass'] = float(_BH_MIN_MASS)\n"
        "_sd_audit['branch_sep_low'] = float(_BH_SEP_LOW)\n"
        "_sd_audit['branch_sep_high'] = float(_BH_SEP_HIGH)\n"
        "_sd_audit['branch_cap'] = float(_BH_CAP)\n"
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
    write_metadata(F594_SOURCE, output, slug)
    return path


def _continuity_cell() -> dict:
    notebook = json.loads(CONTINUITY_SOURCE.read_text(encoding="utf-8"))
    matches = [
        cell
        for cell in notebook["cells"]
        if source(cell).startswith("# Target-free post-composition U-continuity fade.")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one continuity source cell, found {len(matches)}")
    cell = copy.deepcopy(matches[0])
    cell["execution_count"] = None
    cell["outputs"] = []
    cell.get("metadata", {}).pop("execution", None)
    cell.get("metadata", {}).pop("papermill", None)
    text = source(cell)
    if "_UC_CAP = 8.000000" not in text or "_UC_TAU = 240.000000" not in text:
        raise RuntimeError("continuity source settings changed")
    return cell


def build_mha400_continuity() -> Path:
    notebook = json.loads(MHA400_SOURCE.read_text(encoding="utf-8"))
    if not notebook.get("cells"):
        raise RuntimeError("MHA400 source notebook is empty")
    final_audit = source(notebook["cells"][-1])
    if "stop-dose final audit" not in final_audit:
        raise RuntimeError("MHA400 final integrity audit was not found")
    notebook["cells"].insert(-1, _continuity_cell())
    joined = _joined(notebook["cells"])
    required = (
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "4.0, 0.22, 1.0, 40.0, 10.0",
        "_UC_CAP = 8.000000",
        "_UC_TAU = 240.000000",
        "Final hidden-set contract transaction",
        "stop-dose final audit",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise RuntimeError(f"MHA400-continuity fragments missing: {missing}")

    slug = "rogii-mha400-continuity"
    validate_notebook(notebook, slug)
    MHA400_CONT_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = MHA400_CONT_OUTPUT / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(MHA400_SOURCE, MHA400_CONT_OUTPUT, slug)
    return path


def build_mha_dose(spec_name: str) -> Path:
    alpha, cap, output, slug = MHA_DOSE_SPECS[spec_name]
    notebook = json.loads(MHA400_SOURCE.read_text(encoding="utf-8"))
    old_config = (
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        "4.0, 0.22, 1.0, 40.0, 10.0"
    )
    new_config = (
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = "
        f"{alpha:.1f}, 0.22, 1.0, 40.0, {cap:.1f}"
    )
    config_matches = 0
    strategy_matches = 0
    for cell in notebook["cells"]:
        text = source(cell)
        if old_config in text:
            config_matches += text.count(old_config)
            text = text.replace(old_config, new_config)
        old_strategy = "runtime_safe_exact_mha400_sep1_hidden_hedge_frontier"
        if old_strategy in text:
            strategy_matches += text.count(old_strategy)
            text = text.replace(
                old_strategy,
                f"runtime_safe_exact_{spec_name}_sep1_hidden_hedge_frontier",
            )
        cell["source"] = text.splitlines(keepends=True)
    if config_matches != 1 or strategy_matches != 1:
        raise RuntimeError(
            f"unexpected MHA dose replacement counts: config={config_matches}, "
            f"strategy={strategy_matches}"
        )
    joined = _joined(notebook["cells"])
    if new_config not in joined or old_config in joined:
        raise RuntimeError(f"{spec_name} dose configuration was not isolated")
    if "00e12e8b" in joined:
        raise RuntimeError(f"{spec_name} unexpectedly contains a fixed public well ID")

    validate_notebook(notebook, slug)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    write_metadata(MHA400_SOURCE, output, slug)
    return path


def main() -> None:
    builders = {
        "f594": lambda: _write_f594(False),
        "f594_hmm": lambda: _write_f594(True),
        "mha400_cont": build_mha400_continuity,
        **{
            name: (lambda spec_name=name: build_mha_dose(spec_name))
            for name in MHA_DOSE_SPECS
        },
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()
    unknown = sorted(set(args.targets) - set(builders))
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)}")
    for target in args.targets or tuple(builders):
        print(f"built {builders[target]()}")


if __name__ == "__main__":
    main()
