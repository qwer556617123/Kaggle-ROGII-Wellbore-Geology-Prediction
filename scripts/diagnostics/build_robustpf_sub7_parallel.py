"""Build the public robust-PF recipe with deterministic per-well parallelism."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from build_robustpf_sub7_rebuild import AUDIT_CELL, SOURCE_METADATA, SOURCE_NOTEBOOK, code_cell


TARGET_SLUG = "rogii-robustpf-sub7-parallel"
TARGET_DIR = Path("kaggle") / TARGET_SLUG


def parallelize_well_loop(source: str) -> str:
    """Wrap the independent test-well loop and merge results in source order."""
    lines = source.splitlines()
    init_index = lines.index("rows = []")
    loop_index = lines.index("for i, wid in enumerate(test_wells):")
    suffix_index = lines.index("if robust_pf_report_rows:")
    if lines[init_index : loop_index + 1] != [
        "rows = []",
        "bimodal_report_rows = []",
        "robust_pf_report_rows = []",
        "for i, wid in enumerate(test_wells):",
    ]:
        raise RuntimeError("unexpected robust-PF loop preamble")

    prefix = lines[:init_index]
    body = lines[loop_index + 1 : suffix_index]
    suffix = lines[suffix_index:]
    transformed = prefix + [
        "from joblib import Parallel as _RobustParallel, delayed as _robust_delayed",
        "",
        "def _robust_process_well(i, wid):",
        "    rows = []",
        "    bimodal_report_rows = []",
        "    robust_pf_report_rows = []",
    ]
    # The original loop body already has the four spaces required by the function.
    transformed.extend(body)
    transformed.extend(
        [
            "    return rows, bimodal_report_rows, robust_pf_report_rows",
            "",
            "_robust_results = _RobustParallel(",
            "    n_jobs=min(3, len(test_wells)), prefer='threads', require='sharedmem'",
            ")(",
            "    _robust_delayed(_robust_process_well)(i, wid)",
            "    for i, wid in enumerate(test_wells)",
            ")",
            "rows = []",
            "bimodal_report_rows = []",
            "robust_pf_report_rows = []",
            "for _well_rows, _well_bimodal, _well_robust in _robust_results:",
            "    rows.extend(_well_rows)",
            "    bimodal_report_rows.extend(_well_bimodal)",
            "    robust_pf_report_rows.extend(_well_robust)",
            "",
        ]
    )
    transformed.extend(suffix)
    result = "\n".join(transformed) + "\n"
    if result.count("def _robust_process_well") != 1:
        raise RuntimeError("parallel wrapper count mismatch")
    return result


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
    notebook["cells"] = [copy.deepcopy(cell) for cell in notebook["cells"][:59]]
    cell = notebook["cells"][27]
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    cell["source"] = parallelize_well_loop(source)
    for item in notebook["cells"]:
        if item.get("cell_type") == "code":
            item["execution_count"] = None
            item["outputs"] = []
    audit = AUDIT_CELL.replace(
        "public_robust_pf_raw128_smooth32_vp_balanced_exact",
        "public_robust_pf_raw128_smooth32_parallel_exact",
    )
    notebook["cells"].append(code_cell(audit))
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
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"built {notebook_path}")


if __name__ == "__main__":
    main()
