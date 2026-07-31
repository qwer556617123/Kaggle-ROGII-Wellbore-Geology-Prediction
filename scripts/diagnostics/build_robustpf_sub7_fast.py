"""Build a runtime-safe sequential approximation of the robust-PF recipe."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from build_robustpf_sub7_rebuild import AUDIT_CELL, SOURCE_METADATA, SOURCE_NOTEBOOK, code_cell


TARGET_SLUG = "rogii-robustpf-sub7-fast"
TARGET_DIR = Path("kaggle") / TARGET_SLUG


def make_runtime_safe(source: str) -> str:
    gate_start = source.index("        try:\n            _pf_tvt_input")
    final_pf_start = source.index("        pf_raw_by_scale", gate_start)
    fixed_gate = '''        # Runtime-safe fixed gate: all audited visible wells retained W=.125.
        _pf_tvt_input = hw_te['TVT_input'].to_numpy(dtype=float)
        _pf_missing = np.flatnonzero(~np.isfinite(_pf_tvt_input))
        _pf_prefix_end = int(_pf_missing[0]) if len(_pf_missing) else len(_pf_tvt_input)
        _pf_gate_prefix_rows = int(_pf_prefix_end)
        _pf_smooth_weight = 0.125
        _pf_gate_status = 'fixed_w125_runtime'
        _pf_gate_error = ''
        _pf_gate_scores = {}

'''
    result = source[:gate_start] + fixed_gate + source[final_pf_start:]
    replacements = {
        "n_seeds=128": "n_seeds=96",
        "n_seeds=32": "n_seeds=16",
        "'raw_seeds': 128": "'raw_seeds': 96",
        "'smooth_seeds': 32": "'smooth_seeds': 16",
        "PF raw128 + smooth32": "PF raw96 + smooth16",
        "'raw_seeds': 128,": "'raw_seeds': 96,",
        "'smooth_seeds': 32,": "'smooth_seeds': 16,",
    }
    for old, new in replacements.items():
        result = result.replace(old, new)
    if "_pf_gate_winners" in result or "_pf_gate_fraction" in result:
        raise RuntimeError("prefix cutback gate was not fully removed")
    if result.count("n_seeds=96") != 2 or result.count("n_seeds=16") != 1:
        raise RuntimeError("runtime-safe seed replacement count mismatch")
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
    cell["source"] = make_runtime_safe(source)
    for item in notebook["cells"]:
        if item.get("cell_type") == "code":
            item["execution_count"] = None
            item["outputs"] = []
    audit = AUDIT_CELL.replace(
        "public_robust_pf_raw128_smooth32_vp_balanced_exact",
        "public_robust_pf_raw96_smooth16_fixed125_runtime_safe",
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
