"""Build the PRVS grouped-OOF meta trajectory without the final overwrite guards."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_prvs_oof_meta_rebuild import (
    AUDIT_CELL,
    SOURCE_METADATA,
    SOURCE_NOTEBOOK,
    code_cell,
)


TARGETS = {
    "direct": {
        "slug": "rogii-prvs-oof-meta-direct",
        "selector_pf_seeds": None,
        "strategy": "prvs_grouped_oof_meta_residual_direct_no_final_guards",
    },
    "direct_fast": {
        "slug": "rogii-prvs-oof-meta-direct-fast",
        "selector_pf_seeds": 80,
        "strategy": "prvs_grouped_oof_meta_residual_direct_pf80_runtime_safe",
    },
}
FALLBACK_SHA256 = "fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9"


def fix_source_quotes(notebook: dict) -> None:
    old = "print(f'  PF {int(globals().get('SELECTOR_PF_SEEDS', SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK scales={SELECTOR_SCALES}')"
    new = 'print(f"  PF {int(globals().get(\'SELECTOR_PF_SEEDS\', SP45_SELECTOR_N_SEEDS))}-seed lik-ensemble OK scales={SELECTOR_SCALES}")'
    fixes = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if old in source:
            source = source.replace(old, new)
            fixes += 1
        cell["source"] = source
    if fixes != 1:
        raise RuntimeError(f"expected one source quote fix, found {fixes}")


def validate_notebook(notebook: dict, label: str) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        compile(source, f"{label}:cell-{index}", "exec")


def build(target: str) -> Path:
    config = TARGETS[target]
    target_slug = config["slug"]
    target_dir = Path("kaggle") / target_slug
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    # Cell 65 writes the OOF-meta trajectory. Cells 66-68 overwrite it with same-ID guards/VP.
    notebook["cells"] = [copy.deepcopy(cell) for cell in notebook["cells"][:66]]
    notebook["cells"].insert(
        0,
        code_cell(
            "import time as _oof_meta_time\n_OOF_META_STARTED_AT = _oof_meta_time.time()\n"
        ),
    )
    fix_source_quotes(notebook)
    if config["selector_pf_seeds"] is not None:
        old = "SELECTOR_PF_SEEDS = SP45_SELECTOR_N_SEEDS"
        new = f"SELECTOR_PF_SEEDS = {int(config['selector_pf_seeds'])}"
        replacements = 0
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if old in source:
                source = source.replace(old, new)
                replacements += 1
            cell["source"] = source
        if replacements != 1:
            raise RuntimeError(f"expected one selector PF seed replacement, found {replacements}")
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    audit = AUDIT_CELL.replace(
        "prvs_grouped_oof_meta_residual_slim",
        config["strategy"],
    )
    notebook["cells"].append(code_cell(audit))
    notebook["cells"].append(
        code_cell(
            f"if _OM_AUDIT['submission_sha256'] == {FALLBACK_SHA256!r}:\n"
            "    raise RuntimeError('direct OOF-meta output collapsed to fallback hash')\n"
            "print('Direct OOF-meta differs from fallback:', _OM_AUDIT['submission_sha256'])\n"
        )
    )
    notebook.setdefault("metadata", {}).pop("papermill", None)
    validate_notebook(notebook, target_slug)

    target_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = target_dir / f"{target_slug}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=True), encoding="utf-8")
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata["id"] = f"qwer556617123/{target_slug}"
    metadata["title"] = target_slug
    metadata["code_file"] = notebook_path.name
    metadata["dataset_sources"] = [
        source for source in metadata.get("dataset_sources", []) if str(source).strip()
    ]
    (target_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"built {notebook_path}")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(TARGETS), nargs="?", default="direct")
    args = parser.parse_args()
    build(args.target)


if __name__ == "__main__":
    main()
