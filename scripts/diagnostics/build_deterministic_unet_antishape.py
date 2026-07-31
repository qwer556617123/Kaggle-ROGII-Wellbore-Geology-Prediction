"""Build the symmetric -0.25 U-Net shape probe from the audited +0.25 notebook."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def build(source_path: Path, output_dir: Path) -> Path:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    changed_move = changed_audit = changed_strategy = False
    cells = []
    for original in notebook["cells"]:
        cell = copy.deepcopy(original)
        source = _source(cell)
        if "0.25 * _du_basis" in source:
            source = source.replace("0.25 * _du_basis", "-0.25 * _du_basis")
            changed_move = True
        if "'unet_weight': 0.25" in source:
            source = source.replace("'unet_weight': 0.25", "'unet_weight': -0.25")
            changed_audit = True
        if "deterministic7166_native_unet_zero_mean_shape" in source:
            source = source.replace(
                "deterministic7166_native_unet_zero_mean_shape",
                "deterministic7166_native_unet_zero_mean_antishape",
            )
            changed_strategy = True
        cell["source"] = source.splitlines(keepends=True)
        cell["execution_count"] = None
        cell["outputs"] = []
        cells.append(cell)
    if not (changed_move and changed_audit and changed_strategy):
        raise RuntimeError("expected +0.25 shape markers were not all found")
    notebook["cells"] = cells

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "rogii-deterministic-unet-antishape.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    source_metadata = json.loads((source_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    source_metadata.update(
        {
            "id": "qwer556617123/rogii-deterministic-unet-antishape",
            "title": "rogii-deterministic-unet-antishape",
            "code_file": notebook_path.name,
        }
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(source_metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {notebook_path} with symmetric U-Net weight -0.25")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("kaggle/rogii-deterministic-unet/rogii-deterministic-unet.ipynb"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/rogii-deterministic-unet-antishape"),
    )
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
