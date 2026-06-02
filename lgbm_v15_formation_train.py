"""Root wrapper for the active v15 formation training script."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parent / "scripts" / "main" / "lgbm_v15_formation_train.py",
        run_name="__main__",
    )
