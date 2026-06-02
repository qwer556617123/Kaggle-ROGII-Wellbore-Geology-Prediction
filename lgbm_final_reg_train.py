"""Root wrapper for the stable final-regression training script."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parent / "scripts" / "main" / "lgbm_final_reg_train.py",
        run_name="__main__",
    )
