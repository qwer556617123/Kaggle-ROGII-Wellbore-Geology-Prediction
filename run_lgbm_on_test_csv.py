"""Root wrapper for the local test CSV inference helper."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parent / "scripts" / "main" / "run_lgbm_on_test_csv.py",
        run_name="__main__",
    )
