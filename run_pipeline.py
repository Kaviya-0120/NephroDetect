"""
One-command setup: download data, preprocess, train, evaluate.
Usage: python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run_step(label: str, script: str, *args: str) -> None:
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    cmd = [sys.executable, str(SRC / script), *args]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    print("NephroDetect MLOps Pipeline")
    print("=" * 60)

    run_step("Step 1/4: Download Dataset (UCI)", "download_data.py")
    run_step("Step 2/4: Preprocess Data", "data_preprocessing.py")
    run_step("Step 3/4: Train Model", "train.py")
    run_step("Step 4/4: Evaluate Model", "evaluate.py")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete! Start the app with: python app.py")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
