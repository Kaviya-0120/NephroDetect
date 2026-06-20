"""
Download Chronic Kidney Disease dataset from Kaggle.
Requires Kaggle API credentials (~/.kaggle/kaggle.json).
"""

import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Kaggle dataset: Chronic Kidney Disease Prediction
KAGGLE_DATASET = "mansoordatascience/ckd-prediction"
OUTPUT_FILENAME = "kidney_disease.csv"


def download_from_kaggle() -> Path:
    """Download dataset using Kaggle API."""
    try:
        from kaggle.api import KaggleApi
    except ImportError as err:
        raise ImportError(
            "Install kaggle: pip install kaggle"
        ) from err

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found after Kaggle download.")

    source = csv_files[0]
    target = DATA_DIR / OUTPUT_FILENAME

    if source != target:
        shutil.move(str(source), str(target))

    for zf in DATA_DIR.glob("*.zip"):
        zf.unlink()

    print(f"Dataset saved to {target}")
    return target


def create_sample_dataset() -> Path:
    """
    Create a minimal sample dataset for testing when Kaggle is unavailable.
    Not for production — replace with real Kaggle data.
    """
    import numpy as np
    import pandas as pd

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 200

    data = {
        "id": range(1, n + 1),
        "age": np.random.randint(20, 80, n),
        "bp": np.random.randint(70, 160, n),
        "sg": np.random.choice([1.005, 1.01, 1.015, 1.02, 1.025], n),
        "al": np.random.randint(0, 5, n),
        "su": np.random.randint(0, 5, n),
        "bgr": np.random.randint(70, 300, n),
        "sc": np.round(np.random.uniform(0.5, 5.0, n), 2),
        "hemo": np.round(np.random.uniform(8, 17, n), 1),
    }

    df = pd.DataFrame(data)
    df["class"] = (
        ((df["sc"] > 1.5) | (df["hemo"] < 12) | (df["al"] >= 3))
        .map({True: "ckd", False: "notckd"})
    )

    target = DATA_DIR / OUTPUT_FILENAME
    df.to_csv(target, index=False)
    print(f"Sample dataset created at {target} ({n} rows)")
    print("Replace with real Kaggle data for accurate results.")
    return target


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        create_sample_dataset()
    else:
        try:
            download_from_kaggle()
        except Exception as exc:
            print(f"Kaggle download failed: {exc}")
            print("Creating sample dataset instead...")
            create_sample_dataset()
