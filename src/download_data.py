"""
Download Chronic Kidney Disease dataset.
Sources: UCI ML Repository (default) or Kaggle API.
"""

import shutil
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

KAGGLE_DATASET = "mansoordatascience/ckd-prediction"
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/336/data.csv"
OUTPUT_FILENAME = "kidney_disease.csv"


def download_from_uci() -> Path:
    """Download the real CKD dataset from UCI ML Repository (400 patients)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / OUTPUT_FILENAME

    print(f"Downloading from UCI: {UCI_DATASET_URL}")
    urllib.request.urlretrieve(UCI_DATASET_URL, target)

    import pandas as pd

    df = pd.read_csv(target)
    df.columns = df.columns.str.strip().str.lower()
    print(f"UCI dataset saved to {target} ({len(df)} rows, {len(df.columns)} columns)")
    return target


def download_from_kaggle() -> Path:
    """Download dataset using Kaggle API."""
    try:
        from kaggle.api import KaggleApi
    except ImportError as err:
        raise ImportError("Install kaggle: pip install kaggle") from err

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

    print(f"Kaggle dataset saved to {target}")
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

    source = sys.argv[1] if len(sys.argv) > 1 else "uci"

    if source == "--sample":
        create_sample_dataset()
    elif source == "kaggle":
        try:
            download_from_kaggle()
        except Exception as exc:
            print(f"Kaggle download failed: {exc}")
            print("Falling back to UCI dataset...")
            download_from_uci()
    else:
        try:
            download_from_uci()
        except Exception as exc:
            print(f"UCI download failed: {exc}")
            print("Creating sample dataset instead...")
            create_sample_dataset()
