"""
Data preprocessing module for NephroDetect CKD prediction.
Handles missing values, encoding, feature selection, and train/test split.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Features used for prediction (as per project specification)
FEATURE_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "sc",
    "hemo",
]
TARGET_COLUMN = "class"


def find_dataset_path() -> Path:
    """Locate the CKD dataset CSV in the data directory."""
    candidates = [
        DATA_DIR / "kidney_disease.csv",
        DATA_DIR / "chronic_kidney_disease.csv",
        DATA_DIR / "ckd.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    csv_files = list(DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"No dataset found in {DATA_DIR}. "
        "Download the Kaggle CKD dataset and place it in data/ "
        "(e.g., kidney_disease.csv)."
    )


def load_raw_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load raw CKD dataset from CSV."""
    if filepath is None:
        filepath = find_dataset_path()

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and remove ID column if present."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' and empty strings, then impute missing numeric values."""
    df = df.copy()
    df.replace(["?", ""], np.nan, inplace=True)

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target: ckd -> 1, notckd -> 0."""
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"ckd": 1, "notckd": 0})
    )

    invalid = df[TARGET_COLUMN].isna() | ~df[TARGET_COLUMN].isin([0, 1])
    if invalid.any():
        df = df[~invalid]

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only required feature columns and target."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def remove_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and out-of-range values."""
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Reasonable clinical ranges for sanity filtering
    ranges = {
        "age": (1, 120),
        "bp": (50, 200),
        "sg": (1.0, 1.05),
        "al": (0, 5),
        "su": (0, 5),
        "bgr": (50, 500),
        "sc": (0.1, 20),
        "hemo": (3, 20),
    }

    for col, (low, high) in ranges.items():
        if col in df.columns:
            df = df[(df[col] >= low) & (df[col] <= high)]

    return df


def preprocess_data(
    filepath: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test.
    """
    df = load_raw_data(filepath)
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = encode_target(df)
    df = select_features(df)
    df = remove_inconsistencies(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save processed train/test splits for retraining."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


def load_all_data_for_retrain(filepath: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess full dataset for retraining (no split)."""
    df = load_raw_data(filepath)
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = encode_target(df)
    df = select_features(df)
    df = remove_inconsistencies(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    save_processed_data(X_train, X_test, y_train, y_test)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {FEATURE_COLUMNS}")
    print("Processed data saved to data/processed/")
