"""
Model training module for NephroDetect.
Trains Random Forest Classifier and saves the model.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import (
    FEATURE_COLUMNS,
    preprocess_data,
    save_processed_data,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
MODEL_PATH = MODELS_DIR / "ckd_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EXPERIMENT_LOG = LOGS_DIR / "experiment_log.txt"

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}


def get_next_version() -> str:
    """Generate next model version based on existing experiment logs."""
    if not EXPERIMENT_LOG.exists():
        return "v1.0"

    count = 0
    with open(EXPERIMENT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return f"v1.{count}"


def log_experiment(
    version: str,
    accuracy: float,
    params: dict,
    extra_metrics: dict | None = None,
) -> None:
    """Append experiment details to logs/experiment_log.txt."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "version": version,
        "accuracy": round(accuracy, 4),
        "date": timestamp,
        "parameters": params,
    }
    if extra_metrics:
        entry.update(extra_metrics)

    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Experiment logged: {entry}")


def train_model(
    params: dict | None = None,
    log_experiment_flag: bool = True,
) -> tuple[RandomForestClassifier, dict]:
    """
    Train Random Forest model on preprocessed data.
    Returns trained model and training info dict.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_params = {**DEFAULT_PARAMS, **(params or {})}
    X_train, X_test, y_train, y_test = preprocess_data()
    save_processed_data(X_train, X_test, y_train, y_test)

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    version = get_next_version()
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "version": version,
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "parameters": model_params,
        "features": FEATURE_COLUMNS,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if log_experiment_flag:
        log_experiment(
            version=version,
            accuracy=test_accuracy,
            params=model_params,
            extra_metrics={
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
            },
        )

    print(f"Model saved to {MODEL_PATH}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")

    return model, metadata


def retrain_model(params: dict | None = None) -> tuple[RandomForestClassifier, dict]:
    """Retrain model when new data is added to data/ folder."""
    print("Retraining model with updated dataset...")
    return train_model(params=params, log_experiment_flag=True)


if __name__ == "__main__":
    train_model()
