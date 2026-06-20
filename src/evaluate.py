"""
Model evaluation module for NephroDetect.
Computes accuracy, precision, recall, and F1-score.
"""

from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_preprocessing import preprocess_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ckd_model.pkl"


def load_model():
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


def evaluate_model(model=None, verbose: bool = True) -> dict:
    """
    Evaluate model on test set.
    Returns dict with accuracy, precision, recall, f1, confusion matrix.
    """
    if model is None:
        model = load_model()

    _, X_test, _, y_test = preprocess_data()
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Not CKD", "CKD"]
        ),
    }

    if verbose:
        print("=" * 50)
        print("NephroDetect Model Evaluation")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])
        print("\nClassification Report:")
        print(metrics["classification_report"])

    return metrics


if __name__ == "__main__":
    evaluate_model()
