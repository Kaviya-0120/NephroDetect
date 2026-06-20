"""
NephroDetect Flask Application
Early Prediction of Chronic Kidney Disease using ML and MLOps
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

# Add src to path for imports
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import FEATURE_COLUMNS  # noqa: E402
from train import retrain_model  # noqa: E402

app = Flask(__name__)
app.secret_key = "nephrodetect-mlops-secret-key"

MODEL_PATH = BASE_DIR / "models" / "ckd_model.pkl"
LOGS_DIR = BASE_DIR / "logs"
PREDICTION_LOG = LOGS_DIR / "prediction_logs.csv"

FEATURE_LABELS = {
    "age": "Age (years)",
    "bp": "Blood Pressure (mm Hg)",
    "sg": "Specific Gravity",
    "al": "Albumin (0-5)",
    "su": "Sugar (0-5)",
    "bgr": "Blood Glucose Random (mg/dL)",
    "sc": "Serum Creatinine (mg/dL)",
    "hemo": "Hemoglobin (g/dL)",
}

LOG_HEADERS = [
    "timestamp",
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "sc",
    "hemo",
    "prediction",
    "risk_level",
    "probability",
]


def load_model():
    """Load trained model; return None if not found."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def ensure_prediction_log():
    """Create prediction log CSV with headers if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTION_LOG.exists():
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)


def log_prediction(inputs: dict, prediction: str, risk_level: str, probability: float):
    """Append prediction to logs/prediction_logs.csv."""
    ensure_prediction_log()
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        inputs["age"],
        inputs["bp"],
        inputs["sg"],
        inputs["al"],
        inputs["su"],
        inputs["bgr"],
        inputs["sc"],
        inputs["hemo"],
        prediction,
        risk_level,
        round(probability, 4),
    ]
    with open(PREDICTION_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def get_risk_level(prediction: int, probability: float, inputs: dict) -> str:
    """Determine risk level based on prediction and clinical indicators."""
    if prediction == 0:
        return "Low"

    if probability >= 0.85 or inputs["sc"] > 2.0 or inputs["hemo"] < 10:
        return "High"
    if probability >= 0.6 or inputs["sc"] > 1.5 or inputs["al"] >= 3:
        return "Medium"
    return "Low"


def get_health_suggestion(prediction: int, risk_level: str) -> str:
    """Return health advice based on prediction result."""
    if prediction == 1:
        if risk_level == "High":
            return (
                "High risk detected. Please consult a nephrologist immediately. "
                "Avoid high-sodium foods, monitor blood pressure, and schedule "
                "kidney function tests as soon as possible."
            )
        return (
            "CKD indicators detected. Please consult a healthcare professional "
            "for further evaluation. Maintain hydration and follow a kidney-friendly diet."
        )
    return (
        "No CKD detected. Maintain a healthy lifestyle with balanced nutrition, "
        "regular exercise, adequate hydration, and periodic health checkups."
    )


def parse_form_inputs(form) -> dict | None:
    """Parse and validate form inputs."""
    try:
        inputs = {
            "age": float(form["age"]),
            "bp": float(form["bp"]),
            "sg": float(form["sg"]),
            "al": float(form["al"]),
            "su": float(form["su"]),
            "bgr": float(form["bgr"]),
            "sc": float(form["sc"]),
            "hemo": float(form["hemo"]),
        }
    except (KeyError, ValueError):
        return None

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

    for field, (low, high) in ranges.items():
        if not (low <= inputs[field] <= high):
            return None

    return inputs


def read_prediction_logs(limit: int = 50) -> list[dict]:
    """Read recent prediction logs for monitoring page."""
    ensure_prediction_log()
    logs = []
    with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logs.append(row)
    return list(reversed(logs[-limit:]))


@app.route("/")
def home():
    """Home page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form and processing."""
    model = load_model()

    if request.method == "GET":
        return render_template("predict.html", feature_labels=FEATURE_LABELS)

    if model is None:
        flash("Model not trained yet. Please run train.py first.", "danger")
        return redirect(url_for("predict"))

    inputs = parse_form_inputs(request.form)
    if inputs is None:
        flash("Invalid input values. Please check all fields.", "danger")
        return redirect(url_for("predict"))

    feature_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(feature_df)[0])
    probabilities = model.predict_proba(feature_df)[0]
    probability = float(probabilities[prediction])

    result_label = "CKD Detected" if prediction == 1 else "No CKD Detected"
    risk_level = get_risk_level(prediction, probability, inputs)
    suggestion = get_health_suggestion(prediction, risk_level)

    log_prediction(inputs, result_label, risk_level, probability)

    return render_template(
        "result.html",
        prediction=result_label,
        is_ckd=prediction == 1,
        risk_level=risk_level,
        suggestion=suggestion,
        inputs=inputs,
        feature_labels=FEATURE_LABELS,
        probability=round(probability * 100, 1),
    )


@app.route("/logs")
def logs():
    """Prediction monitoring logs page."""
    prediction_logs = read_prediction_logs()
    return render_template("logs.html", logs=prediction_logs)


@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring."""
    model = load_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "NephroDetect",
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for CKD prediction."""
    model = load_model()
    if model is None:
        return {"error": "Model not trained. Run train.py first."}, 503

    data = request.get_json(silent=True) or {}
    try:
        inputs = {key: float(data[key]) for key in FEATURE_COLUMNS}
    except (KeyError, TypeError, ValueError):
        return {"error": "Invalid input. Required fields: " + ", ".join(FEATURE_COLUMNS)}, 400

    ranges = {
        "age": (1, 120), "bp": (50, 200), "sg": (1.0, 1.05),
        "al": (0, 5), "su": (0, 5), "bgr": (50, 500),
        "sc": (0.1, 20), "hemo": (3, 20),
    }
    for field, (low, high) in ranges.items():
        if not (low <= inputs[field] <= high):
            return {"error": f"{field} out of valid range ({low}-{high})"}, 400

    feature_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(feature_df)[0])
    probabilities = model.predict_proba(feature_df)[0]
    probability = float(probabilities[prediction])
    result_label = "CKD Detected" if prediction == 1 else "No CKD Detected"
    risk_level = get_risk_level(prediction, probability, inputs)

    log_prediction(inputs, result_label, risk_level, probability)

    return {
        "prediction": result_label,
        "is_ckd": prediction == 1,
        "risk_level": risk_level,
        "confidence": round(probability * 100, 1),
        "inputs": inputs,
        "suggestion": get_health_suggestion(prediction, risk_level),
    }


@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrain model when new data is added."""
    try:
        _, metadata = retrain_model()
        flash(
            f"Model retrained successfully! Version: {metadata['version']}, "
            f"Test Accuracy: {metadata['test_accuracy']:.2%}",
            "success",
        )
    except FileNotFoundError:
        flash("Dataset not found in data/ folder. Add dataset before retraining.", "danger")
    except Exception as exc:
        flash(f"Retraining failed: {exc}", "danger")

    return redirect(url_for("home"))


if __name__ == "__main__":
    import os

    ensure_prediction_log()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
