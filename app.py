"""
NephroDetect Flask Application
Clinical screening — loads pre-trained model only (no training in app).
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from clinical_analysis import build_clinical_analysis  # noqa: E402
from data_preprocessing import FEATURE_COLUMNS  # noqa: E402

app = Flask(__name__)
app.secret_key = "nephrodetect-clinical-secret-key"

MODEL_PATH = BASE_DIR / "models" / "ckd_model.pkl"
LOGS_DIR = BASE_DIR / "logs"
PREDICTION_LOG = LOGS_DIR / "prediction_logs.csv"
DEFAULT_SG = 1.020

FEATURE_LABELS = {
    "age": "Age (Years)",
    "bp": "Blood Pressure (mmHg)",
    "al": "Albumin",
    "su": "Urine Sugar Level",
    "sugar": "Blood Sugar Level",
    "bgr": "Blood Glucose Random (mg/dL)",
    "sc": "Serum Creatinine (mg/dL)",
    "hemo": "Hemoglobin (g/dL)",
}

FIELD_TOOLTIPS = {
    "age": "Patient age in years. Kidney disease risk increases with age.",
    "bp": "Systolic blood pressure. High BP is a leading cause of kidney damage.",
    "al": "Albumin in urine (0 = Normal). Higher levels suggest kidney filter damage.",
    "su": "Urine sugar indicates glucose leakage into urine and helps detect kidney dysfunction.",
    "sugar": "Blood sugar helps understand diabetes-related kidney risks.",
    "bgr": "Random blood glucose test result in mg/dL.",
    "sc": "Serum creatinine level. Elevated values indicate reduced kidney function.",
    "hemo": "Hemoglobin in blood. Low levels are common in advanced CKD.",
}

LOG_HEADERS = [
    "timestamp", "age", "bp", "sg", "al", "su", "sugar", "bgr", "sc", "hemo",
    "prediction", "risk_level", "probability",
]

RISK_DISPLAY = {"Low": "Low Risk Level", "Medium": "Medium Risk Level", "High": "High Risk Level"}


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def ensure_prediction_log():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTION_LOG.exists():
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)


def log_prediction(inputs, prediction, risk_level, probability):
    ensure_prediction_log()
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        inputs["age"], inputs["bp"], inputs["sg"], inputs["al"],
        int(inputs["su"]), inputs["sugar"], inputs["bgr"], inputs["sc"], inputs["hemo"],
        prediction, risk_level, round(probability, 4),
    ]
    with open(PREDICTION_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def get_risk_level(prediction, probability, inputs):
    if prediction == 0:
        return "Low"
    if probability >= 0.85 or inputs["sc"] > 2.0 or inputs["hemo"] < 10:
        return "High"
    if probability >= 0.6 or inputs["sc"] > 1.5 or inputs["al"] >= 3:
        return "Medium"
    return "Low"


def parse_form_inputs(form) -> dict | None:
    """Parse form — sg is set automatically; su and sugar are separate fields."""
    try:
        age = float(form["age"])
        bp = float(form["bp"])
        al = float(form["al"])
        urine_sugar = int(float(form["su"]))
        blood_sugar = float(form["sugar"])
        bgr = float(form["bgr"])
        sc = float(form["sc"])
        hemo = float(form["hemo"])
    except (KeyError, ValueError, TypeError):
        return None

    ranges = {
        "age": (1, 120), "bp": (50, 200),
        "al": (0, 5), "su": (0, 5), "sugar": (40, 600),
        "bgr": (50, 500), "sc": (0.1, 20), "hemo": (3, 20),
    }
    values = {
        "age": age, "bp": bp, "al": al, "su": urine_sugar,
        "sugar": blood_sugar, "bgr": bgr, "sc": sc, "hemo": hemo,
    }
    for field, (low, high) in ranges.items():
        if not (low <= values[field] <= high):
            return None

    values["sg"] = DEFAULT_SG
    return values


def read_all_logs() -> list[dict]:
    ensure_prediction_log()
    with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_all_logs(rows: list[dict]) -> None:
    ensure_prediction_log()
    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_logs(limit=50):
    all_logs = read_all_logs()
    return list(reversed(all_logs[-limit:]))


def delete_log_by_index(display_index: int) -> bool:
    """Delete a log row by its index in the newest-first display list."""
    all_logs = read_all_logs()
    if not all_logs:
        return False

    displayed = list(reversed(all_logs))
    if display_index < 0 or display_index >= len(displayed):
        return False

    displayed.pop(display_index)
    write_all_logs(list(reversed(displayed)))
    return True


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model = load_model()

    if request.method == "GET":
        return render_template(
            "predict.html",
            feature_labels=FEATURE_LABELS,
            tooltips=FIELD_TOOLTIPS,
        )

    if model is None:
        flash("Screening service is temporarily unavailable. Please try again later.", "danger")
        return redirect(url_for("predict"))

    inputs = parse_form_inputs(request.form)
    if inputs is None:
        flash("Please check all fields and enter valid medical values.", "danger")
        return redirect(url_for("predict"))

    feature_df = pd.DataFrame([{
        "age": inputs["age"],
        "bp": inputs["bp"],
        "sg": inputs["sg"],
        "al": inputs["al"],
        "su": int(inputs["su"]),
        "bgr": inputs["bgr"],
        "sc": inputs["sc"],
        "hemo": inputs["hemo"],
    }], columns=FEATURE_COLUMNS)

    print("MODEL INPUT:", feature_df.to_dict("records"))

    prediction = int(model.predict(feature_df)[0])
    probabilities = model.predict_proba(feature_df)[0]
    probability = float(probabilities[prediction])

    is_ckd = prediction == 1
    result_label = "CKD Detected" if is_ckd else "No CKD Detected"
    risk_level = get_risk_level(prediction, probability, inputs)
    risk_display = RISK_DISPLAY[risk_level]
    analysis = build_clinical_analysis(inputs, is_ckd)

    log_prediction(inputs, result_label, risk_display, probability)

    return render_template(
        "result.html",
        prediction=result_label,
        is_ckd=is_ckd,
        risk_level=risk_display,
        inputs=inputs,
        feature_labels=FEATURE_LABELS,
        probability=round(probability * 100, 1),
        analysis=analysis,
    )


@app.route("/logs")
def logs():
    return render_template("logs.html", logs=read_prediction_logs())


@app.route("/delete-log/<int:index>", methods=["POST"])
def delete_log(index):
    if delete_log_by_index(index):
        flash("Record deleted successfully.", "success")
    else:
        flash("Unable to delete record. Please try again.", "danger")
    return redirect(url_for("logs"))


if __name__ == "__main__":
    import os

    ensure_prediction_log()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
