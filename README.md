# NephroDetect

**Early Prediction of Chronic Kidney Disease using Machine Learning and MLOps**

An intelligent healthcare application that predicts Chronic Kidney Disease (CKD) at an early stage using patient clinical data and follows the complete MLOps lifecycle.

---

## Project Overview

| Component | Technology |
|-----------|------------|
| ML Model | Random Forest Classifier |
| Backend | Flask |
| Frontend | HTML, CSS, Bootstrap 5 |
| Monitoring | CSV prediction logs |
| Experiment Tracking | Manual logging (`logs/experiment_log.txt`) |

---

## Project Structure

```
NephroDetect/
├── data/                    # Raw dataset (kidney_disease.csv)
│   └── processed/           # Train/test splits after preprocessing
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── download_data.py
├── models/
│   └── ckd_model.pkl        # Saved model (after training)
├── logs/
│   ├── experiment_log.txt   # Manual experiment logging
│   └── prediction_logs.csv  # Prediction monitoring
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── logs.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
└── README.md
```

---

## MLOps Workflow

```
Data Collection → Preprocessing → Training → Evaluation → Deployment → Monitoring → Retraining
```

1. **Data Collection** — Download Kaggle CKD dataset into `data/`
2. **Preprocessing** — Handle missing values, encode labels, feature selection
3. **Training** — Train Random Forest, save to `models/ckd_model.pkl`
4. **Evaluation** — Accuracy, Precision, Recall, F1-score
5. **Experiment Logging** — Log version, metrics, date, parameters
6. **Deployment** — Flask web app with prediction form
7. **Monitoring** — Log every prediction to `logs/prediction_logs.csv`
8. **Retraining** — Retrain when new data is added (via UI or CLI)

---

## Setup Instructions

### Step 1: Navigate to Project

```bash
cd NephroDetect
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

**Option A — Kaggle (Recommended)**

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create New API Token → saves `kaggle.json`
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
4. Run:

```bash
python src/download_data.py
```

**Option B — Manual Download**

1. Download from [Kaggle CKD Dataset](https://www.kaggle.com/datasets/mansoordatascience/ckd-prediction)
2. Save CSV as `data/kidney_disease.csv`

**Option C — Sample Data (Testing Only)**

```bash
python src/download_data.py --sample
```

### Step 5: Preprocess Data

```bash
python src/data_preprocessing.py
```

### Step 6: Train Model

```bash
python src/train.py
```

### Step 7: Evaluate Model

```bash
python src/evaluate.py
```

### Step 8: Run Flask App

```bash
python app.py
```

Open browser: **http://127.0.0.1:5000**

---

## Features Used

| Feature | Description |
|---------|-------------|
| age | Age (years) |
| bp | Blood Pressure (mm Hg) |
| sg | Specific Gravity |
| al | Albumin (0-5) |
| su | Sugar (0-5) |
| bgr | Blood Glucose Random (mg/dL) |
| sc | Serum Creatinine (mg/dL) |
| hemo | Hemoglobin (g/dL) |

**Output:** CKD / Not CKD

---

## Web Application Pages

| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | Landing page with project info |
| Predict | `/predict` | Patient data input form |
| Result | `/predict` (POST) | Prediction result with risk level |
| Logs | `/logs` | Monitoring dashboard |

---

## Retraining

When new patient data is added to `data/kidney_disease.csv`:

- Click **Retrain Model** on the home page, or
- Run: `python src/train.py`

Each retraining run appends a new entry to `logs/experiment_log.txt`.

---

## Experiment Log Format

```json
{"version": "v1.0", "accuracy": 0.95, "date": "2026-06-20 14:30:00", "parameters": {"n_estimators": 100, "max_depth": 10}}
```

---

## Disclaimer

This application is built for **educational and academic purposes** (college project, viva, portfolio). It is **not** a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

---

## Author

MLOps Healthcare Project — Nephrology Field
