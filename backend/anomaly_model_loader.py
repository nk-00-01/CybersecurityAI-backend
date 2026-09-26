from pathlib import Path

import joblib


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "anomaly_model.pkl"
)


# --------------------------------------------------
# Load anomaly model
# --------------------------------------------------

print("Loading anomaly detection model...")

anomaly_model = joblib.load(
    MODEL_FILE
)

# Use a single worker for live single-record
# predictions to avoid unnecessary parallelism
# warnings from scikit-learn.
anomaly_model.set_params(
    n_jobs=1
)

print(
    "Anomaly detection model loaded successfully."
)
print(
    "Anomaly detection model loaded successfully."
)