from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "training_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = (
    MODEL_DIR
    / "anomaly_model.pkl"
)


# --------------------------------------------------
# Load prepared dataset
# --------------------------------------------------

print("Loading prepared network traffic dataset...")

df = pd.read_csv(DATA_FILE)

print(
    f"Dataset loaded: {len(df)} records"
)


# --------------------------------------------------
# Keep only BENIGN traffic
# --------------------------------------------------

if "Label" not in df.columns:
    raise ValueError(
        "Expected 'Label' column was not found."
    )

benign_df = df[
    df["Label"].astype(str).str.upper() == "BENIGN"
].copy()

print(
    f"Benign records available: {len(benign_df)}"
)


# --------------------------------------------------
# Remove label
# --------------------------------------------------

X = benign_df.drop(
    columns=["Label"]
)


# --------------------------------------------------
# Convert to numeric
# --------------------------------------------------

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

X = X.fillna(0)


# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

print(
    "Training anomaly detection model..."
)

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

model.fit(X)


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "Anomaly model saved successfully:"
)

print(MODEL_FILE)