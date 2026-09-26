from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "ransomware_model.pkl"


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

np.random.seed(42)


# --------------------------------------------------
# Generate safe simulated behavioural telemetry
# --------------------------------------------------

N_SAMPLES = 2000


def create_dataset():

    rows = []

    for _ in range(N_SAMPLES):

        file_operations = np.random.randint(5, 1000)
        rename_rate = np.random.randint(0, 500)

        entropy_score = np.random.uniform(
            2.0,
            8.0
        )

        extension_change_rate = np.random.uniform(
            0,
            1
        )

        process_count = np.random.randint(
            1,
            30
        )

        suspicious_process = np.random.randint(
            0,
            2
        )

        shadow_copy_activity = np.random.randint(
            0,
            2
        )

        rapid_file_modification = np.random.randint(
            0,
            2
        )

        # ----------------------------------------------
        # Generate safe demonstration labels
        # ----------------------------------------------

        risk_signal = 0

        if file_operations > 500:
            risk_signal += 1

        if rename_rate > 200:
            risk_signal += 1

        if entropy_score > 6.5:
            risk_signal += 1

        if extension_change_rate > 0.60:
            risk_signal += 1

        if suspicious_process == 1:
            risk_signal += 1

        if shadow_copy_activity == 1:
            risk_signal += 1

        if rapid_file_modification == 1:
            risk_signal += 1

        label = (
            "SUSPICIOUS"
            if risk_signal >= 4
            else "BENIGN"
        )

        rows.append({
            "file_operations_per_sec": file_operations,
            "rename_operations_per_sec": rename_rate,
            "entropy_score": entropy_score,
            "extension_change_rate": extension_change_rate,
            "active_process_count": process_count,
            "suspicious_process": suspicious_process,
            "shadow_copy_activity": shadow_copy_activity,
            "rapid_file_modification": rapid_file_modification,
            "label": label
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

print(
    "Generating safe ransomware behaviour telemetry..."
)

df = create_dataset()

print(
    f"Dataset created: {len(df)} records"
)

print("\nClass distribution:")
print(
    df["label"].value_counts()
)


# --------------------------------------------------
# Split features and labels
# --------------------------------------------------

X = df.drop(
    columns=["label"]
)

y = df["label"]


# --------------------------------------------------
# Train / test
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


print(
    "\nTraining ransomware behaviour model..."
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nModel accuracy: {accuracy:.4f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "\nRansomware model saved:"
)

print(MODEL_FILE)