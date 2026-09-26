import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "training_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "anomaly_model.pkl"
)

API_URL = (
    "http://127.0.0.1:8001"
    "/api/ids/analyze-anomaly"
)


# --------------------------------------------------
# Load training data and anomaly model
# --------------------------------------------------

print("Loading training data...")

df = pd.read_csv(CSV_FILE)

print(
    f"Training dataset rows: {len(df)}"
)

print("Loading anomaly model...")

model = joblib.load(MODEL_FILE)

expected_features = list(
    model.feature_names_in_
)


# --------------------------------------------------
# Select BENIGN training traffic
# --------------------------------------------------

if "Label" not in df.columns:
    raise ValueError(
        "Expected Label column was not found."
    )

benign_df = df[
    df["Label"].astype(str).str.upper() == "BENIGN"
].copy()

print(
    f"Benign records available: {len(benign_df)}"
)


# --------------------------------------------------
# Create an intentionally extreme outlier
# --------------------------------------------------

features = {}

for feature in expected_features:

    column = pd.to_numeric(
        benign_df[feature],
        errors="coerce"
    )

    column = column.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    column = column.dropna()

    if len(column) == 0:

        features[feature] = 1000000.0

        continue

    minimum = float(column.min())
    maximum = float(column.max())

    data_range = maximum - minimum

    if data_range == 0:

        if maximum == 0:
            features[feature] = 1000000.0
        else:
            features[feature] = (
                maximum * 1000.0
            )

    else:

        features[feature] = (
            maximum + (10.0 * data_range)
        )


# --------------------------------------------------
# Send request
# --------------------------------------------------

payload = {
    "features": features
}

data = json.dumps(payload).encode(
    "utf-8"
)

request = Request(
    API_URL,
    data=data,
    headers={
        "Content-Type": "application/json"
    },
    method="POST"
)


# --------------------------------------------------
# Read response
# --------------------------------------------------

try:

    with urlopen(request) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    print("\nAnomaly Detection Result:")

    print(
        json.dumps(
            result,
            indent=2
        )
    )


except HTTPError as error:

    print(
        "\nAnomaly API test failed."
    )

    print(
        f"HTTP status: {error.code}"
    )

    try:

        body = error.read().decode(
            "utf-8"
        )

        print(
            "Backend response:"
        )

        print(body)

    except Exception:

        print(
            "Unable to read backend error response."
        )


except Exception as error:

    print(
        "\nAnomaly API test failed:"
    )

    print(error)