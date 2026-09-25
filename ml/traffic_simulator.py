import time
import requests
import pandas as pd

CSV_FILE = "data/prepared/test_sample.csv"
API_URL = "http://127.0.0.1:8001/api/ids/predict"

df = pd.read_csv(CSV_FILE)

# Remove label because the model predicts the label
if "Label" in df.columns:
    df = df.drop(columns=["Label"])

print(f"Loaded {len(df)} traffic records.")
print("Starting automatic traffic simulation...\n")

for index, row in df.iterrows():

    features = row.to_dict()

    # Convert invalid values to 0
    features = {
        key: 0 if pd.isna(value) else value
        for key, value in features.items()
    }

    try:
        response = requests.post(
            API_URL,
            json={"features": features}
        )

        if response.status_code == 200:
            result = response.json()

            print(
                f"Record {index + 1}: "
                f"{result['attack_type']} | "
                f"Confidence: {result['confidence'] * 100:.2f}%"
            )
        else:
            print(
                f"Record {index + 1}: API Error "
                f"{response.status_code} - {response.text}"
            )

    except Exception as e:
        print(f"Connection error: {e}")

    # Wait before sending the next record
    time.sleep(2)