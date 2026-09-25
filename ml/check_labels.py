import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_ROOT / "data" / "MachineLearningCVE"

csv_files = list(DATASET_FOLDER.glob("*.csv"))

print("CSV files found:", len(csv_files))
print("=" * 60)

all_labels = []

for file in csv_files:
    print(f"\nFile: {file.name}")

    df = pd.read_csv(file, usecols=[" Label"])

    df[" Label"] = df[" Label"].astype(str).str.strip()

    labels = df[" Label"].value_counts()

    print("Total records:", len(df))
    print("Labels:")

    for label, count in labels.items():
        print(f"  {label}: {count:,}")

    all_labels.append(df[" Label"])

combined_labels = pd.concat(all_labels, ignore_index=True)

print("\n" + "=" * 60)
print("OVERALL LABEL DISTRIBUTION")
print("=" * 60)

total_counts = combined_labels.value_counts()

for label, count in total_counts.items():
    print(f"{label}: {count:,}")

print("\nTotal records:", len(combined_labels))
print("Total unique labels:", combined_labels.nunique())