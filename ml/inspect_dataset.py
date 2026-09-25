import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATASET_FOLDER = PROJECT_ROOT / "data" / "MachineLearningCVE"

print("Project root:", PROJECT_ROOT)
print("Dataset folder:", DATASET_FOLDER)
print("Dataset folder exists:", DATASET_FOLDER.exists())

csv_files = list(DATASET_FOLDER.glob("*.csv"))

print("CSV files found:", len(csv_files))
print("=" * 60)

for file in csv_files:
    print("\nFile:", file.name)
    print("Size:", round(file.stat().st_size / (1024 * 1024), 2), "MB")

    df = pd.read_csv(file, nrows=5)

    print("Columns:", len(df.columns))
    print("Column names:")
    print(list(df.columns))

    print("\nSample:")
    print(df.head(2))

    print("=" * 60)