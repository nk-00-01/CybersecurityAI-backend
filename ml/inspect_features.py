import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_ROOT / "data" / "MachineLearningCVE"

# Use one representative file first
file = DATASET_FOLDER / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Reading:", file.name)
print("=" * 60)

df = pd.read_csv(file, nrows=10000)

print("\nShape:")
print(df.shape)

print("\nColumn names:")
for i, column in enumerate(df.columns):
    print(f"{i}: {column}")

print("\nData types:")
print(df.dtypes.value_counts())

print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nInfinite values:")
numeric_df = df.select_dtypes(include="number")

print(
    "Total infinite values:",
    numeric_df.isin([float("inf"), float("-inf")]).sum().sum()
)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nLabel distribution:")
print(df[" Label"].value_counts())