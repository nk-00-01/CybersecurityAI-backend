import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_ROOT / "data" / "MachineLearningCVE"

# Use one file first for testing
file = DATASET_FOLDER / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Reading:", file.name)
print("=" * 60)

# Read dataset
df = pd.read_csv(file)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Clean column names
# --------------------------------------------------

df.columns = df.columns.str.strip()

print("\nColumns cleaned.")

# --------------------------------------------------
# 2. Clean labels
# --------------------------------------------------

df["Label"] = df["Label"].astype(str).str.strip()

print("\nLabels:")
print(df["Label"].value_counts())

# --------------------------------------------------
# 3. Replace infinite values
# --------------------------------------------------

df = df.replace([np.inf, -np.inf], np.nan)

print("\nInfinite values replaced with NaN.")

# --------------------------------------------------
# 4. Remove rows containing missing values
# --------------------------------------------------

before_missing = len(df)

df = df.dropna()

after_missing = len(df)

print(
    f"Rows removed because of missing values: "
    f"{before_missing - after_missing}"
)

# --------------------------------------------------
# 5. Remove duplicate rows
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(
    f"Duplicate rows removed: "
    f"{before_duplicates - after_duplicates}"
)

# --------------------------------------------------
# 6. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Label"])
y = df["Label"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# --------------------------------------------------
# 7. Check data types
# --------------------------------------------------

print("\nNon-numeric columns:")

non_numeric = X.select_dtypes(exclude=["number"]).columns

print(list(non_numeric))

# --------------------------------------------------
# 8. Final information
# --------------------------------------------------

print("\nFinal dataset shape:", df.shape)

print("\nFinal label distribution:")
print(y.value_counts())

print("\nPreprocessing check completed successfully.")