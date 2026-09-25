import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_ROOT / "data" / "MachineLearningCVE"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "prepared"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "training_data.csv"


# --------------------------------------------------
# Settings
# --------------------------------------------------

# Maximum number of records kept for each class
MAX_SAMPLES_PER_CLASS = 10000

# Random seed for reproducible sampling
RANDOM_STATE = 42


# --------------------------------------------------
# Find CSV files
# --------------------------------------------------

csv_files = list(DATASET_FOLDER.glob("*.csv"))

print("CSV files found:", len(csv_files))
print("=" * 60)


# --------------------------------------------------
# Read and combine datasets
# --------------------------------------------------

all_data = []

for file in csv_files:

    print(f"\nReading: {file.name}")

    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean labels
    df["Label"] = df["Label"].astype(str).str.strip()

    # Replace infinity values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows containing missing values
    before = len(df)
    df = df.dropna()
    removed = before - len(df)

    print("Rows after cleaning:", len(df))
    print("Missing/invalid rows removed:", removed)

    # Remove duplicates inside this file
    before = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before - len(df)

    print("Duplicate rows removed:", removed_duplicates)

    all_data.append(df)


# --------------------------------------------------
# Combine all files
# --------------------------------------------------

print("\n" + "=" * 60)
print("Combining datasets...")
print("=" * 60)

combined_df = pd.concat(all_data, ignore_index=True)

print("Combined records:", len(combined_df))


# --------------------------------------------------
# Show original distribution
# --------------------------------------------------

print("\nOriginal label distribution:")

print(
    combined_df["Label"]
    .value_counts()
)


# --------------------------------------------------
# Balance the dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("Creating balanced training dataset...")
print("=" * 60)

balanced_parts = []

for label, group in combined_df.groupby("Label"):

    print(
        f"{label}: "
        f"{len(group):,} available"
    )

    # Keep maximum allowed records
    if len(group) > MAX_SAMPLES_PER_CLASS:

        group = group.sample(
            n=MAX_SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE
        )

    balanced_parts.append(group)


# Combine balanced classes
training_df = pd.concat(
    balanced_parts,
    ignore_index=True
)


# --------------------------------------------------
# Shuffle dataset
# --------------------------------------------------

training_df = training_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TRAINING DATASET")
print("=" * 60)

print("Total records:", len(training_df))
print("Total features:", len(training_df.columns) - 1)

print("\nFinal label distribution:")

print(
    training_df["Label"]
    .value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nTraining data preparation completed successfully.")