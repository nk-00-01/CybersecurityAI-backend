import pandas as pd
import joblib

from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "prepared" / "training_data.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "ids_model.pkl"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading training dataset...")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Label"])
y = df["Label"]

print("Features:", X.shape)
print("Labels:", y.shape)

print("\nClasses:")
print(y.value_counts())


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

print("\n" + "=" * 60)
print("Creating train/test split...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# Create Random Forest model
# --------------------------------------------------

print("\n" + "=" * 60)
print("Training Random Forest...")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("Evaluating model...")
print("=" * 60)

y_pred = model.predict(X_test)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# --------------------------------------------------
# Save model
# --------------------------------------------------

print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)

joblib.dump(model, MODEL_FILE)

print("Model saved successfully:")
print(MODEL_FILE)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

print("\nTop 15 important features:")

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print(feature_importance.head(15))

print("\nTraining and evaluation completed successfully.")