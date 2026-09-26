from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "ransomware_model.pkl"
)


print("Loading ransomware behaviour model...")

ransomware_model = joblib.load(
    MODEL_FILE
)

print(
    "Ransomware behaviour model loaded successfully."
)