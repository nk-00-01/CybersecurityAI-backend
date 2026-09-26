from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "ids_model.pkl"

print("Loading IDS model...")

model = joblib.load(MODEL_FILE)

print("IDS model loaded successfully.")