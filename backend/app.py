from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "ids_model.pkl"


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("Loading IDS model...")

model = joblib.load(MODEL_FILE)

print("IDS model loaded successfully.")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SentinelAI Cyber Threat Detection API",
    description="AI-powered network intrusion detection API",
    version="1.0.0"
)   
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------
# Request model
# --------------------------------------------------

class IDSPredictionRequest(BaseModel):
    features: dict


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "SentinelAI API is running",
        "model": "Random Forest IDS"
    }


# --------------------------------------------------
# IDS prediction
# --------------------------------------------------

@app.post("/api/ids/predict")
def predict_ids(request: IDSPredictionRequest):

    try:

        # Convert received features into DataFrame
        input_data = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Highest probability
        confidence = float(max(probabilities))

        return {
            "attack_type": str(prediction),
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
@app.post("/api/ids/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        contents = await file.read()

        from io import BytesIO
        df = pd.read_csv(BytesIO(contents))

        # Make sure the CSV has the expected label column if present
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        # Keep only the features used by the trained model
        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature for feature in expected_features
            if feature not in df.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features[:10]}"
            )

        # Correct feature order
        df = df[expected_features]

        # Clean invalid values
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.fillna(0)

        # Predict
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        # Count predictions
        prediction_counts = pd.Series(predictions).value_counts()

        results = []

        for attack_type, count in prediction_counts.items():
            indices = [
                i for i, prediction in enumerate(predictions)
                if prediction == attack_type
            ]

            confidence = float(
                max(probabilities[indices[0]])
            )

            results.append({
                "attack_type": str(attack_type),
                "count": int(count),
                "confidence": round(confidence, 4)
            })

        return {
            "filename": file.filename,
            "rows_analyzed": len(df),
            "results": results
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )