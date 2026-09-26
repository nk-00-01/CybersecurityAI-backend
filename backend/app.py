from backend.model_loader import model
from backend.anomaly_model_loader import anomaly_model
from backend.ransomware_model_loader import ransomware_model
from pathlib import Path
import asyncio
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model_loader import model
from backend.live_simulator import run_live_stream

from backend.response_engine import (
    process_threat,
    get_blocked_sources,
    get_alerts,
    get_live_events,
    block_source,
    unblock_source,
    quarantine_host,
    unquarantine_host,
    get_quarantined_hosts,
    process_ransomware_event,
    process_anomaly_event
)

live_task = None
live_running = False

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "ids_model.pkl"


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("Loading IDS model...")



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
@app.post("/api/ids/analyze")
def analyze_security_event(request: IDSPredictionRequest):

    try:

        # Convert features into DataFrame
        input_data = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Calculate confidence
        confidence = float(max(probabilities))

        # Get source IP if provided
        source_ip = request.features.get(
            "Source IP",
            request.features.get("source_ip", "UNKNOWN")
        )

        # Send prediction to response engine
        event = process_threat(
            source_ip=source_ip,
            attack_type=str(prediction),
            confidence=confidence
        )

        return event

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
# --------------------------------------------------
# Security alerts
# --------------------------------------------------

@app.get("/api/alerts")
def get_security_alerts():
    return {
        "count": len(get_alerts()),
        "alerts": get_alerts()
    }


# --------------------------------------------------
# Live security events
# --------------------------------------------------

@app.get("/api/live/events")
def get_security_events():
    return {
        "count": len(get_live_events()),
        "events": get_live_events()
    }


# --------------------------------------------------
# Blocked sources
# --------------------------------------------------

@app.get("/api/blocked-sources")
def get_blocked():
    return {
        "count": len(get_blocked_sources()),
        "blocked_sources": get_blocked_sources()
    }


# --------------------------------------------------
# Manually block a source
# --------------------------------------------------

class SourceRequest(BaseModel):
    source_ip: str


@app.post("/api/block")
def block_security_source(request: SourceRequest):

    block_source(request.source_ip)

    return {
        "source_ip": request.source_ip,
        "status": "BLOCKED"
    }


# --------------------------------------------------
# Manually unblock a source
# --------------------------------------------------

@app.post("/api/unblock")
def unblock_security_source(request: SourceRequest):

    unblock_source(request.source_ip)

    return {
        "source_ip": request.source_ip,
        "status": "UNBLOCKED"
    }
# --------------------------------------------------
# Live stream controls
# --------------------------------------------------

@app.post("/api/live/start")
async def start_live_stream():

    global live_task
    global live_running

    if live_running:
        return {
            "status": "ALREADY_RUNNING"
        }

    live_running = True

    async def stream_wrapper():
        global live_running

        try:
            await run_live_stream()

        finally:
            live_running = False

    live_task = asyncio.create_task(stream_wrapper())

    return {
        "status": "STARTED"
    }


@app.post("/api/live/stop")
async def stop_live_stream():

    global live_task
    global live_running

    if live_task and not live_task.done():
        live_task.cancel()

    live_running = False

    return {
        "status": "STOPPED"
    }


@app.get("/api/live/status")
def live_status():

    return {
        "running": live_running
    }
# --------------------------------------------------
# Ransomware Quarantine
# --------------------------------------------------

class RansomwareHostRequest(BaseModel):
    host_ip: str
    reason: str = "Suspicious ransomware behaviour detected"


@app.post("/api/ransomware/quarantine")
def quarantine_ransomware_host(request: RansomwareHostRequest):

    result = quarantine_host(
        host_ip=request.host_ip,
        reason=request.reason
    )

    return {
        "status": "SUCCESS",
        "host": result["host"],
        "reason": result["reason"],
        "action": result["action"],
        "severity": result["severity"],
        "timestamp": result["timestamp"]
    }


@app.post("/api/ransomware/unquarantine")
def unquarantine_ransomware_host(
    request: RansomwareHostRequest
):

    result = unquarantine_host(
        request.host_ip
    )

    return {
        "status": "SUCCESS",
        "host": result["host"],
        "action": result["action"],
        "was_quarantined": result["was_quarantined"],
        "remaining_quarantined_hosts": result[
            "remaining_quarantined_hosts"
        ],
        "timestamp": result["timestamp"]
    }

@app.get("/api/ransomware/quarantined-hosts")
def get_ransomware_quarantined_hosts():

    hosts = get_quarantined_hosts()

    return {
        "count": len(hosts),
        "quarantined_hosts": hosts
    }
# --------------------------------------------------
# Ransomware Behaviour Response
# --------------------------------------------------

class RansomwareEventRequest(BaseModel):
    host_ip: str
    risk_level: str
    process_name: str
    entropy_score: float


@app.post("/api/ransomware/process")
def process_ransomware(request: RansomwareEventRequest):

    try:

        result = process_ransomware_event(
            host_ip=request.host_ip,
            risk_level=request.risk_level,
            process_name=request.process_name,
            entropy_score=request.entropy_score
        )

        return {
            "status": "SUCCESS",
            **result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
# --------------------------------------------------
# Ransomware ML Analysis
# --------------------------------------------------

class RansomwareAnalysisRequest(BaseModel):
    host_ip: str
    features: dict


@app.post("/api/ransomware/analyze")
def analyze_ransomware(request: RansomwareAnalysisRequest):

    try:

        # Convert features to DataFrame
        input_data = pd.DataFrame([
            request.features
        ])

        # Ensure correct feature order
        expected_features = list(
            ransomware_model.feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:

            raise HTTPException(
                status_code=400,
                detail=f"Missing ransomware features: {missing_features}"
            )

        input_data = input_data[
            expected_features
        ]

        # Clean invalid values
        input_data = input_data.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )

        input_data = input_data.fillna(0)

        # ML prediction
        prediction = ransomware_model.predict(
            input_data
        )[0]

        probabilities = ransomware_model.predict_proba(
            input_data
        )[0]

        confidence = float(
            max(probabilities)
        )

        # --------------------------------------------------
        # Convert ML result into operational risk
        # --------------------------------------------------

        if str(prediction).upper() == "SUSPICIOUS":

            if confidence >= 0.90:
                risk_level = "HIGH"

            else:
                risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # --------------------------------------------------
        # Send result to response engine
        # --------------------------------------------------

        result = process_ransomware_event(
            host_ip=request.host_ip,
            risk_level=risk_level,
            process_name=str(
                request.features.get(
                    "process_name",
                    "Unknown"
                )
            ),
            entropy_score=float(
                request.features.get(
                    "entropy_score",
                    0
                )
            )
        )

        return {
            "prediction": str(prediction),
            "confidence": round(
                confidence,
                4
            ),
            "risk_level": risk_level,
            "response": result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
# --------------------------------------------------
# Potential Unknown / Anomalous Threat Detection
# --------------------------------------------------

class AnomalyAnalysisRequest(BaseModel):
    features: dict


@app.post("/api/ids/analyze-anomaly")
def analyze_anomalous_traffic(
    request: AnomalyAnalysisRequest
):

    try:

        # ----------------------------------------------
        # Convert input to DataFrame
        # ----------------------------------------------

        input_data = pd.DataFrame([
            request.features
        ])


        # ----------------------------------------------
        # Get expected features
        # ----------------------------------------------

        expected_features = list(
            anomaly_model.feature_names_in_
        )


        # ----------------------------------------------
        # Check missing features
        # ----------------------------------------------

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing anomaly model features: "
                    f"{missing_features[:10]}"
                )
            )


        # ----------------------------------------------
        # Correct feature order
        # ----------------------------------------------

        input_data = input_data[
            expected_features
        ]


        # ----------------------------------------------
        # Clean values
        # ----------------------------------------------

        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        input_data = input_data.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )

        input_data = input_data.fillna(0)


        # ----------------------------------------------
        # Anomaly prediction
        # ----------------------------------------------

        prediction = anomaly_model.predict(
            input_data
        )[0]


        anomaly_score = float(
            anomaly_model.decision_function(
                input_data
            )[0]
        )


        is_anomaly = (
            prediction == -1
        )


        if is_anomaly:

            status = "POTENTIAL_UNKNOWN_OR_ANOMALOUS"

            severity = "HIGH"

            anomaly_event = process_anomaly_event(
                source_ip=request.features.get(
                   "Source IP",
                request.features.get(
                "source_ip",
                "UNKNOWN"
            )
        ),
        anomaly_score=anomaly_score
    )

        else:

            status = "NORMAL_PATTERN"

            severity = "LOW"

            anomaly_event = None


        return {
    "anomaly_detected": bool(
        is_anomaly
    ),
    "status": status,
    "severity": severity,
    "anomaly_score": round(
        anomaly_score,
        6
    ),
    "response": anomaly_event
   }


    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
@app.post("/api/ids/test-anomaly")
def test_anomaly():

    import pandas as pd
    import math

    try:
        csv_file = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "prepared"
            / "training_data.csv"
        )

        df = pd.read_csv(csv_file)

        if "Label" not in df.columns:
            raise ValueError("Label column not found.")

        benign_df = df[
            df["Label"].astype(str).str.upper() == "BENIGN"
        ].copy()

        expected_features = list(anomaly_model.feature_names_in_)

        features = {}

        for feature in expected_features:

            column = pd.to_numeric(
                benign_df[feature],
                errors="coerce"
            )

            column = column.replace(
                [float("inf"), float("-inf")],
                pd.NA
            ).dropna()

            if len(column) == 0:
                features[feature] = 1000000.0
                continue

            minimum = float(column.min())
            maximum = float(column.max())

            data_range = maximum - minimum

            if data_range == 0:

                if maximum == 0:
                    features[feature] = 1000000.0
                else:
                    features[feature] = maximum * 1000.0

            else:

                features[feature] = (
                    maximum + (10.0 * data_range)
                )

        result = anomaly_model.predict(
            pd.DataFrame([features])
        )[0]

        score = float(
            anomaly_model.decision_function(
                pd.DataFrame([features])
            )[0]
        )

        anomaly_detected = result == -1

        if anomaly_detected:

            event = process_anomaly_event(
                source_ip="10.0.0.99",
                anomaly_score=score
            )

            return {
                "test": True,
                "anomaly_detected": True,
                "status": "POTENTIAL_UNKNOWN_OR_ANOMALOUS",
                "severity": "HIGH",
                "anomaly_score": round(score, 5),
                "response": event
            }

        return {
            "test": True,
            "anomaly_detected": False,
            "status": "NORMAL_PATTERN",
            "severity": "LOW",
            "anomaly_score": round(score, 5)
        }

    except Exception as error:

        return {
            "test": True,
            "error": str(error)
        }