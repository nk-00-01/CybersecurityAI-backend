import asyncio
from pathlib import Path

import pandas as pd

from backend.model_loader import model
from backend.anomaly_model_loader import anomaly_model

from backend.response_engine import (
    process_threat,
    process_anomaly_event
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "test_sample.csv"
)


# --------------------------------------------------
# Stream configuration
# --------------------------------------------------

STREAM_DELAY = 1.0


# --------------------------------------------------
# Live stream
# --------------------------------------------------

async def run_live_stream():
    """
    Simulates a live cybersecurity traffic stream
    using recorded CSV data.

    Source IPs are simulated because the prepared
    CICIDS2017 CSV does not contain source-IP metadata.

    Every record is analyzed by:
    1. Random Forest IDS
    2. Isolation Forest anomaly detector

    Known malicious traffic is sent to the normal
    response engine.

    Traffic classified as BENIGN but detected as
    anomalous is treated as a potential unknown
    or anomalous threat and generates an alert.
    """

    # --------------------------------------------------
    # Check CSV
    # --------------------------------------------------

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Live stream CSV not found: {CSV_FILE}"
        )


    # --------------------------------------------------
    # Read CSV
    # --------------------------------------------------

    df = pd.read_csv(CSV_FILE)

    print(
        f"[LIVE] Loaded {len(df)} records from "
        f"{CSV_FILE.name}"
    )


    # --------------------------------------------------
    # Remove label column
    # --------------------------------------------------

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])


    # --------------------------------------------------
    # Get Random Forest feature list
    # --------------------------------------------------

    expected_features = list(
        model.feature_names_in_
    )


    # --------------------------------------------------
    # Verify required features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing IDS model features: "
            f"{missing_features[:10]}"
        )


    # --------------------------------------------------
    # Keep correct feature order
    # --------------------------------------------------

    model_df = df[expected_features].copy()


    # --------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------

    model_df = model_df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    model_df = model_df.fillna(0)


    # --------------------------------------------------
    # Process records one by one
    # --------------------------------------------------

    for index, row in model_df.iterrows():

        try:

            # --------------------------------------------------
            # Create one-row DataFrame
            # --------------------------------------------------

            row_df = pd.DataFrame([row])


            # --------------------------------------------------
            # Random Forest prediction
            # --------------------------------------------------

            prediction = model.predict(
                row_df
            )[0]


            probabilities = model.predict_proba(
                row_df
            )[0]


            confidence = float(
                max(probabilities)
            )


            # --------------------------------------------------
            # Anomaly detection
            # --------------------------------------------------

            anomaly_prediction = anomaly_model.predict(
                row_df
            )[0]


            anomaly_score = float(
                anomaly_model.decision_function(
                    row_df
                )[0]
            )


            anomaly_detected = (
                anomaly_prediction == -1
            )


            # --------------------------------------------------
            # Simulated source IP
            # --------------------------------------------------

            source_ip = (
                f"10.0.0.{(index % 5) + 1}"
            )


            # --------------------------------------------------
            # Final security decision
            # --------------------------------------------------

            if (
                str(prediction).upper() == "BENIGN"
                and anomaly_detected
            ):

                # ----------------------------------------------
                # Potential unknown/anomalous threat
                # ----------------------------------------------

                event = process_anomaly_event(
                    source_ip=source_ip,
                    anomaly_score=anomaly_score
                )


                action = event["action"]


                print(
                    f"[LIVE] "
                    f"{source_ip} | "
                    f"RF={prediction} | "
                    f"Confidence={confidence:.2%} | "
                    f"Anomaly=True | "
                    f"Score={anomaly_score:.4f} | "
                    f"Action={action}"
                )


            else:

                # ----------------------------------------------
                # Known attack OR normal traffic
                # ----------------------------------------------

                event = process_threat(
                    source_ip=source_ip,
                    attack_type=str(prediction),
                    confidence=confidence
                )


                action = event["action"]


                print(
                    f"[LIVE] "
                    f"{source_ip} | "
                    f"RF={prediction} | "
                    f"Confidence={confidence:.2%} | "
                    f"Anomaly={anomaly_detected} | "
                    f"Score={anomaly_score:.4f} | "
                    f"Action={action}"
                )


            # --------------------------------------------------
            # Wait before next record
            # --------------------------------------------------

            await asyncio.sleep(
                STREAM_DELAY
            )


        except asyncio.CancelledError:

            print(
                "[LIVE] Live stream stopped."
            )

            raise


        except Exception as error:

            print(
                f"[LIVE] Error processing record "
                f"{index}: {error}"
            )

            # Continue with next record
            await asyncio.sleep(
                STREAM_DELAY
            )


    # --------------------------------------------------
    # Stream completed
    # --------------------------------------------------

    print(
        "[LIVE] Live stream completed."
    )