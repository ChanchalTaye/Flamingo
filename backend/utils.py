import os
import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Load model artifacts
# ----------------------------
def load_artifacts():
    """
    Load model, scaler, and baseline from backend directory
    regardless of where the script is executed from.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    baseline_path = os.path.join(BASE_DIR, "baseline.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    baseline = joblib.load(baseline_path)

    return model, scaler, baseline


# ----------------------------
# Trust Score Calculation
# ----------------------------
def compute_trust_score(model, scaler, data_row, feature_names):
    """
    Computes trust score (0-100) based on Isolation Forest anomaly score.
    Higher anomaly -> lower trust.
    """

    # Convert to DataFrame to preserve feature names
    data_df = pd.DataFrame([data_row], columns=feature_names)

    # Scale data
    data_scaled = scaler.transform(data_df)

    # Get anomaly score
    anomaly_score = model.decision_function(data_scaled)[0]

    # Normalize anomaly score into 0–100 range
    min_score = -0.5
    max_score = 0.5

    normalized = (anomaly_score - min_score) / (max_score - min_score)
    normalized = np.clip(normalized, 0, 1)

    trust_score = normalized * 100

    return trust_score, anomaly_score


# ----------------------------
# Risk Level Classification
# ----------------------------
def risk_level(trust_score):
    if trust_score > 70:
        return "Low"
    elif trust_score >= 40:
        return "Medium"
    else:
        return "High"


# ----------------------------
# Drift Detection
# ----------------------------
def detect_drift(data_row, baseline, feature_names):
    """
    Detects behavioral drift compared to baseline.
    If value > 2x baseline -> drift.
    """

    drifted_features = []
    row_dict = dict(zip(feature_names, data_row))

    for feature in feature_names:
        baseline_value = baseline.get(feature)
        current_value = row_dict.get(feature)

        if baseline_value is not None and current_value is not None:
            if current_value > baseline_value * 2:
                drifted_features.append(feature)

    drift_detected = len(drifted_features) > 0

    return drift_detected, drifted_features


# ----------------------------
# Explainability Engine
# ----------------------------
def generate_explanations(data_row, anomaly_score, feature_names):
    """
    Rule-based explanation layer.
    """

    explanations = []
    row_dict = dict(zip(feature_names, data_row))

    if row_dict.get("Rate", 0) > 1000:
        explanations.append("Unusual traffic rate detected.")

    if row_dict.get("Srate", 0) > 1000:
        explanations.append("High source transmission rate observed.")

    if row_dict.get("syn_flag_number", 0) > 50:
        explanations.append("Excessive SYN flag activity (possible SYN flood).")

    if anomaly_score < 0:
        explanations.append("Behavior deviates from learned baseline.")

    return explanations