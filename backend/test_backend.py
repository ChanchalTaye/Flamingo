import pandas as pd
from utils import (
    load_artifacts,
    compute_trust_score,
    risk_level,
    detect_drift,
    generate_explanations
)

# ----------------------------
# Load artifacts
# ----------------------------
model, scaler, baseline = load_artifacts()

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/iot_data.csv")

# Select one ATTACK sample for testing
test_sample = df[df["label"] != "BenignTraffic"].iloc[0]

# Extract feature columns (exclude label)
feature_columns = df.drop(columns=["label"]).columns

# Prepare data row
data_row = test_sample.drop("label").values

# ----------------------------
# Compute Trust Score
# ----------------------------
trust_score, anomaly_score = compute_trust_score(
    model,
    scaler,
    data_row,
    feature_columns
)

# ----------------------------
# Risk Level
# ----------------------------
risk = risk_level(trust_score)

# ----------------------------
# Drift Detection
# ----------------------------
drift, drift_features = detect_drift(
    data_row,
    baseline,
    feature_columns
)

# ----------------------------
# Generate Explanations
# ----------------------------
explanations = generate_explanations(
    data_row,
    anomaly_score,
    feature_columns
)

# ----------------------------
# Print Results
# ----------------------------
print("\n========== Flamingo AI Backend Test ==========")
print("Attack Type:", test_sample["label"])
print("Trust Score:", round(trust_score, 2))
print("Risk Level:", risk)
print("Drift Detected:", drift)
print("Drift Features:", drift_features)
print("Explanations:", explanations)