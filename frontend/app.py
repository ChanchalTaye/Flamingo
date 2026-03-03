import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# ----------------------------
# Proper Backend Import Setup
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(BASE_DIR, "backend")
sys.path.append(BACKEND_PATH)

from utils import (
    load_artifacts,
    compute_trust_score,
    risk_level,
    detect_drift,
    generate_explanations
)

st.set_page_config(page_title="Flamingo AI", layout="wide")

st.title("Flamingo AI – IoT Trust & Drift Analytics")

# ----------------------------
# Load ML Artifacts
# ----------------------------
try:
    model, scaler, baseline = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ----------------------------
# File Upload Section
# ----------------------------
uploaded_file = st.file_uploader("Upload IoT CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")

        # Performance protection
        if len(df) > 1000:
            df = df.sample(500, random_state=42)
            st.warning("Large file detected. Using 500 sampled rows for analysis.")

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

else:
    st.info("No file uploaded. Using sample from trained dataset.")
    data_path = os.path.join(BASE_DIR, "backend", "data", "iot_data.csv")

    try:
        df = pd.read_csv(data_path).sample(50, random_state=42)
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        st.stop()

# ----------------------------
# Feature Handling
# ----------------------------
if "label" in df.columns:
    feature_columns = df.drop(columns=["label"]).columns
else:
    feature_columns = df.columns

if len(feature_columns) == 0:
    st.error("No valid feature columns found.")
    st.stop()

results = []

# ----------------------------
# Process Each Row
# ----------------------------
for index, row in df.iterrows():

    if "label" in df.columns:
        data_row = row.drop("label").values
    else:
        data_row = row.values

    try:
        trust_score, anomaly_score = compute_trust_score(
            model,
            scaler,
            data_row,
            feature_columns
        )

        risk = risk_level(trust_score)

        drift, drift_features = detect_drift(
            data_row,
            baseline,
            feature_columns
        )

        explanations = generate_explanations(
            data_row,
            anomaly_score,
            feature_columns
        )

        results.append({
            "Device_ID": index,
            "Trust_Score": round(trust_score, 2),
            "Risk_Level": risk,
            "Drift": drift,
            "Explanation": ", ".join(explanations) if explanations else "No major anomaly detected"
        })

    except Exception as e:
        continue  # skip problematic rows safely

result_df = pd.DataFrame(results)

if result_df.empty:
    st.error("No results generated. Check dataset format.")
    st.stop()

# ----------------------------
# Display Results
# ----------------------------
st.subheader("Device Trust Overview")
st.dataframe(result_df)

# ----------------------------
# Bar Chart
# ----------------------------
st.subheader("Trust Score Distribution")
st.bar_chart(result_df["Trust_Score"])

# ----------------------------
# Pie Chart
# ----------------------------
st.subheader("Risk Level Distribution")
risk_counts = result_df["Risk_Level"].value_counts()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%")
st.pyplot(fig)