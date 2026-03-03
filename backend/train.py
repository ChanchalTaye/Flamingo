import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/iot_data.csv"

def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Separate benign traffic for training
    benign_df = df[df["label"] == "BenignTraffic"].copy()

    print(f"Benign samples used for training: {len(benign_df)}")

    # Drop label column
    X = benign_df.drop(columns=["label"])

    # Keep only numeric columns (extra safety)
    X = X.select_dtypes(include=["float64", "int64"])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.2,
        random_state=42
    )

    print("Training Isolation Forest...")
    model.fit(X_scaled)

    # Save baseline (mean of benign behavior)
    baseline = X.mean().to_dict()

    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(baseline, "baseline.pkl")

    print("✅ Model training complete!")
    print("Artifacts saved: model.pkl, scaler.pkl, baseline.pkl")


if __name__ == "__main__":
    main()