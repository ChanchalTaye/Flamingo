#  Flamingo AI – IoT Trust & Drift Analytics

Flamingo AI is an intelligent **IoT network monitoring platform** that uses machine learning to detect anomalous device behaviour, compute real-time trust scores, and flag behavioral drift — all through an interactive web dashboard.

---

##  Features

-  **Trust Score Engine** – Scores each IoT device 0–100 based on how closely its traffic matches learned benign behaviour
-  **Risk Level Classification** – Categorises devices as **Low**, **Medium**, or **High** risk
-  **Drift Detection** – Identifies devices whose current behaviour deviates significantly from the baseline
-  **Rule-Based Explainability** – Human-readable explanations for flagged anomalies (e.g. SYN floods, unusual traffic rates)
-  **Interactive Dashboard** – Upload your own CSV or analyse sample data via a clean Streamlit UI
-  **Visual Analytics** – Bar charts for trust score distribution and pie charts for risk level breakdown

---

##  Project Structure

```
Flamingo AI/
├── backend/
│   ├── train.py          # Trains the Isolation Forest model on benign IoT traffic
│   ├── utils.py          # Core ML functions: trust scoring, drift detection, explanations
│   ├── model.pkl         # Trained Isolation Forest model
│   ├── scaler.pkl        # StandardScaler fitted on training data
│   ├── baseline.pkl      # Mean baseline of benign traffic (used for drift detection)
│   └── test_backend.py   # Backend unit tests
├── frontend/
│   └── app.py            # Streamlit web application
├── requirements.txt
└── README.md
```

---

##  How It Works

### 1. Model Training (`backend/train.py`)
- Loads IoT network traffic data (`iot_data.csv`)
- Filters only **BenignTraffic** samples for unsupervised training
- Trains an **Isolation Forest** (contamination = 0.2) to learn normal behaviour
- Saves `model.pkl`, `scaler.pkl`, and `baseline.pkl`

### 2. Trust Score Computation (`backend/utils.py`)
| Function | Description |
|---|---|
| `load_artifacts()` | Loads model, scaler, and baseline from disk |
| `compute_trust_score()` | Normalises Isolation Forest anomaly score into a 0–100 trust score |
| `risk_level()` | Maps trust score → Low / Medium / High |
| `detect_drift()` | Flags features that exceed 2× their baseline mean |
| `generate_explanations()` | Returns human-readable reasons for anomalies |

### 3. Web Dashboard (`frontend/app.py`)
- Built with **Streamlit**
- Upload any compatible IoT CSV or use sample data
- Displays per-device trust scores, risk levels, drift status, and explanations
- Visualises results with bar and pie charts

---

##  Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ChanchalTaye/Flamingo.git
cd Flamingo

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Retrain the Model (optional)
>  Requires `backend/data/iot_data.csv` (not included — see note below)

```bash
cd backend
python train.py
```

### Run the Dashboard

```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`

---

##  Dataset

The model was trained on an IoT network traffic dataset (`iot_data.csv`). This file is **not included** in the repository due to its large size (188 MB).

To use the full dataset, place your `iot_data.csv` inside `backend/data/`.  
The CSV must include a `label` column where benign traffic is labelled `BenignTraffic`.

---

##  Dependencies

Key packages from `requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard UI |
| `scikit-learn` | Isolation Forest, StandardScaler |
| `pandas` | Data handling |
| `numpy` | Numerical operations |
| `joblib` | Model serialisation |
| `matplotlib` | Pie chart visualisation |

---

##  Risk Level Reference

| Trust Score | Risk Level |
|---|---|
| > 70 | 🟢 Low |
| 40 – 70 | 🟡 Medium |
| < 40 | 🔴 High |

---

##  License

This project is open-source. Feel free to fork, modify, and contribute!
