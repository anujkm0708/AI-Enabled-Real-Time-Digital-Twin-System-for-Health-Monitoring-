"""
AeroTwin - Anomaly Detection Model
=====================================
Answers: "Is something unusual happening right now?"

Uses Isolation Forest on the digital-twin residuals. Trained ONLY on
healthy-engine data (unsupervised) so it doesn't need labeled faults --
it just learns what "normal residual patterns" look like and flags
anything that deviates.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RESIDUAL_COLS = ["resid_cht_c", "resid_egt_c", "resid_oil_temp_c", "resid_oil_press_bar",
                  "resid_fuel_flow_lph", "resid_vibration_g", "resid_battery_v"]


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )

    def fit(self, df_healthy: pd.DataFrame):
        X = df_healthy[RESIDUAL_COLS].values
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Returns an anomaly score in [0, 1], higher = more anomalous."""
        X = df[RESIDUAL_COLS].values
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)  # higher = more normal
        # rescale so 0 = very normal, 1 = very anomalous
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return score

    def predict_is_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        X = df[RESIDUAL_COLS].values
        Xs = self.scaler.transform(X)
        pred = self.model.predict(Xs)  # -1 = anomaly, 1 = normal
        return pred == -1

    def save(self, path="models/artifacts/anomaly_detector.joblib"):
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)

    @classmethod
    def load(cls, path="models/artifacts/anomaly_detector.joblib"):
        obj = cls()
        blob = joblib.load(path)
        obj.scaler, obj.model = blob["scaler"], blob["model"]
        return obj


if __name__ == "__main__":
    import os
    os.makedirs("models/artifacts", exist_ok=True)
    df = pd.read_csv("data/aerotwin_dataset.csv")
    df_healthy = df[df["fault_active"] == "none"]

    det = AnomalyDetector(contamination=0.02).fit(df_healthy)
    det.save()

    # quick eval: does it flag faulty rows more than healthy ones?
    df_fault = df[df["fault_active"] != "none"]
    healthy_flag_rate = det.predict_is_anomaly(df_healthy.sample(5000, random_state=1)).mean()
    fault_flag_rate = det.predict_is_anomaly(df_fault.sample(5000, random_state=1)).mean()
    print(f"Anomaly flag rate on HEALTHY samples: {healthy_flag_rate:.3f} (want low)")
    print(f"Anomaly flag rate on FAULTY samples:  {fault_flag_rate:.3f} (want high)")
