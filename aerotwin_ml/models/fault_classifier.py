"""
AeroTwin - Fault Classification Model
========================================
Answers: "Given an anomaly, WHAT specifically is wrong?"

Random Forest classifier trained on labeled fault-injection runs from the
simulator. Distinguishes: sensor_drift, overheating_trend, oil_pressure_loss,
misfire_condition, or none (healthy).
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURE_COLS = ["resid_cht_c", "resid_egt_c", "resid_oil_temp_c", "resid_oil_press_bar",
                "resid_fuel_flow_lph", "resid_vibration_g", "resid_battery_v",
                "throttle", "rpm"]
LABEL_COL = "fault_active"


class FaultClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        self.classes_ = None

    def fit(self, df: pd.DataFrame):
        X = df[FEATURE_COLS]
        y = df[LABEL_COL]
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df[FEATURE_COLS])

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df[FEATURE_COLS])

    def feature_importances(self):
        return dict(zip(FEATURE_COLS, self.model.feature_importances_))

    def save(self, path="models/artifacts/fault_classifier.joblib"):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path="models/artifacts/fault_classifier.joblib"):
        obj = cls()
        obj.model = joblib.load(path)
        obj.classes_ = obj.model.classes_
        return obj


if __name__ == "__main__":
    import os
    os.makedirs("models/artifacts", exist_ok=True)
    df = pd.read_csv("data/aerotwin_dataset.csv")

    # only train on samples where a fault is either fully active or absent
    # (skip the ramp-in ambiguity zone for cleaner training labels)
    df_clean = df[(df["fault_active"] == "none") | (df.get("fault_severity", 0) > 0.6)]

    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42,
                                          stratify=df_clean[LABEL_COL])

    clf = FaultClassifier().fit(train_df)
    clf.save()

    preds = clf.predict(test_df)
    print(classification_report(test_df[LABEL_COL], preds))
    print("\nFeature importances:")
    for feat, imp in sorted(clf.feature_importances().items(), key=lambda x: -x[1]):
        print(f"  {feat:25s} {imp:.3f}")
