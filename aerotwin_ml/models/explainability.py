"""
AeroTwin - Explainability Layer (SHAP)
=========================================
Answers: "WHY did the AI classify this as a fault?"

Wraps the trained fault classifier with SHAP TreeExplainer so every
prediction can be broken down into which residuals pushed the decision,
turned into a short plain-language explanation for the dashboard.
"""

import numpy as np
import pandas as pd
import shap
from models.fault_classifier import FaultClassifier, FEATURE_COLS

PLAIN_NAMES = {
    "resid_cht_c": "cylinder head temperature",
    "resid_egt_c": "exhaust gas temperature",
    "resid_oil_temp_c": "oil temperature",
    "resid_oil_press_bar": "oil pressure",
    "resid_fuel_flow_lph": "fuel flow",
    "resid_vibration_g": "vibration",
    "resid_battery_v": "battery voltage",
    "throttle": "throttle position",
    "rpm": "engine RPM",
}


class FaultExplainer:
    def __init__(self, classifier: FaultClassifier):
        self.classifier = classifier
        self.explainer = shap.TreeExplainer(classifier.model)

    def explain(self, row: pd.DataFrame, top_k: int = 3) -> dict:
        """row: a single-row DataFrame with FEATURE_COLS. Returns predicted
        label + a short list of the top contributing factors in plain English."""
        X = row[FEATURE_COLS]
        pred = self.classifier.model.predict(X)[0]
        proba = dict(zip(self.classifier.classes_, self.classifier.model.predict_proba(X)[0]))

        shap_values = self.explainer.shap_values(X)
        # shap_values shape for multiclass RF: [n_classes][n_samples, n_features]
        class_idx = list(self.classifier.classes_).index(pred)
        if isinstance(shap_values, list):
            contribs = shap_values[class_idx][0]
        else:
            contribs = shap_values[0, :, class_idx]

        pairs = sorted(zip(FEATURE_COLS, contribs), key=lambda x: -abs(x[1]))[:top_k]
        reasons = []
        for feat, val in pairs:
            direction = "higher than expected" if val > 0 else "lower than expected"
            plain = PLAIN_NAMES.get(feat, feat)
            reasons.append(f"{plain} is {direction} (impact: {val:+.2f})")

        return {
            "predicted_fault": pred,
            "confidence": round(float(proba[pred]), 3),
            "top_reasons": reasons,
            "explanation_text": self._to_sentence(pred, reasons),
        }

    @staticmethod
    def _to_sentence(pred: str, reasons: list) -> str:
        if pred == "none":
            return "No fault detected — all monitored parameters are within expected range."
        reason_str = "; ".join(reasons)
        label = pred.replace("_", " ")
        return f"Classified as '{label}' because: {reason_str}."


if __name__ == "__main__":
    import pandas as pd
    clf = FaultClassifier.load()
    explainer = FaultExplainer(clf)

    df = pd.read_csv("data/aerotwin_dataset.csv")
    sample_fault = df[df["fault_active"] == "overheating_trend"].iloc[[500]]
    print(explainer.explain(sample_fault))

    sample_healthy = df[df["fault_active"] == "none"].iloc[[100]]
    print(explainer.explain(sample_healthy))
