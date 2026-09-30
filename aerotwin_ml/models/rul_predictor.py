"""
AeroTwin - Remaining Useful Life (RUL) Prediction Model
==========================================================
Answers: "How much longer can this engine safely run?"

Gradient-boosted regression trained to predict remaining operating hours
before the engine's internal `degradation` state reaches end-of-life (1.0),
using current residuals + accumulated wear indicators as features.

Note: this is a simple, fast-to-train regression approach appropriate for
an MVP. An LSTM/sequence model trained on full degradation
trajectories (e.g. validated against NASA C-MAPSS) is the natural upgrade
path noted in the deployment roadmap.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

END_OF_LIFE_DEGRADATION = 1.0
FEATURE_COLS = ["resid_cht_c", "resid_egt_c", "resid_oil_temp_c", "resid_oil_press_bar",
                "resid_vibration_g", "degradation", "throttle", "rpm"]


def add_rul_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a ground-truth RUL (in simulated engine-hours) per row using each
    mission's own degradation trajectory: RUL = time remaining until
    degradation would reach end-of-life at the CURRENT wear rate.
    """
    out = []
    for mid, g in df.groupby("mission_id"):
        g = g.sort_values("t_s").copy()
        wear_rate = g["degradation"].diff().fillna(0).clip(lower=1e-9)
        avg_rate = wear_rate.rolling(60, min_periods=1).mean().replace(0, 1e-9)
        remaining_degradation = (END_OF_LIFE_DEGRADATION - g["degradation"]).clip(lower=0)
        rul_seconds = remaining_degradation / avg_rate
        g["rul_hours"] = (rul_seconds / 3600.0).clip(upper=500)  # cap for sane scale
        out.append(g)
    return pd.concat(out, ignore_index=True)


class RULPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )

    def fit(self, df: pd.DataFrame):
        self.model.fit(df[FEATURE_COLS], df["rul_hours"])
        return self

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df[FEATURE_COLS])

    def save(self, path="models/artifacts/rul_predictor.joblib"):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path="models/artifacts/rul_predictor.joblib"):
        obj = cls()
        obj.model = joblib.load(path)
        return obj


if __name__ == "__main__":
    import os
    os.makedirs("models/artifacts", exist_ok=True)
    df = pd.read_csv("data/aerotwin_dataset.csv")
    df = add_rul_label(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    rul = RULPredictor().fit(train_df)
    rul.save()

    preds = rul.predict(test_df)
    mae = mean_absolute_error(test_df["rul_hours"], preds)
    r2 = r2_score(test_df["rul_hours"], preds)
    print(f"RUL model -> MAE: {mae:.2f} hours, R2: {r2:.3f}")
