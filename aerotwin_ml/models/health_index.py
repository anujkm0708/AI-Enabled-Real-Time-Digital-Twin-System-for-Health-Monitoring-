"""
AeroTwin - Composite Health Index
====================================
Computes 5 subsystem health scores (0-100) and one overall composite score
from the current residuals, anomaly score, fault label, and fault confidence.

Each subsystem maps to the sensor channels that most directly reflect its
health, with a penalty applied when a fault label matching that subsystem
is present and confidence is high. Scores are then bucketed into status bands.

Subsystems:
  thermal               - CHT + EGT (cylinder and exhaust heat)
  lubrication           - oil_temp + oil_press (oil system integrity)
  combustion            - EGT + fuel_flow + RPM (fuel-air combustion quality)
  mechanical_vibration  - vibration (structural/mechanical integrity)
  electrical            - battery_v (alternator/electrical health)
"""

import numpy as np
from typing import Optional

# Status band thresholds (score -> band)
BANDS = [
    (85, "excellent"),
    (70, "good"),
    (50, "degraded"),
    (30, "poor"),
    (0,  "critical"),
]

# Which fault labels degrade which subsystem (label -> list of subsystems penalised)
FAULT_SUBSYSTEM_MAP = {
    "overheating_trend":      ["thermal", "lubrication"],
    "cooling_degradation":    ["thermal"],
    "oil_pressure_loss":      ["lubrication"],
    "lubrication_degradation":["lubrication"],
    "misfire_condition":      ["combustion", "mechanical_vibration"],
    "injector_abnormality":   ["combustion"],
    "combustion_instability": ["combustion", "mechanical_vibration"],
    "sensor_drift":           [],   # single-sensor glitch, no subsystem penalty
    "none":                   [],
}

# Residual thresholds for each subsystem sensor channel
# (residual_key, warn_threshold, critical_threshold)
THERMAL_CHANNELS = [
    ("cht_c",   25.0,  50.0),
    ("egt_c",   40.0,  80.0),
]
LUBRICATION_CHANNELS = [
    ("oil_temp_c",    15.0,  30.0),
    ("oil_press_bar",  0.5,   1.2),
]
COMBUSTION_CHANNELS = [
    ("egt_c",          40.0,  80.0),
    ("fuel_flow_lph",   3.0,   6.0),
]
VIBRATION_CHANNELS = [
    ("vibration_g",    0.5,   1.2),
]
ELECTRICAL_CHANNELS = [
    ("battery_v",      0.8,   1.6),
]


def _band(score: float) -> str:
    for threshold, name in BANDS:
        if score >= threshold:
            return name
    return "critical"


def _channel_penalty(residuals: dict, channels: list) -> float:
    """Return 0-100 penalty from residual magnitudes across channels."""
    penalties = []
    for key, warn, crit in channels:
        r = abs(residuals.get(key, 0.0))
        if r >= crit:
            penalties.append(40.0)
        elif r >= warn:
            # linear between warn and crit -> 5 to 40
            penalties.append(5.0 + 35.0 * (r - warn) / (crit - warn))
        else:
            penalties.append(0.0)
    return max(penalties) if penalties else 0.0


def _fault_penalty(subsystem: str, fault_label: str, fault_confidence: float) -> float:
    """Extra penalty when a fault directly matching this subsystem is confirmed."""
    affected = FAULT_SUBSYSTEM_MAP.get(fault_label, [])
    if subsystem in affected:
        return 25.0 * fault_confidence   # up to 25 extra points off at full confidence
    return 0.0


def _anomaly_base_penalty(anomaly_score: float) -> float:
    """Small global penalty shared across all subsystems from anomaly score."""
    return anomaly_score * 10.0   # max -10 at anomaly_score=1


def compute_health_indices(
    residuals: dict,
    anomaly_score: float,
    fault_label: str,
    fault_confidence: float,
) -> dict:
    """
    Compute composite health indices from residuals + AI outputs.

    Parameters
    ----------
    residuals        : dict from compute_residuals() — keys like 'cht_c', 'oil_press_bar' etc.
    anomaly_score    : float [0,1] from AnomalyDetector (higher = more anomalous)
    fault_label      : str — predicted fault class (e.g. 'overheating_trend' or 'none')
    fault_confidence : float [0,1] — classifier confidence for fault_label

    Returns
    -------
    dict with keys:
        thermal, lubrication, combustion, mechanical_vibration, electrical
          -> each: {"score": float, "status": str}
        overall_score: float
        overall_status: str
    """
    base_penalty = _anomaly_base_penalty(anomaly_score)

    def score_for(channels, subsystem_name):
        raw = (100.0
               - _channel_penalty(residuals, channels)
               - _fault_penalty(subsystem_name, fault_label, fault_confidence)
               - base_penalty)
        return float(np.clip(raw, 0.0, 100.0))

    thermal_score      = score_for(THERMAL_CHANNELS,      "thermal")
    lubrication_score  = score_for(LUBRICATION_CHANNELS,  "lubrication")
    combustion_score   = score_for(COMBUSTION_CHANNELS,   "combustion")
    vibration_score    = score_for(VIBRATION_CHANNELS,    "mechanical_vibration")
    electrical_score   = score_for(ELECTRICAL_CHANNELS,   "electrical")

    # Weighted overall: thermal and lubrication are most safety-critical
    weights = {
        "thermal": 0.30,
        "lubrication": 0.25,
        "combustion": 0.25,
        "mechanical_vibration": 0.15,
        "electrical": 0.05,
    }
    overall = (
        weights["thermal"]             * thermal_score +
        weights["lubrication"]         * lubrication_score +
        weights["combustion"]          * combustion_score +
        weights["mechanical_vibration"]* vibration_score +
        weights["electrical"]          * electrical_score
    )
    overall = float(np.clip(overall, 0.0, 100.0))

    return {
        "thermal":              {"score": round(thermal_score, 1),     "status": _band(thermal_score)},
        "lubrication":          {"score": round(lubrication_score, 1), "status": _band(lubrication_score)},
        "combustion":           {"score": round(combustion_score, 1),  "status": _band(combustion_score)},
        "mechanical_vibration": {"score": round(vibration_score, 1),   "status": _band(vibration_score)},
        "electrical":           {"score": round(electrical_score, 1),  "status": _band(electrical_score)},
        "overall_score":        round(overall, 1),
        "overall_status":       _band(overall),
    }


if __name__ == "__main__":
    # Quick smoke test
    residuals = {"cht_c": 55.0, "egt_c": 5.0, "oil_temp_c": 30.0,
                 "oil_press_bar": -1.0, "fuel_flow_lph": 2.0,
                 "vibration_g": 0.3, "battery_v": 0.1}
    result = compute_health_indices(residuals, anomaly_score=0.7,
                                    fault_label="overheating_trend", fault_confidence=0.95)
    for k, v in result.items():
        print(f"  {k}: {v}")
