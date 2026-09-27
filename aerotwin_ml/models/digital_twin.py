"""
AeroTwin - Digital Twin Expected-Value Model
==============================================
This is the "healthy engine" reference model. Given the current operating
point (throttle, altitude, ambient temp, flight phase), it independently
computes what each sensor SHOULD read if the engine were healthy.

This is intentionally a SEPARATE, simplified physics model from the
simulator (not just a copy of it) -- in a real deployment the twin doesn't
have privileged access to the "true" engine internals, only to commanded
inputs and ambient conditions, same as the real engine would provide.
Small modeling differences between twin and simulator are expected and
are exactly what produce the baseline noise floor for residuals.
"""

import numpy as np
from simulator.engine_simulator import REF


def expected_values(throttle: float, altitude_ft: float, ambient_temp_c: float,
                     rpm_actual: float) -> dict:
    """Return the digital twin's independent estimate of healthy-engine values
    for the current operating point. rpm_actual is taken as a trusted input
    (RPM sensors are highly reliable / redundant in real engines) and used to
    derive expectations for the other parameters, mirroring how a real digital
    twin conditions its estimate on the most trustworthy available signal."""

    alt_factor = 1.0 + (altitude_ft / 25000.0) * 0.15
    air_density_factor = max(0.55, 1.0 - altitude_ft / 40000.0)
    rpm_frac = np.clip((rpm_actual - REF["rpm_idle"]) / (REF["rpm_max"] - REF["rpm_idle"]), 0, 1)

    exp_cht = ambient_temp_c + REF["cht_ambient_offset"] + throttle * REF["cht_max_gain"] * alt_factor
    exp_oil_temp = ambient_temp_c + REF["oil_temp_base"] * 0.5 + throttle * REF["oil_temp_max_gain"]
    exp_egt = REF["egt_base"] + throttle * REF["egt_max_gain"] * air_density_factor
    exp_oil_press = (REF["oil_press_idle"] + rpm_frac * (REF["oil_press_cruise"] - REF["oil_press_idle"])
                      - max(0.0, (exp_oil_temp - 90) * 0.01))
    exp_fuel_flow = (REF["fuel_flow_idle"] + throttle * (REF["fuel_flow_max"] - REF["fuel_flow_idle"])) \
        * (0.9 + 0.1 * air_density_factor)
    exp_vibration = REF["vibration_base"] + throttle * REF["vibration_max_gain"] * 0.4
    exp_battery_v = REF["battery_v_base"] + rpm_frac * 0.6

    return {
        "cht_c": exp_cht,
        "egt_c": exp_egt,
        "oil_temp_c": exp_oil_temp,
        "oil_press_bar": exp_oil_press,
        "fuel_flow_lph": exp_fuel_flow,
        "vibration_g": exp_vibration,
        "battery_v": exp_battery_v,
    }


RESIDUAL_FIELDS = ["cht_c", "egt_c", "oil_temp_c", "oil_press_bar",
                    "fuel_flow_lph", "vibration_g", "battery_v"]


def compute_residuals(actual_sample: dict) -> dict:
    """Given one telemetry sample (as produced by EngineSimulator.step),
    return {field: residual} where residual = actual - expected."""
    exp = expected_values(
        throttle=actual_sample["throttle"],
        altitude_ft=actual_sample["altitude_ft"],
        ambient_temp_c=actual_sample["ambient_temp_c"],
        rpm_actual=actual_sample["rpm"],
    )
    return {f: actual_sample[f] - exp[f] for f in RESIDUAL_FIELDS}


def corroboration_check(residuals: dict, thresholds: dict = None) -> dict:
    """
    Implements the 'single sensor spike vs real fault' logic described in the
    problem statement: if only ONE residual is abnormal while thermally/
    mechanically correlated residuals stay normal, classify as likely
    instrumentation fault and keep severity at advisory level rather than
    a hard alert.
    """
    thresholds = thresholds or {
        "cht_c": 25, "egt_c": 40, "oil_temp_c": 15, "oil_press_bar": 0.5,
        "fuel_flow_lph": 3.0, "vibration_g": 0.5, "battery_v": 0.8,
    }
    flagged = {f: abs(residuals[f]) > t for f, t in thresholds.items()}
    n_flagged = sum(flagged.values())

    # thermally correlated group: if CHT spikes alone but oil_temp/vibration
    # stay clean, that's a strong instrumentation-fault signature
    thermal_group_flagged = sum(flagged[f] for f in ["cht_c", "oil_temp_c", "vibration_g"])

    if flagged.get("cht_c") and thermal_group_flagged == 1:
        verdict = "likely_instrumentation_fault"
        severity_level = "advisory"
    elif n_flagged == 0:
        verdict = "nominal"
        severity_level = "none"
    elif n_flagged == 1:
        verdict = "single_sensor_anomaly"
        severity_level = "advisory"
    else:
        verdict = "corroborated_anomaly"
        severity_level = "warning" if n_flagged <= 3 else "critical"

    return {
        "flagged_sensors": [f for f, v in flagged.items() if v],
        "n_flagged": n_flagged,
        "verdict": verdict,
        "severity_level": severity_level,
    }
