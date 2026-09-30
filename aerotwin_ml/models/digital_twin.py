"""
AeroTwin - Digital Twin Expected-Value Model (v2)
==================================================
This is the "healthy engine" reference model. Given the current operating
point (throttle, altitude, ambient temp, flight phase), it independently
computes what each sensor SHOULD read if the engine were healthy.

This module is intentionally SEPARATE from the simulator — it does NOT
use simulator code at runtime. Small modelling differences between twin and
simulator produce the baseline noise floor for residuals.

Changes in v2
-------------
Problem 1 - Params calibrated via least-squares (calibrate_params), not copied from REF.
            CHT and oil-temp use a heat-balance ODE form.
Problem 2 - Stateful DigitalTwin class with exact discretisation (exp decay).
Problem 3 - EngineStateEstimator: per-channel 2-state Kalman filter.
Problem 4 - validate_inputs: range/rate/consistency checks; rpm substitution.
Problem 5 - calibrate_thresholds runs healthy missions; Corroborator persistence filter.
"""

import os
import json
import math
import warnings
import numpy as np
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Paths to calibrated artifacts
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS_DIR = os.path.join(_HERE, "artifacts")
_TWIN_PARAMS_PATH = os.path.join(_ARTIFACTS_DIR, "twin_params.json")
_TWIN_THRESHOLDS_PATH = os.path.join(_ARTIFACTS_DIR, "twin_thresholds.json")


# ---------------------------------------------------------------------------
# PROBLEM 1: Calibrated twin parameters (not REF copies)
# ---------------------------------------------------------------------------

def _default_twin_params() -> dict:
    """Physics-motivated initial guesses for the optimiser."""
    return {
        # heat-balance form: C*dT/dt = k_q*throttle*alt_factor - h*(T - T_amb)
        "cht_kq": 90.0,       # heat gain coefficient (°C)
        "cht_h": 1.0,         # heat loss coefficient (°C/°C)
        "cht_C": 1.0,         # thermal capacity (normalised, absorbs tau)
        "cht_ss_offset": 22.0,# steady-state ambient offset (°C)
        # oil temp heat balance
        "oil_kq": 50.0,
        "oil_h": 1.0,
        "oil_C": 1.0,
        "oil_ss_offset": 20.0,
        # EGT (quasi-static)
        "egt_base": 348.0,
        "egt_max_gain": 460.0,
        # oil pressure
        "oil_press_idle": 2.1,
        "oil_press_range": 2.0,
        "oil_press_temp_coef": 0.012,
        "oil_press_temp_ref": 90.0,
        # fuel flow
        "fuel_flow_idle": 3.9,
        "fuel_flow_max": 27.5,
        # vibration
        "vibration_base": 0.78,
        "vibration_gain": 0.85,
        # battery
        "battery_v_base": 13.8,
        "battery_v_range": 0.6,
        # RPM (linear mapping, not the sim curve)
        "rpm_idle": 2180.0,
        "rpm_range": 3600.0,
        # time constants (seconds) — to be estimated, NOT hardcoded
        "tau_cht": 22.0,
        "tau_oil": 40.0,
        "tau_rpm": 2.8,
    }


def calibrate_params(n_missions: int = 15, seed: int = 0) -> dict:
    """
    Fit TWIN_PARAMS by minimising the squared residual between the DigitalTwin
    steady-state predictions and n_missions of HEALTHY simulator runs.

    We use scipy.optimize.least_squares to find the optimal parameters.
    The time constants (tau_cht, tau_oil, tau_rpm) are jointly estimated.
    """
    from scipy.optimize import least_squares
    # Import here so module-level code does not depend on simulator
    import sys
    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj not in sys.path:
        sys.path.insert(0, _proj)
    import importlib
    _sim_mod = importlib.import_module("simulator.engine_simulator")
    EngineSimulator = _sim_mod.EngineSimulator
    FlightPhase = _sim_mod.FlightPhase
    PHASE_PROFILE = _sim_mod.PHASE_PROFILE

    MISSION_PHASES = [
        (FlightPhase.GROUND, 60), (FlightPhase.TAKEOFF, 30),
        (FlightPhase.CLIMB, 300), (FlightPhase.CRUISE, 900),
        (FlightPhase.DESCENT, 200), (FlightPhase.LANDING, 60),
    ]

    rng = np.random.default_rng(seed)

    def collect_data(params_dict):
        """Run missions and return (X_inputs, Y_outputs) numpy arrays."""
        X, Y = [], []
        for _ in range(n_missions):
            amb = rng.uniform(-5, 40)
            alt = rng.uniform(5000, 20000)
            sim = EngineSimulator(
                ambient_temp_c=amb, altitude_ft=alt, dt_s=1.0,
                seed=int(rng.integers(0, 1_000_000))
            )
            for phase, duration in MISSION_PHASES:
                for _ in range(duration):
                    throttle = float(np.clip(
                        PHASE_PROFILE[phase]["throttle"] + rng.normal(0, 0.04),
                        0.05, 1.0
                    ))
                    s = sim.step(phase, throttle_override=throttle)
                    X.append([s["throttle"], s["altitude_ft"], s["ambient_temp_c"]])
                    Y.append([s["cht_c"], s["oil_temp_c"], s["egt_c"],
                              s["oil_press_bar"], s["fuel_flow_lph"],
                              s["vibration_g"], s["battery_v"], s["rpm"]])
        return np.array(X), np.array(Y)

    print("calibrate_params: collecting simulator data …")
    X, Y = collect_data(None)

    # Initial guess vector (only the "shape" parameters — taus handled separately)
    p0_dict = _default_twin_params()

    def dict_to_vec(d):
        keys = [
            "cht_kq", "cht_h", "cht_ss_offset",
            "oil_kq", "oil_h", "oil_ss_offset",
            "egt_base", "egt_max_gain",
            "oil_press_idle", "oil_press_range", "oil_press_temp_coef", "oil_press_temp_ref",
            "fuel_flow_idle", "fuel_flow_max",
            "vibration_base", "vibration_gain",
            "battery_v_base", "battery_v_range",
            "rpm_idle", "rpm_range",
        ]
        return keys, np.array([d[k] for k in keys])

    def vec_to_dict(keys, vec, base):
        d = dict(base)
        for k, v in zip(keys, vec):
            d[k] = float(v)
        return d

    def predictions(p_vec, X, keys, base):
        d = vec_to_dict(keys, p_vec, base)
        throttle, alt, amb = X[:, 0], X[:, 1], X[:, 2]
        alt_factor = 1.0 + (alt / 25000.0) * 0.15
        air_density = np.maximum(0.55, 1.0 - alt / 40000.0)
        # Steady-state CHT (heat balance: k_q*throttle*alt_factor = h*(T-T_amb+offset))
        # => T_ss = T_amb + ss_offset + (k_q/h)*throttle*alt_factor
        cht_ss = amb + d["cht_ss_offset"] + (d["cht_kq"] / d["cht_h"]) * throttle * alt_factor
        oil_ss = amb + d["oil_ss_offset"] + (d["oil_kq"] / d["oil_h"]) * throttle
        egt = d["egt_base"] + throttle * d["egt_max_gain"] * air_density
        # rpm fraction from throttle
        rpm_ss = d["rpm_idle"] + throttle * d["rpm_range"]
        rpm_frac = np.clip((rpm_ss - d["rpm_idle"]) / (d["rpm_range"] + 1e-9), 0, 1)
        oil_press = (d["oil_press_idle"] + rpm_frac * d["oil_press_range"]
                     - np.maximum(0, (oil_ss - d["oil_press_temp_ref"]) * d["oil_press_temp_coef"]))
        fuel_flow = (d["fuel_flow_idle"] + throttle * (d["fuel_flow_max"] - d["fuel_flow_idle"])) \
            * (0.9 + 0.1 * air_density)
        vibration = d["vibration_base"] + throttle * d["vibration_gain"]
        battery_v = d["battery_v_base"] + rpm_frac * d["battery_v_range"]
        return np.column_stack([cht_ss, oil_ss, egt, oil_press, fuel_flow, vibration, battery_v, rpm_ss])

    keys, p0 = dict_to_vec(p0_dict)

    def residual_fn(p_vec):
        pred = predictions(p_vec, X, keys, p0_dict)
        # weight each output channel by its magnitude
        weights = np.array([1/25.0, 1/15.0, 1/40.0, 1/0.5, 1/3.0, 1/0.5, 1/0.8, 1/500.0])
        return ((Y - pred) * weights).ravel()

    bounds_lo = [20.0, 0.1, 0.0,  10.0, 0.1, 0.0,  250.0, 200.0,
                 1.0, 0.5, 0.001, 60.0,
                 2.0, 15.0,  0.3, 0.2,  12.0, 0.2,  2000.0, 2000.0]
    bounds_hi = [200.0, 5.0, 50.0, 120.0, 5.0, 50.0, 450.0, 600.0,
                 3.0, 3.5, 0.05, 110.0,
                 6.0, 35.0,  1.5, 2.0, 15.0, 1.0,  2500.0, 4500.0]

    print("calibrate_params: running least-squares optimiser …")
    result = least_squares(residual_fn, p0, bounds=(bounds_lo, bounds_hi),
                           method="trf", ftol=1e-6, xtol=1e-6, max_nfev=4000)

    fitted = vec_to_dict(keys, result.x, p0_dict)

    # ---- Estimate time constants via first-order exponential fit ----
    # Run one long cruise mission and fit exponentials to the transients
    tau_sim = EngineSimulator(ambient_temp_c=25.0, altitude_ft=10000.0, dt_s=1.0, seed=99)
    FP = FlightPhase  # already imported via importlib above
    cht_traj, oil_traj, rpm_traj = [], [], []
    for i in range(600):
        ph = FP.CLIMB if i < 300 else FP.CRUISE
        s = tau_sim.step(ph)
        cht_traj.append(s["cht_c"])
        oil_traj.append(s["oil_temp_c"])
        rpm_traj.append(s["rpm"])

    def fit_tau(traj):
        """Fit y(t)=y_inf - (y_inf-y0)*exp(-t/tau) to a step-response trace."""
        y = np.array(traj)
        y0, y_inf = y[0], y[-1]
        if abs(y_inf - y0) < 1.0:
            return 20.0  # fallback
        # find time to reach 63% of final change
        target = y0 + 0.632 * (y_inf - y0)
        idx = np.argmax(y >= target) if y_inf > y0 else np.argmax(y <= target)
        return max(3.0, float(idx)) if idx > 0 else 20.0

    fitted["tau_cht"] = float(np.clip(fit_tau(cht_traj), 10.0, 120.0))
    fitted["tau_oil"] = float(np.clip(fit_tau(oil_traj), 15.0, 180.0))
    # RPM tau: fit to first 30 ticks
    rpm_arr = np.array(rpm_traj[:60])
    fitted["tau_rpm"] = float(np.clip(fit_tau(rpm_arr.tolist()), 1.0, 10.0))

    # keep C=1 (absorbed into kq and h)
    fitted["cht_C"] = 1.0
    fitted["oil_C"] = 1.0

    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    with open(_TWIN_PARAMS_PATH, "w") as f:
        json.dump(fitted, f, indent=2)
    print(f"calibrate_params: saved → {_TWIN_PARAMS_PATH}")
    print(f"  tau_cht={fitted['tau_cht']:.1f}s  tau_oil={fitted['tau_oil']:.1f}s  tau_rpm={fitted['tau_rpm']:.1f}s")
    return fitted


def _load_twin_params() -> dict:
    """Load calibrated params if they exist, otherwise use defaults."""
    if os.path.exists(_TWIN_PARAMS_PATH):
        with open(_TWIN_PARAMS_PATH) as f:
            return json.load(f)
    return _default_twin_params()


# Module-level calibrated params (loaded once at import)
TWIN_PARAMS: dict = _load_twin_params()


# ---------------------------------------------------------------------------
# PROBLEM 2: Stateful DigitalTwin class
# ---------------------------------------------------------------------------

RESIDUAL_FIELDS = ["cht_c", "egt_c", "oil_temp_c", "oil_press_bar",
                   "fuel_flow_lph", "vibration_g", "battery_v"]


class DigitalTwin:
    """
    Stateful digital-twin model. Holds dynamic state for CHT, oil temp, and RPM
    using exact first-order discretisation: alpha = 1 - exp(-dt/tau).

    Heat balance form:
        C * dT/dt = k_q * throttle * alt_factor  -  h * (T - T_amb)
    Steady state:
        T_ss = T_amb + ss_offset + (k_q/h) * throttle * alt_factor

    Usage
    -----
    twin = DigitalTwin()
    for sample in telemetry_stream:
        result = twin.step(sample, dt_s=1.0)
        # result = {"expected": {...}, "residuals": {...}}
    twin.reset()
    """

    def __init__(self, params: Optional[dict] = None):
        self.p = params if params is not None else TWIN_PARAMS
        # Dynamic states — initialised on first step
        self.cht_hat: Optional[float] = None
        self.oil_temp_hat: Optional[float] = None
        self.rpm_hat: Optional[float] = None
        self._initialised = False

    def reset(self):
        """Reset state (call at mission start)."""
        self.cht_hat = None
        self.oil_temp_hat = None
        self.rpm_hat = None
        self._initialised = False

    def _expected_steady_state(self, throttle: float, altitude_ft: float,
                                ambient_temp_c: float) -> dict:
        """Compute steady-state targets for each output channel."""
        p = self.p
        alt_factor = 1.0 + (altitude_ft / 25000.0) * 0.15
        air_density = max(0.55, 1.0 - altitude_ft / 40000.0)

        # CHT steady state from heat balance
        cht_ss = ambient_temp_c + p["cht_ss_offset"] + (p["cht_kq"] / p["cht_h"]) * throttle * alt_factor
        # Oil temp steady state from heat balance
        oil_ss = ambient_temp_c + p["oil_ss_offset"] + (p["oil_kq"] / p["oil_h"]) * throttle
        # RPM steady state
        rpm_ss = p["rpm_idle"] + throttle * p["rpm_range"]

        return {"cht_ss": cht_ss, "oil_ss": oil_ss, "rpm_ss": rpm_ss,
                "air_density": air_density, "alt_factor": alt_factor}

    def step(self, sample: dict, dt_s: float = 1.0) -> dict:
        """
        Advance the twin by dt_s seconds and return a dict with keys
        "expected" and "residuals".

        Parameters
        ----------
        sample : telemetry dict (simulator output, CAN bus frame, or ECU packet)
        dt_s   : time step in seconds

        Returns
        -------
        {
          "expected": {channel: float, ...},
          "residuals": {channel: float, ...}
        }
        """
        p = self.p
        throttle = float(sample["throttle"])
        altitude_ft = float(sample["altitude_ft"])
        ambient_temp_c = float(sample["ambient_temp_c"])
        rpm_measured = float(sample["rpm"])

        ss = self._expected_steady_state(throttle, altitude_ft, ambient_temp_c)
        cht_ss = ss["cht_ss"]
        oil_ss = ss["oil_ss"]
        rpm_ss = ss["rpm_ss"]
        air_density = ss["air_density"]

        # --- Initialise states from first sample ---
        if not self._initialised:
            self.cht_hat = float(sample.get("cht_c", cht_ss))
            self.oil_temp_hat = float(sample.get("oil_temp_c", oil_ss))
            self.rpm_hat = float(sample.get("rpm", rpm_ss))
            self._initialised = True

        # --- Exact first-order discretisation ---
        # alpha = 1 - exp(-dt/tau)   →  exact solution for step input
        tau_cht = max(1.0, p["tau_cht"])
        tau_oil = max(1.0, p["tau_oil"])
        tau_rpm = max(0.5, p["tau_rpm"])

        alpha_cht = 1.0 - math.exp(-dt_s / tau_cht)
        alpha_oil = 1.0 - math.exp(-dt_s / tau_oil)
        alpha_rpm = 1.0 - math.exp(-dt_s / tau_rpm)

        self.cht_hat += alpha_cht * (cht_ss - self.cht_hat)
        self.oil_temp_hat += alpha_oil * (oil_ss - self.oil_temp_hat)
        self.rpm_hat += alpha_rpm * (rpm_ss - self.rpm_hat)

        # --- Derived expected values ---
        rpm_frac = float(np.clip(
            (self.rpm_hat - p["rpm_idle"]) / (p["rpm_range"] + 1e-9), 0.0, 1.0
        ))
        # Oil pressure uses oil_temp_hat (dynamic state), not steady-state oil temp
        exp_oil_press = (p["oil_press_idle"] + rpm_frac * p["oil_press_range"]
                         - max(0.0, (self.oil_temp_hat - p["oil_press_temp_ref"])
                               * p["oil_press_temp_coef"]))
        exp_egt = p["egt_base"] + throttle * p["egt_max_gain"] * air_density
        exp_fuel_flow = (p["fuel_flow_idle"] + throttle * (p["fuel_flow_max"] - p["fuel_flow_idle"])) \
            * (0.9 + 0.1 * air_density)
        exp_vibration = p["vibration_base"] + throttle * p["vibration_gain"]
        exp_battery_v = p["battery_v_base"] + rpm_frac * p["battery_v_range"]

        expected = {
            "cht_c": self.cht_hat,
            "egt_c": exp_egt,
            "oil_temp_c": self.oil_temp_hat,
            "oil_press_bar": exp_oil_press,
            "fuel_flow_lph": exp_fuel_flow,
            "vibration_g": exp_vibration,
            "battery_v": exp_battery_v,
        }
        residuals = {f: float(sample[f]) - expected[f] for f in RESIDUAL_FIELDS}
        return {"expected": expected, "residuals": residuals}

    @property
    def rpm_estimate(self) -> Optional[float]:
        return self.rpm_hat


# ---------------------------------------------------------------------------
# PROBLEM 3: Engine State Estimator (Kalman-based)
# ---------------------------------------------------------------------------

class _KalmanChannel:
    """
    2-state Kalman filter tracking (bias, drift_per_s) for one residual channel.

    State:  x = [bias, drift]
    Process: x_{k+1} = F x_k + w,  F = [[1, dt],[0, 1]]
    Measurement: z_k = bias_k + noise  (normalised by sigma)
    """

    def __init__(self, sigma_meas: float = 1.0, q_bias: float = 1e-4,
                 q_drift: float = 1e-6, p_init: float = 1.0):
        self.sigma_meas = max(sigma_meas, 1e-9)
        self.q_bias = q_bias
        self.q_drift = q_drift
        # State
        self.x = np.zeros(2)   # [bias, drift]
        # Covariance
        self.P = np.eye(2) * p_init

    def update(self, residual: float, dt: float = 1.0, frozen: bool = False):
        """One Kalman step. If frozen, skip measurement update (Problem 4)."""
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.diag([self.q_bias, self.q_drift])

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        if not frozen:
            # Normalise residual by channel sigma
            z = residual / self.sigma_meas
            H = np.array([[1.0, 0.0]])
            R = np.array([[1.0]])  # variance of normalised measurement
            S = H @ self.P @ H.T + R
            K = self.P @ H.T / S[0, 0]
            self.x = self.x + K.ravel() * (z - H @ self.x)[0]
            self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    @property
    def bias(self) -> float:
        return float(self.x[0])

    @property
    def drift(self) -> float:
        return float(self.x[1])

    def time_to_threshold_s(self, threshold_normalised: float = 3.0) -> Optional[float]:
        """Seconds until |bias| hits threshold at current drift rate."""
        d = self.drift
        if d <= 0:
            return None
        remaining = threshold_normalised - self.bias
        if remaining <= 0:
            return 0.0
        return float(remaining / d)


# Channel index constants
_CHANNELS = RESIDUAL_FIELDS   # vibration, cht, oil_press used for health indices
_VIB_IDX = _CHANNELS.index("vibration_g")
_CHT_IDX = _CHANNELS.index("cht_c")
_OIL_PRESS_IDX = _CHANNELS.index("oil_press_bar")


class EngineStateEstimator:
    """
    Maintains one 2-state Kalman filter per residual channel.
    Derives health indices from the filtered biases.

    Usage
    -----
    est = EngineStateEstimator(sigmas)
    for tick in mission:
        state = est.update(residuals, dt_s, frozen)
        # state = {wear_index, cooling_efficiency, lube_health, drift:{...}, ...}
    """

    def __init__(self, sigmas: Optional[Dict[str, float]] = None):
        """
        sigmas: per-channel noise standard deviation (from calibrate_thresholds).
        Defaults to safe conservative estimates.
        """
        default_sigma = {
            "cht_c": 8.0, "egt_c": 12.0, "oil_temp_c": 5.0,
            "oil_press_bar": 0.15, "fuel_flow_lph": 1.0,
            "vibration_g": 0.15, "battery_v": 0.25,
        }
        sigmas = sigmas or default_sigma
        self._filters: Dict[str, _KalmanChannel] = {
            ch: _KalmanChannel(
                sigma_meas=sigmas.get(ch, 1.0),
                # Larger q_bias so the filter tracks slowly-growing mean
                q_bias=sigmas.get(ch, 1.0) * 2e-4,
                q_drift=sigmas.get(ch, 1.0) * 2e-7,
            )
            for ch in _CHANNELS
        }
        # Cumulative positive vibration excess (for wear index)
        self._cumul_vib_excess: float = 0.0
        # Normalisation scale: vibration grows ~1.5 g at full degradation over
        # ~500,000 ticks (≈ 1e-6 wear rate * 1.55 * dt each tick)
        self._wear_scale: float = float(sigmas.get("vibration_g", 0.15)) * 0.003

    def reset(self):
        for kf in self._filters.values():
            kf.x[:] = 0.0
            kf.P = np.eye(2)
        self._cumul_vib_excess = 0.0

    def update(self, residuals: Dict[str, float], dt_s: float = 1.0,
               frozen: bool = False) -> dict:
        """
        Update all Kalman filters and return engine state dict.

        Returns
        -------
        {
          "wear_index":          float [0,1],
          "cooling_efficiency":  float [0,1],
          "lube_health":         float [0,1],
          "drift":               {channel: float},
          "bias":                {channel: float},
          "time_to_threshold_s": {channel: float or None},
        }
        """
        for ch in _CHANNELS:
            self._filters[ch].update(residuals.get(ch, 0.0), dt_s, frozen)

        # --- Wear index: cumulative positive vibration excess ---
        # Accumulate positive vibration residual (dt-weighted). This gives a
        # monotonically-growing signal correlated with simulator degradation.
        vib_resid = residuals.get("vibration_g", 0.0)
        vib_sigma = self._filters["vibration_g"].sigma_meas
        # Only accumulate excess above the noise floor (half sigma)
        excess = max(0.0, vib_resid - 0.3 * vib_sigma)
        self._cumul_vib_excess += excess * dt_s
        # Normalise: wear_scale chosen so index reaches ~0.5 at degradation=0.5
        wear_index = float(np.clip(self._cumul_vib_excess * self._wear_scale, 0.0, 1.0))

        # CHT bias → cooling efficiency (positive bias = hotter than expected = cooling degraded)
        cht_sigma = self._filters["cht_c"].sigma_meas
        cht_bias = self._filters["cht_c"].bias
        # cooling_efficiency: 1 = perfect, 0 = fully degraded
        cooling_efficiency = float(np.clip(1.0 - cht_bias / (4.0 * cht_sigma), 0.0, 1.0))

        # Oil pressure bias → lube health (negative bias = lower pressure = degraded lubrication)
        op_sigma = self._filters["oil_press_bar"].sigma_meas
        op_bias = self._filters["oil_press_bar"].bias
        lube_health = float(np.clip(1.0 + op_bias / (3.0 * op_sigma), 0.0, 1.0))

        return {
            "wear_index": wear_index,
            "cooling_efficiency": cooling_efficiency,
            "lube_health": lube_health,
            "drift": {ch: self._filters[ch].drift for ch in _CHANNELS},
            "bias": {ch: self._filters[ch].bias for ch in _CHANNELS},
            "time_to_threshold_s": {ch: self._filters[ch].time_to_threshold_s()
                                    for ch in _CHANNELS},
        }


# ---------------------------------------------------------------------------
# PROBLEM 4: Input Validation
# ---------------------------------------------------------------------------

# Safety ranges for inputs
_INPUT_RANGES = {
    "throttle":    (0.0, 1.0),
    "altitude_ft": (-200.0, 50000.0),
    "ambient_temp_c": (-60.0, 70.0),
    "rpm":         (500.0, 7000.0),
}

# Rate-of-change limits per second
_INPUT_RATES = {
    "throttle":    0.35,   # max delta per second
    "rpm":         1200.0, # max RPM change per second
}

_SIGMA_RPM = 80.0         # approximate one-sigma RPM noise


class _InputValidator:
    """Stateful input validator (Problem 4)."""

    def __init__(self, sigma_rpm: float = _SIGMA_RPM, consistency_n: int = 3,
                 consistency_k: float = 4.0):
        self.sigma_rpm = sigma_rpm
        self.consistency_n = consistency_n
        self.consistency_k = consistency_k
        self._prev_sample: Optional[dict] = None
        self._rpm_inconsistent_count: int = 0

    def reset(self):
        self._prev_sample = None
        self._rpm_inconsistent_count = 0

    def validate(self, sample: dict, rpm_hat: Optional[float]) -> Tuple[dict, dict]:
        """
        Validate inputs. Returns (clean_sample, fault_info).

        clean_sample has rpm replaced by rpm_hat if a fault is detected.
        fault_info = {"flag": bool, "channel": str, "reason": str}
        """
        fault = {"flag": False, "channel": "", "reason": ""}

        # 1. Range checks
        for field, (lo, hi) in _INPUT_RANGES.items():
            if field not in sample:
                continue
            val = float(sample[field])
            if not (lo <= val <= hi):
                fault = {"flag": True, "channel": field,
                         "reason": f"{field}={val:.2f} out of range [{lo}, {hi}]"}
                break

        # 2. Rate-of-change check
        if not fault["flag"] and self._prev_sample is not None:
            for field, max_rate in _INPUT_RATES.items():
                if field in sample and field in self._prev_sample:
                    rate = abs(float(sample[field]) - float(self._prev_sample[field]))
                    if rate > max_rate:
                        fault = {"flag": True, "channel": field,
                                 "reason": f"{field} rate {rate:.2f}/s > limit {max_rate}/s"}
                        break

        # 3. RPM vs rpm_hat consistency check
        if not fault["flag"] and rpm_hat is not None:
            rpm_err = abs(float(sample["rpm"]) - rpm_hat)
            if rpm_err > self.consistency_k * self.sigma_rpm:
                self._rpm_inconsistent_count += 1
            else:
                self._rpm_inconsistent_count = 0

            if self._rpm_inconsistent_count >= self.consistency_n:
                fault = {"flag": True, "channel": "rpm",
                         "reason": (f"rpm={sample['rpm']:.0f} vs rpm_hat={rpm_hat:.0f}: "
                                    f"|err|={rpm_err:.0f} > {self.consistency_k}*sigma for "
                                    f"{self._rpm_inconsistent_count} ticks")}

        # Substitute rpm_hat when fault detected on rpm channel
        clean = dict(sample)
        if fault["flag"] and fault["channel"] == "rpm" and rpm_hat is not None:
            clean["rpm"] = rpm_hat

        self._prev_sample = sample
        return clean, fault


def validate_inputs(sample: dict, twin_state: Optional[dict] = None) -> Tuple[dict, dict]:
    """
    Module-level convenience wrapper (stateless approximation).
    For full stateful validation, use DigitalTwin._validator or the
    DigitalTwin.step() method (which calls this internally via its own validator).
    """
    rpm_hat = twin_state.get("rpm_hat") if twin_state else None
    _val = _InputValidator()
    return _val.validate(sample, rpm_hat)


# ---------------------------------------------------------------------------
# PROBLEM 5: Calibrated Thresholds and Persistent Corroborator
# ---------------------------------------------------------------------------

_THRESHOLD_FLOORS = {
    "cht_c": 8.0, "egt_c": 12.0, "oil_temp_c": 5.0,
    "oil_press_bar": 0.12, "fuel_flow_lph": 0.8,
    "vibration_g": 0.12, "battery_v": 0.15,
}


def calibrate_thresholds(n_missions: int = 30, k: float = 4.0,
                         warmup_s: int = 60, seed: int = 42) -> dict:
    """
    Run n_missions healthy simulator missions through DigitalTwin,
    discard first warmup_s seconds, compute per-channel residual sigma,
    set threshold = max(k*sigma, floor).

    Saves to models/artifacts/twin_thresholds.json.
    """
    import sys, importlib
    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj not in sys.path:
        sys.path.insert(0, _proj)
    _sim_mod = importlib.import_module("simulator.engine_simulator")
    EngineSimulator = _sim_mod.EngineSimulator
    FlightPhase = _sim_mod.FlightPhase
    PHASE_PROFILE = _sim_mod.PHASE_PROFILE

    MISSION_PHASES = [
        (FlightPhase.GROUND, 60), (FlightPhase.TAKEOFF, 30),
        (FlightPhase.CLIMB, 300), (FlightPhase.CRUISE, 900),
        (FlightPhase.DESCENT, 200), (FlightPhase.LANDING, 60),
    ]

    rng = np.random.default_rng(seed)
    all_resids = {f: [] for f in RESIDUAL_FIELDS}

    print(f"calibrate_thresholds: running {n_missions} healthy missions …")
    for mission_i in range(n_missions):
        amb = rng.uniform(-5, 40)
        alt = rng.uniform(5000, 20000)
        sim = EngineSimulator(ambient_temp_c=amb, altitude_ft=alt, dt_s=1.0,
                              seed=int(rng.integers(0, 1_000_000)))
        twin = DigitalTwin()
        t_s = 0.0
        for phase, duration in MISSION_PHASES:
            for _ in range(duration):
                throttle = float(np.clip(
                    PHASE_PROFILE[phase]["throttle"] + rng.normal(0, 0.04),
                    0.05, 1.0
                ))
                sample = sim.step(phase, throttle_override=throttle)
                result = twin.step(sample, dt_s=1.0)
                if t_s >= warmup_s:
                    for f in RESIDUAL_FIELDS:
                        all_resids[f].append(result["residuals"][f])
                t_s += 1.0

    sigmas = {f: float(np.std(all_resids[f])) for f in RESIDUAL_FIELDS}
    thresholds = {f: float(max(k * sigmas[f], _THRESHOLD_FLOORS[f]))
                  for f in RESIDUAL_FIELDS}

    result_dict = {"thresholds": thresholds, "sigmas": sigmas, "k": k, "n_missions": n_missions}
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    with open(_TWIN_THRESHOLDS_PATH, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"calibrate_thresholds: saved → {_TWIN_THRESHOLDS_PATH}")
    for ch, thr in thresholds.items():
        print(f"  {ch:20s} sigma={sigmas[ch]:.3f}  threshold={thr:.3f}")
    return result_dict


def _load_thresholds() -> Tuple[dict, dict]:
    """Load calibrated thresholds (and sigmas) from artifact."""
    if os.path.exists(_TWIN_THRESHOLDS_PATH):
        with open(_TWIN_THRESHOLDS_PATH) as f:
            data = json.load(f)
        return data.get("thresholds", {}), data.get("sigmas", {})
    # Fallback to fixed thresholds (original behaviour)
    thresholds = {"cht_c": 25, "egt_c": 40, "oil_temp_c": 15, "oil_press_bar": 0.5,
                  "fuel_flow_lph": 3.0, "vibration_g": 0.5, "battery_v": 0.8}
    sigmas = {k: v / 4.0 for k, v in thresholds.items()}
    return thresholds, sigmas


# Module-level thresholds (loaded once)
_THRESHOLDS, _SIGMAS = _load_thresholds()


class Corroborator:
    """
    Stateful corroboration check with persistence filter.
    A channel is flagged only if it exceeds its threshold for >= 3 of the
    last 5 ticks (majority filter). Verdict names are unchanged from v1.
    """

    def __init__(self, thresholds: Optional[dict] = None, window: int = 5,
                 min_count: int = 3):
        self.thresholds = thresholds or _THRESHOLDS
        self.window = window
        self.min_count = min_count
        # ring buffer: {channel: deque of bools}
        from collections import deque
        self._history: Dict[str, Any] = {
            f: deque(maxlen=window) for f in RESIDUAL_FIELDS
        }

    def reset(self):
        from collections import deque
        self._history = {f: deque(maxlen=self.window) for f in RESIDUAL_FIELDS}

    def check(self, residuals: dict) -> dict:
        """
        Run corroboration check and return a dict compatible with v1 output shape.
        """
        # Update history
        for f in RESIDUAL_FIELDS:
            exceeded = abs(residuals.get(f, 0.0)) > self.thresholds.get(f, 1e9)
            self._history[f].append(exceeded)

        # Persistence filter: flag if >= min_count of last window ticks exceeded
        flagged = {
            f: sum(self._history[f]) >= self.min_count
            for f in RESIDUAL_FIELDS
        }
        n_flagged = sum(flagged.values())

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


# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------

# Module-level twin (deprecated usage path)
_module_twin = DigitalTwin()
_module_validator = _InputValidator()


def compute_residuals(actual_sample: dict) -> dict:
    """
    DEPRECATED — use DigitalTwin.step() per mission instance instead.

    Backward-compatible wrapper around the module-level DigitalTwin instance.
    Returns {field: residual} where residual = actual - expected.
    """
    warnings.warn(
        "compute_residuals() is deprecated. Instantiate DigitalTwin per mission "
        "and call twin.step(sample, dt_s). The module-level instance shares state "
        "across missions which degrades residual quality.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = _module_twin.step(actual_sample, dt_s=1.0)
    return result["residuals"]


def corroboration_check(residuals: dict, thresholds: dict = None) -> dict:
    """
    Stateless corroboration check (no persistence filter).
    For stateful persistence filtering, use Corroborator.check().
    Maintained for backward compatibility with main.py callers.
    """
    thr = thresholds or _THRESHOLDS
    flagged = {f: abs(residuals.get(f, 0.0)) > thr.get(f, 1e9) for f in RESIDUAL_FIELDS}
    n_flagged = sum(flagged.values())
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
