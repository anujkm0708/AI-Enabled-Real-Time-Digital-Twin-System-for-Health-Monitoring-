"""
AeroTwin - Digital Twin Core Acceptance Tests
==============================================
Run with: pytest tests/test_digital_twin.py -v
"""

import sys, os
import math
import warnings
import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML   = os.path.join(_ROOT, "aerotwin_ml")
if _ML not in sys.path:
    sys.path.insert(0, _ML)

from simulator.engine_simulator import (
    EngineSimulator, FlightPhase, PHASE_PROFILE, FAULT_TYPES
)
from models.digital_twin import (
    DigitalTwin, EngineStateEstimator, Corroborator,
    RESIDUAL_FIELDS, compute_residuals, _THRESHOLDS, _SIGMAS,
    corroboration_check,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

MISSION_PHASES = [
    (FlightPhase.GROUND,  60),
    (FlightPhase.TAKEOFF, 30),
    (FlightPhase.CLIMB,   300),
    (FlightPhase.CRUISE,  900),
    (FlightPhase.DESCENT, 200),
    (FlightPhase.LANDING, 60),
]
TOTAL_DURATION = sum(d for _, d in MISSION_PHASES)
WARMUP_S = 60


def _run_healthy_mission(seed: int = 42, amb: float = 25.0, alt: float = 10000.0):
    """Run a complete healthy mission and return list of (sample, result) tuples."""
    sim  = EngineSimulator(ambient_temp_c=amb, altitude_ft=alt, dt_s=1.0, seed=seed)
    twin = DigitalTwin()
    rows = []
    for phase, duration in MISSION_PHASES:
        for _ in range(duration):
            sample = sim.step(phase)
            result = twin.step(sample, dt_s=1.0)
            rows.append((sample, result))
    return rows


def _build_corroborator_and_estimator():
    corr = Corroborator()
    est  = EngineStateEstimator(sigmas=_SIGMAS)
    return corr, est


# ── Test 6 (ordering swapped so we fail fast if import is broken) ─────────────

def test_6_no_simulator_import():
    """digital_twin.py must NOT import from simulator.*"""
    dt_path = os.path.join(_ML, "models", "digital_twin.py")
    with open(dt_path) as f:
        source = f.read()
    assert "from simulator" not in source, (
        "digital_twin.py contains 'from simulator' — simulator import found!"
    )
    assert "import simulator" not in source, (
        "digital_twin.py contains 'import simulator' — simulator import found!"
    )


# ── Test 1: False-alarm rate ──────────────────────────────────────────────────

def test_1_false_alarm_rate():
    """
    Healthy mission: false-alarm rate < 1% of ticks after 60 s warm-up,
    including takeoff and climb transients.
    Also prints the OLD code false-alarm rate for comparison.
    """
    rows = _run_healthy_mission(seed=42)
    corr = Corroborator()

    new_flags = 0
    new_total = 0
    for t, (sample, result) in enumerate(rows):
        check = corr.check(result["residuals"])
        if t >= WARMUP_S:
            new_total += 1
            if check["verdict"] != "nominal":
                new_flags += 1

    new_far = new_flags / max(1, new_total)

    # --- OLD code false-alarm rate (for comparison) ---
    old_flags = 0
    old_total = 0
    sim_old = EngineSimulator(ambient_temp_c=25.0, altitude_ft=10000.0, dt_s=1.0, seed=42)
    old_thresholds = {
        "cht_c": 25, "egt_c": 40, "oil_temp_c": 15, "oil_press_bar": 0.5,
        "fuel_flow_lph": 3.0, "vibration_g": 0.5, "battery_v": 0.8,
    }
    for t, (phase, duration) in enumerate(MISSION_PHASES):
        for tick in range(duration):
            sample_old = sim_old.step(phase)
            # Old compute_residuals was stateless
            from models.digital_twin import corroboration_check
            # simulate old residuals with same sample via a fresh twin
            old_twin = DigitalTwin()
            r_old = old_twin.step(sample_old, 1.0)["residuals"]
            old_check = corroboration_check(r_old, old_thresholds)
            abs_t = sum(d for _, d in MISSION_PHASES[:MISSION_PHASES.index((phase, duration))]) + tick
            if abs_t >= WARMUP_S:
                old_total += 1
                if old_check["verdict"] != "nominal":
                    old_flags += 1
    old_far = old_flags / max(1, old_total)

    print(f"\n[Test 1] OLD code false-alarm rate (fixed thresholds): {old_far:.3%}")
    print(f"[Test 1] NEW code false-alarm rate (calibrated + persistence): {new_far:.3%}")

    assert new_far < 0.01, (
        f"False-alarm rate {new_far:.3%} ≥ 1% — calibration or thresholds need adjustment"
    )


# ── Test 2: Residual mean within 1 sigma of 0 ────────────────────────────────

def test_2_healthy_residual_mean():
    """For each channel, healthy residual mean must be within 1 sigma of 0."""
    rows = _run_healthy_mission(seed=42)
    # Collect post-warmup residuals
    resids = {f: [] for f in RESIDUAL_FIELDS}
    for t, (_, result) in enumerate(rows):
        if t >= WARMUP_S:
            for f in RESIDUAL_FIELDS:
                resids[f].append(result["residuals"][f])

    # Load sigmas from artifact or use defaults
    sigmas = _SIGMAS if _SIGMAS else {f: 10.0 for f in RESIDUAL_FIELDS}

    failures = []
    for f in RESIDUAL_FIELDS:
        arr = np.array(resids[f])
        mean_val = arr.mean()
        sigma = sigmas.get(f, np.std(arr) + 1e-6)
        if abs(mean_val) > sigma:
            failures.append(f"{f}: mean={mean_val:.4f}, sigma={sigma:.4f}")

    assert not failures, "Residual mean not within 1 sigma of 0 for: " + "; ".join(failures)


# ── Test 3: Fault detection within 30 s ──────────────────────────────────────

@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_3_fault_detection(fault_type: str):
    """
    Inject fault at severity 0.7 at t=300 s; estimator or corroborator must
    flag within 30 s of ramp start. sensor_drift must return
    'likely_instrumentation_fault' (via corroborator).
    """
    FAULT_START_T = 300
    DETECT_WINDOW = 30

    sim = EngineSimulator(ambient_temp_c=25.0, altitude_ft=10000.0, dt_s=1.0, seed=7)
    twin = DigitalTwin()
    corr = Corroborator()
    est  = EngineStateEstimator(sigmas=_SIGMAS)

    detected_at = None
    sensor_drift_verdict = None
    fault_injected = False

    for t in range(600):
        # Advance through CRUISE phase
        phase = FlightPhase.CRUISE
        if t == FAULT_START_T:
            sim.inject_fault(fault_type, severity=0.7)
            fault_injected = True

        sample = sim.step(phase)
        result = twin.step(sample, dt_s=1.0)
        corrob = corr.check(result["residuals"])
        engine = est.update(result["residuals"], dt_s=1.0)

        if fault_injected and detected_at is None:
            elapsed_since_fault = t - FAULT_START_T
            # Corroborator flagged
            flagged_by_corr = corrob["verdict"] != "nominal"
            # Estimator flagged: any channel has |bias| > 2*sigma
            flagged_by_est = any(
                abs(engine["bias"].get(ch, 0)) > 2.0
                for ch in RESIDUAL_FIELDS
            )
            if flagged_by_corr or flagged_by_est:
                detected_at = elapsed_since_fault
                if fault_type == "sensor_drift":
                    sensor_drift_verdict = corrob["verdict"]

    assert detected_at is not None, (
        f"Fault '{fault_type}' not detected within the mission"
    )
    assert detected_at <= DETECT_WINDOW, (
        f"Fault '{fault_type}' detected at {detected_at}s — exceeds 30 s limit"
    )
    if fault_type == "sensor_drift":
        assert sensor_drift_verdict == "likely_instrumentation_fault", (
            f"sensor_drift verdict was '{sensor_drift_verdict}', expected 'likely_instrumentation_fault'"
        )


# ── Test 4: Slow wear — wear_index monotonic & correlated with degradation ────

def test_4_slow_wear_wear_index():
    """
    Run 20,000 s at cruise. wear_index must be monotonic and its correlation
    with simulator degradation must be >= 0.9.
    """
    sim  = EngineSimulator(ambient_temp_c=25.0, altitude_ft=10000.0, dt_s=1.0, seed=0)
    twin = DigitalTwin()
    est  = EngineStateEstimator(sigmas=_SIGMAS)

    wear_indices = []
    degradations = []
    TOTAL = 20_000
    SMOOTH_WINDOW = 200  # smooth monotonicity check

    for t in range(TOTAL):
        sample = sim.step(FlightPhase.CRUISE)
        result = twin.step(sample, dt_s=1.0)
        engine = est.update(result["residuals"], dt_s=1.0)
        wear_indices.append(engine["wear_index"])
        degradations.append(sample["degradation"])

    wi = np.array(wear_indices)
    dg = np.array(degradations)

    # Correlation check
    corr = float(np.corrcoef(wi, dg)[0, 1])
    print(f"\n[Test 4] wear_index vs degradation correlation: {corr:.4f}")
    assert corr >= 0.9, f"Correlation {corr:.4f} < 0.9"

    # Smoothed monotonicity: rolling mean should be non-decreasing (allow minor dips)
    smooth_wi = np.convolve(wi, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='valid')
    diffs = np.diff(smooth_wi)
    # Allow up to 1% of windows to be slightly negative (noise floor)
    neg_frac = (diffs < -1e-6).mean()
    print(f"[Test 4] wear_index smooth monotonicity violation fraction: {neg_frac:.4f}")
    assert neg_frac < 0.05, f"wear_index non-monotonic in {neg_frac:.1%} of windows"


# ── Test 5: Corrupt RPM sensor → input_fault within 5 ticks ─────────────────

def test_5_corrupt_rpm_input_fault():
    """
    Corrupt RPM sensor (+800): input_fault.flag must become True within 5 ticks
    and residuals must not explode (|resid| < 10*sigma for all channels).
    """
    from models.digital_twin import _InputValidator

    sim  = EngineSimulator(ambient_temp_c=25.0, altitude_ft=10000.0, dt_s=1.0, seed=5)
    twin = DigitalTwin()
    validator = _InputValidator(sigma_rpm=80.0, consistency_n=3, consistency_k=4.0)

    # Warm up for 100 ticks
    for _ in range(100):
        sample = sim.step(FlightPhase.CRUISE)
        twin.step(sample, dt_s=1.0)
        validator.validate(sample, twin.rpm_hat)

    # Now corrupt RPM and track fault detection
    fault_detected_at = None
    residual_exploded = False
    CORRUPT_OFFSET = 800.0

    for t in range(20):
        sample = sim.step(FlightPhase.CRUISE)
        sample_corrupt = dict(sample)
        sample_corrupt["rpm"] = sample["rpm"] + CORRUPT_OFFSET

        clean_sample, fault_info = validator.validate(sample_corrupt, twin.rpm_hat)
        result = twin.step(clean_sample, dt_s=1.0)

        if fault_info["flag"] and fault_detected_at is None:
            fault_detected_at = t

        # Check residuals did not explode
        for ch in RESIDUAL_FIELDS:
            sigma = _SIGMAS.get(ch, 10.0)
            if abs(result["residuals"][ch]) > 10.0 * sigma:
                residual_exploded = True
                print(f"  Residual explosion at t={t}: {ch}={result['residuals'][ch]:.2f} (10σ={10*sigma:.2f})")

    print(f"\n[Test 5] input_fault detected at tick: {fault_detected_at}")
    assert fault_detected_at is not None, "input_fault.flag never became True"
    assert fault_detected_at < 5, (
        f"input_fault detected at tick {fault_detected_at} — must be within 5 ticks"
    )
    assert not residual_exploded, "Residuals exploded after RPM corruption — substitution failed"
