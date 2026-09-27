"""
AeroTwin - Dataset Generator
==============================
Runs many simulated missions (healthy + fault-injected) through the
simulator + digital twin, computes residuals, and saves a labeled
dataset for training the anomaly detector, fault classifier, and RUL model.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from simulator.engine_simulator import EngineSimulator, FlightPhase, FAULT_TYPES
from models.digital_twin import compute_residuals, RESIDUAL_FIELDS

MISSION_PHASE_SEQUENCE = [
    (FlightPhase.GROUND, 60), (FlightPhase.TAKEOFF, 30), (FlightPhase.CLIMB, 300),
    (FlightPhase.CRUISE, 900), (FlightPhase.DESCENT, 200), (FlightPhase.LANDING, 60),
]


def run_mission(mission_id: int, inject_fault: bool, rng: np.random.Generator) -> pd.DataFrame:
    ambient = rng.uniform(-5, 40)
    altitude = rng.uniform(5000, 20000)
    sim = EngineSimulator(ambient_temp_c=ambient, altitude_ft=altitude, dt_s=1.0,
                           seed=int(rng.integers(0, 1_000_000)))

    fault_type = None
    fault_start_frac = rng.uniform(0.4, 0.8)
    if inject_fault:
        fault_type = rng.choice(FAULT_TYPES)

    total_duration = sum(d for _, d in MISSION_PHASE_SEQUENCE)
    fault_start_t = fault_start_frac * total_duration

    rows = []
    for phase, duration in MISSION_PHASE_SEQUENCE:
        for _ in range(duration):
            if inject_fault and sim.fault.fault_type is None and sim.state.elapsed_s >= fault_start_t:
                sim.inject_fault(fault_type, severity=rng.uniform(0.5, 1.0))
            sample = sim.step(phase, throttle_override=phase and PHASE_JITTER(phase, rng))
            residuals = compute_residuals(sample)
            row = {**sample, **{f"resid_{k}": v for k, v in residuals.items()}}
            row["mission_id"] = mission_id
            rows.append(row)
    return pd.DataFrame(rows)


def PHASE_JITTER(phase, rng):
    from simulator.engine_simulator import PHASE_PROFILE
    base = PHASE_PROFILE[phase]["throttle"]
    return float(np.clip(base + rng.normal(0, 0.04), 0.05, 1.0))


def generate(n_healthy=25, n_faulty=25, seed=7, out_path="data/aerotwin_dataset.csv"):
    rng = np.random.default_rng(seed)
    dfs = []
    mid = 0
    for _ in range(n_healthy):
        dfs.append(run_mission(mid, inject_fault=False, rng=rng)); mid += 1
    for _ in range(n_faulty):
        dfs.append(run_mission(mid, inject_fault=True, rng=rng)); mid += 1
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(out_path, index=False)
    print(f"Generated {len(full)} rows across {mid} missions -> {out_path}")
    print(full["fault_active"].value_counts())
    return full


if __name__ == "__main__":
    generate()
