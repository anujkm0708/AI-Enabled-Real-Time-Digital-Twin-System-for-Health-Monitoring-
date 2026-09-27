"""
AeroTwin - Physics-Informed Piston Engine Simulator
=====================================================
Simulates a Rotax 914/915-style turbocharged 4-stroke aero piston engine
used on MALE UAVs. Generates realistic dynamic telemetry across mission
flight phases, with support for manual fault injection for demo/training.

This module is the "virtual truth" source for the whole AeroTwin system:
it stands in for real sensors since no physical engine/hardware is available.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FlightPhase(Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    LANDING = "landing"


# Baseline throttle % and target altitude gain per phase
PHASE_PROFILE = {
    FlightPhase.GROUND:   {"throttle": 0.15, "climb_rate_fps": 0.0},
    FlightPhase.TAKEOFF:  {"throttle": 0.95, "climb_rate_fps": 15.0},
    FlightPhase.CLIMB:    {"throttle": 0.80, "climb_rate_fps": 8.0},
    FlightPhase.CRUISE:   {"throttle": 0.55, "climb_rate_fps": 0.0},
    FlightPhase.DESCENT:  {"throttle": 0.30, "climb_rate_fps": -6.0},
    FlightPhase.LANDING:  {"throttle": 0.20, "climb_rate_fps": -3.0},
}

# Healthy-engine reference points (approx Rotax 915iS-class figures, simplified)
REF = {
    "rpm_idle": 2200, "rpm_max": 5800,
    "cht_ambient_offset": 25.0,
    "cht_max_gain": 95.0,
    "egt_base": 350.0, "egt_max_gain": 480.0,
    "oil_temp_base": 60.0, "oil_temp_max_gain": 55.0,
    "oil_press_idle": 2.2, "oil_press_cruise": 4.2,
    "fuel_flow_idle": 4.0, "fuel_flow_max": 28.0,
    "vibration_base": 0.8, "vibration_max_gain": 2.2,
    "battery_v_base": 13.8,
}

# Full 8-class fault taxonomy matching DRDO PS 26054
FAULT_TYPES = [
    "sensor_drift",
    "overheating_trend",
    "oil_pressure_loss",
    "misfire_condition",
    "injector_abnormality",
    "combustion_instability",
    "cooling_degradation",
    "lubrication_degradation",
]


@dataclass
class EngineState:
    """Internal thermal/mechanical state that persists across ticks."""
    cht: float = 45.0
    oil_temp: float = 45.0
    rpm: float = REF["rpm_idle"]
    degradation: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class FaultInjection:
    fault_type: Optional[str] = None
    severity: float = 0.0
    started_at_s: float = 0.0


class EngineSimulator:
    """
    Generates one telemetry sample per call to step(), given the current
    flight phase and ambient conditions. Call inject_fault()/clear_fault()
    to demo the AI's response to failures.
    """

    def __init__(self, ambient_temp_c: float = 25.0, altitude_ft: float = 0.0,
                 dt_s: float = 1.0, seed: Optional[int] = None):
        self.ambient_temp_c = ambient_temp_c
        self.altitude_ft = altitude_ft
        self.dt_s = dt_s
        self.state = EngineState()
        self.fault = FaultInjection()
        self.rng = np.random.default_rng(seed)
        self._tau_cht = 25.0
        self._tau_oil = 45.0

    def inject_fault(self, fault_type: str, severity: float = 0.7):
        assert fault_type in FAULT_TYPES, f"unknown fault type: {fault_type}"
        self.fault = FaultInjection(fault_type=fault_type, severity=severity,
                                     started_at_s=self.state.elapsed_s)

    def clear_fault(self):
        self.fault = FaultInjection()

    def step(self, phase: FlightPhase, throttle_override: Optional[float] = None):
        """Advance simulation by dt_s seconds and return one telemetry sample (dict)."""
        profile = PHASE_PROFILE[phase]
        throttle = throttle_override if throttle_override is not None else profile["throttle"]
        throttle = float(np.clip(throttle, 0.0, 1.0))

        alt_factor = 1.0 + (self.altitude_ft / 25000.0) * 0.15
        air_density_factor = max(0.55, 1.0 - self.altitude_ft / 40000.0)

        # RPM
        target_rpm = REF["rpm_idle"] + throttle * (REF["rpm_max"] - REF["rpm_idle"])
        self.state.rpm += (target_rpm - self.state.rpm) * min(1.0, self.dt_s / 3.0)

        # CHT
        target_cht = (self.ambient_temp_c + REF["cht_ambient_offset"]
                      + throttle * REF["cht_max_gain"] * alt_factor)
        self.state.cht += (target_cht - self.state.cht) * (self.dt_s / self._tau_cht)

        # Oil temp
        target_oil = self.ambient_temp_c + REF["oil_temp_base"] * 0.5 + throttle * REF["oil_temp_max_gain"]
        self.state.oil_temp += (target_oil - self.state.oil_temp) * (self.dt_s / self._tau_oil)

        # EGT
        egt = REF["egt_base"] + throttle * REF["egt_max_gain"] * air_density_factor
        egt += self.rng.normal(0, 4.0)

        # Oil pressure
        rpm_frac = (self.state.rpm - REF["rpm_idle"]) / (REF["rpm_max"] - REF["rpm_idle"])
        oil_press = REF["oil_press_idle"] + rpm_frac * (REF["oil_press_cruise"] - REF["oil_press_idle"])
        oil_press -= max(0.0, (self.state.oil_temp - 90) * 0.01)
        oil_press += self.rng.normal(0, 0.05)

        # Fuel flow
        fuel_flow = REF["fuel_flow_idle"] + throttle * (REF["fuel_flow_max"] - REF["fuel_flow_idle"])
        fuel_flow *= (0.9 + 0.1 * air_density_factor)
        fuel_flow += self.rng.normal(0, 0.3)

        # Vibration
        vibration = (REF["vibration_base"] + throttle * REF["vibration_max_gain"] * 0.4
                     + self.state.degradation * 1.5)
        vibration += abs(self.rng.normal(0, 0.08))

        # Battery voltage
        battery_v = REF["battery_v_base"] + (rpm_frac * 0.6) + self.rng.normal(0, 0.05)

        # Long-term wear
        wear_rate = 1e-6 * (1 + throttle) * (1 + max(0, self.state.oil_temp - 100) * 0.05)
        self.state.degradation = min(1.0, self.state.degradation + wear_rate * self.dt_s)

        sample = {
            "t_s": self.state.elapsed_s,
            "phase": phase.value,
            "throttle": throttle,
            "altitude_ft": self.altitude_ft,
            "ambient_temp_c": self.ambient_temp_c,
            "rpm": self.state.rpm,
            "cht_c": self.state.cht,
            "egt_c": egt,
            "oil_temp_c": self.state.oil_temp,
            "oil_press_bar": oil_press,
            "fuel_flow_lph": fuel_flow,
            "vibration_g": vibration,
            "battery_v": battery_v,
            "degradation": self.state.degradation,
        }

        sample = self._apply_fault(sample)
        self.state.elapsed_s += self.dt_s
        return sample

    def _apply_fault(self, sample: dict) -> dict:
        if self.fault.fault_type is None:
            sample["fault_active"] = "none"
            return sample

        elapsed_since_fault = self.state.elapsed_s - self.fault.started_at_s
        ramp = min(1.0, elapsed_since_fault / 20.0)
        sev = self.fault.severity * ramp

        if self.fault.fault_type == "sensor_drift":
            # Single sensor (CHT) reads high — classic instrumentation fault
            sample["cht_c"] += sev * 120.0

        elif self.fault.fault_type == "overheating_trend":
            # Real thermal fault — corroborated across CHT + oil temp + oil press
            sample["cht_c"] += sev * 60.0
            sample["oil_temp_c"] += sev * 35.0
            sample["oil_press_bar"] -= sev * 0.6

        elif self.fault.fault_type == "oil_pressure_loss":
            sample["oil_press_bar"] -= sev * 1.8
            sample["oil_temp_c"] += sev * 20.0
            sample["vibration_g"] += sev * 0.5

        elif self.fault.fault_type == "misfire_condition":
            sample["rpm"] -= sev * 350
            sample["vibration_g"] += sev * 1.8
            sample["egt_c"] -= sev * 60.0
            sample["fuel_flow_lph"] += sev * 3.0

        elif self.fault.fault_type == "injector_abnormality":
            # Injector stuck/clogged: rich/lean mixture, EGT spike + RPM drop + high fuel flow
            sample["egt_c"] += sev * 80.0
            sample["fuel_flow_lph"] += sev * 5.0
            sample["rpm"] -= sev * 200
            sample["vibration_g"] += sev * 0.6

        elif self.fault.fault_type == "combustion_instability":
            # Cyclic variation in combustion: vibration surge + oscillating EGT/RPM
            osc = np.sin(self.state.elapsed_s * 0.5)  # 0.5 Hz oscillation
            sample["vibration_g"] += sev * (1.5 + 0.5 * osc)
            sample["egt_c"] += sev * (40.0 * osc)
            sample["rpm"] -= sev * (150 * (1 - osc))

        elif self.fault.fault_type == "cooling_degradation":
            # Gradual cooling system loss: CHT rises slowly, EGT up, oil temp up
            sample["cht_c"] += sev * 45.0
            sample["egt_c"] += sev * 30.0
            sample["oil_temp_c"] += sev * 25.0

        elif self.fault.fault_type == "lubrication_degradation":
            # Oil viscosity loss / contamination: oil press drops, oil temp rises, vibration up
            sample["oil_press_bar"] -= sev * 1.2
            sample["oil_temp_c"] += sev * 30.0
            sample["vibration_g"] += sev * 0.8
            sample["fuel_flow_lph"] += sev * 1.5  # engine working harder

        sample["fault_active"] = self.fault.fault_type
        sample["fault_severity"] = round(sev, 3)
        return sample


if __name__ == "__main__":
    sim = EngineSimulator(ambient_temp_c=28, altitude_ft=12000, seed=42)
    for i in range(5):
        print(sim.step(FlightPhase.CLIMB))
