"""
AeroTwin - Telemetry Source Abstraction
=========================================
Defines a clean interface between the AI/backend pipeline and the data
source (simulated, CAN bus, or edge ECU). The backend's simulation loop
should instantiate a TelemetrySource subclass and call read_sample() each
tick — no other backend code changes are needed when swapping data sources.

Phase 1 (current): SimulatedTelemetrySource wraps EngineSimulator.
Phase 2 (real CAN): CANBusTelemetrySource reads from SocketCAN.
Phase 3 (edge ECU): EdgeECUTelemetrySource reads from FADEC serial/UDP.

CAN_FRAME_MAP documents the expected CAN frame layout for a real
Rotax 915iS / custom FADEC integration (SAE J1939-compatible).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from simulator.engine_simulator import EngineSimulator, FlightPhase, FAULT_TYPES

# ---------------------------------------------------------------------------
# CAN Frame Layout Reference (Phase 2 integration)
# ---------------------------------------------------------------------------
# Maps CAN PGN (Parameter Group Number) or message ID -> field -> signal details
# Format: {pgn_hex: {field_name: (start_bit, length_bits, scale, offset, unit)}}
#
# This is the expected wire format from a Rotax 915iS FADEC or a custom
# engine controller running J1939-compatible messaging at 250 kbps.
#
CAN_FRAME_MAP = {
    "0x18FF0101": {  # Engine Speed + Throttle (100 ms cycle)
        "rpm":      (0,  16, 0.125, 0,    "rpm"),
        "throttle": (16,  8, 0.4,   0,    "percent"),
    },
    "0x18FF0102": {  # Cylinder Head Temperature (200 ms cycle)
        "cht_c":    (0, 16, 0.03125, -273.15, "degC"),
    },
    "0x18FF0103": {  # Exhaust Gas Temperature (200 ms cycle)
        "egt_c":    (0, 16, 0.03125, -273.15, "degC"),
    },
    "0x18FF0104": {  # Oil System (200 ms cycle)
        "oil_temp_c":   (0,  16, 0.03125, -273.15, "degC"),
        "oil_press_bar":(16, 12, 0.00391,  0,       "bar"),
    },
    "0x18FF0105": {  # Fuel Flow (100 ms cycle)
        "fuel_flow_lph":(0, 16, 0.05, 0, "L/hr"),
    },
    "0x18FF0106": {  # Vibration Accelerometer (50 ms cycle)
        "vibration_g":  (0, 12, 0.001, 0, "g RMS"),
    },
    "0x18FF0107": {  # Electrical (500 ms cycle)
        "battery_v":    (0, 12, 0.02, 0, "V"),
    },
}


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class TelemetrySource(ABC):
    """
    Abstract interface for any telemetry data source.
    Implementations must return a dict with the same keys as
    EngineSimulator.step() so the AI pipeline works unchanged.
    """

    @abstractmethod
    def read_sample(self, phase: Optional[Any] = None) -> Dict[str, Any]:
        """
        Read one telemetry sample. Returns a dict with at minimum:
          t_s, phase, throttle, altitude_ft, ambient_temp_c,
          rpm, cht_c, egt_c, oil_temp_c, oil_press_bar,
          fuel_flow_lph, vibration_g, battery_v, degradation, fault_active
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the data source is live and producing data."""
        ...

    def inject_fault(self, fault_type: str, severity: float = 0.7):
        """
        Optionally supported by simulation sources for demo/testing.
        Real hardware sources should raise NotImplementedError.
        """
        raise NotImplementedError("Fault injection not supported on this source")

    def clear_fault(self):
        raise NotImplementedError("Fault injection not supported on this source")


# ---------------------------------------------------------------------------
# Phase 1: Simulated Source (current, fully working)
# ---------------------------------------------------------------------------

class SimulatedTelemetrySource(TelemetrySource):
    """
    Wraps EngineSimulator for use as a TelemetrySource.
    Handles phase cycling internally so the backend just calls read_sample().
    """

    PHASE_SEQUENCE = [
        (FlightPhase.GROUND,  60),
        (FlightPhase.TAKEOFF, 30),
        (FlightPhase.CLIMB,   300),
        (FlightPhase.CRUISE,  900),
        (FlightPhase.DESCENT, 200),
        (FlightPhase.LANDING, 60),
    ]

    def __init__(self, ambient_temp_c: float = 25.0, altitude_ft: float = 10000.0,
                 dt_s: float = 1.0, seed: Optional[int] = None):
        self._sim = EngineSimulator(
            ambient_temp_c=ambient_temp_c,
            altitude_ft=altitude_ft,
            dt_s=dt_s,
            seed=seed,
        )
        self._phase_idx = 0
        self._ticks_in_phase = 0
        self._connected = True

    def read_sample(self, phase: Optional[Any] = None) -> Dict[str, Any]:
        """
        If phase is supplied, use it directly.
        Otherwise advance through PHASE_SEQUENCE automatically.
        """
        if phase is not None:
            current_phase = phase
        else:
            current_phase, duration = self.PHASE_SEQUENCE[self._phase_idx]
            self._ticks_in_phase += 1
            if self._ticks_in_phase >= duration:
                self._phase_idx = (self._phase_idx + 1) % len(self.PHASE_SEQUENCE)
                self._ticks_in_phase = 0

        return self._sim.step(current_phase)

    def is_connected(self) -> bool:
        return self._connected

    def inject_fault(self, fault_type: str, severity: float = 0.7):
        self._sim.inject_fault(fault_type, severity)

    def clear_fault(self):
        self._sim.clear_fault()

    @property
    def phase_idx(self) -> int:
        return self._phase_idx

    @property
    def ticks_in_phase(self) -> int:
        return self._ticks_in_phase

    @property
    def simulator(self) -> EngineSimulator:
        """Direct access to underlying simulator (for state inspection)."""
        return self._sim


# ---------------------------------------------------------------------------
# Phase 2: CAN Bus Source (documented stub)
# ---------------------------------------------------------------------------

class CANBusTelemetrySource(TelemetrySource):
    """
    Phase 2 integration: reads real engine telemetry from SocketCAN.

    Setup on Linux UAV edge computer:
        ip link set can0 type can bitrate 250000
        ip link set can0 up
        pip install python-can

    The CAN_FRAME_MAP above documents the expected PGN layout from the
    Rotax 915iS FADEC / custom engine controller.
    """

    def __init__(self, channel: str = "can0", bitrate: int = 250000):
        self._channel = channel
        self._bitrate = bitrate
        self._bus = None
        self._decoder = None
        self._last_sample: Dict[str, Any] = {}
        self._connected = False
        self._try_connect()

    def _try_connect(self):
        try:
            import can  # python-can
            self._bus = can.interface.Bus(channel=self._channel, bustype="socketcan")
            self._connected = True
        except Exception as e:
            print(f"[CANBusTelemetrySource] CAN connection failed: {e}")
            self._connected = False

    def _decode_frame(self, msg) -> None:
        """Map a raw CAN frame to telemetry fields using CAN_FRAME_MAP."""
        pgn_hex = hex(msg.arbitration_id)
        if pgn_hex not in CAN_FRAME_MAP:
            return
        fields = CAN_FRAME_MAP[pgn_hex]
        data_int = int.from_bytes(msg.data, "big")
        for field_name, (start_bit, length, scale, offset, unit) in fields.items():
            raw = (data_int >> (64 - start_bit - length)) & ((1 << length) - 1)
            self._last_sample[field_name] = raw * scale + offset

    def read_sample(self, phase=None) -> Dict[str, Any]:
        raise NotImplementedError(
            "CANBusTelemetrySource.read_sample() requires python-can and a "
            "live SocketCAN interface. See CAN_FRAME_MAP for PGN layout."
        )

    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# Phase 3: Edge ECU Source (documented stub)
# ---------------------------------------------------------------------------

class EdgeECUTelemetrySource(TelemetrySource):
    """
    Phase 3 integration: reads from an onboard FADEC or ECU via
    serial (RS-232/RS-485) or UDP multicast from an edge computer.

    Expected JSON UDP packet (10 Hz):
    {
        "rpm": 4200, "cht_c": 152.3, "egt_c": 620.1,
        "oil_temp_c": 92.4, "oil_press_bar": 3.85,
        "fuel_flow_lph": 18.2, "vibration_g": 1.43,
        "battery_v": 14.1, "throttle": 0.55,
        "altitude_ft": 12000, "ambient_temp_c": 12.5,
        "degradation": 0.0003, "t_s": 12345.0,
        "phase": "cruise"
    }
    """

    def __init__(self, host: str = "239.0.0.1", port: int = 5005):
        self._host = host
        self._port = port
        self._sock = None
        self._connected = False

    def read_sample(self, phase=None) -> Dict[str, Any]:
        raise NotImplementedError(
            "EdgeECUTelemetrySource.read_sample() requires a live FADEC/ECU "
            "streaming JSON packets to UDP multicast. See class docstring."
        )

    def is_connected(self) -> bool:
        return self._connected


if __name__ == "__main__":
    src = SimulatedTelemetrySource(seed=42)
    print("SimulatedTelemetrySource connected:", src.is_connected())
    src.inject_fault("overheating_trend", 0.8)
    for i in range(3):
        s = src.read_sample()
        print(f"  tick {i}: rpm={s['rpm']:.0f} cht={s['cht_c']:.1f} fault={s['fault_active']}")
    src.clear_fault()
    print("Fault cleared. FAULT_TYPES:", FAULT_TYPES)
