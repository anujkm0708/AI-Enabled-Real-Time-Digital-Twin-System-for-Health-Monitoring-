# AeroTwin — AI-Enabled Real-Time Digital Twin System
### DRDO Problem Statement 26054 · Smart India Hackathon 2026

Real-time health monitoring, fault prediction, and mission reliability enhancement for **aero piston engines (Rotax-class, turbocharged 4-stroke)** used in MALE UAVs — implemented as a fully software-simulated system demonstrating the complete AI pipeline from sensor to cockpit decision.

---

## Quick Start

```bash
# 1. Install all dependencies
pip3 install -r aerotwin_ml/requirements.txt
pip3 install fastapi 'uvicorn[standard]' aiosqlite

# 2. Generate training data and train all three ML models
cd aerotwin_ml
PYTHONPATH=. python3 data/generate_dataset.py
PYTHONPATH=. python3 models/anomaly_detector.py
PYTHONPATH=. python3 models/fault_classifier.py
PYTHONPATH=. python3 models/rul_predictor.py
cd ..

# 3. Start the backend
PYTHONPATH=. python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Open the dashboard
open frontend/index.html
# (or serve it: python3 -m http.server 8080 --directory frontend)
```

> **Demo flow:** Start backend, open dashboard, click Start Mission, watch live gauges. Click any fault injection button and observe the AI respond within ~20 seconds as the fault ramps in.

---

## System Architecture

```
+------------------------------------------------------------------+
|                        FRONTEND (Vanilla JS)                     |
|  Live Gauges . Trend Charts . Fault Buttons . Replay Mode        |
+-------------------------------+---------------------------------++
                 WebSocket /ws/telemetry  +  REST /api/*
+-------------------------------v----------------------------------+
|                     BACKEND (FastAPI / Python)                   |
|                                                                  |
|  +--------------+   +----------------+   +------------------+   |
|  |EngineSimulat-|-->| Digital Twin   |-->|   AI Pipeline    |   |
|  |or (1 Hz tick)|   | expected_values|   |                  |   |
|  |              |   | compute_resids |   | AnomalyDetector  |   |
|  | Fault Inject |   | corroboration_ |   | FaultClassifier  |   |
|  | /api/fault/* |   | _check()       |   | RULPredictor     |   |
|  +--------------+   +----------------+   | FaultExplainer   |   |
|                                          |  (SHAP)          |   |
|                              SQLite      +--------+---------+   |
|                              /api/missions/{id}   |             |
+---------------------------------------------------v-------------+
                          Combined JSON -> WebSocket broadcast
```

### Component Map

| Component | File | Purpose |
|---|---|---|
| Engine Simulator | `aerotwin_ml/simulator/engine_simulator.py` | Physics-based synthetic telemetry |
| Digital Twin | `aerotwin_ml/models/digital_twin.py` | Independent "healthy engine" reference model |
| Anomaly Detector | `aerotwin_ml/models/anomaly_detector.py` | Isolation Forest on residuals (unsupervised) |
| Fault Classifier | `aerotwin_ml/models/fault_classifier.py` | Random Forest multi-class fault ID |
| RUL Predictor | `aerotwin_ml/models/rul_predictor.py` | Gradient Boosting regression (hours remaining) |
| SHAP Explainer | `aerotwin_ml/models/explainability.py` | Plain-English reason text per prediction |
| Backend API | `backend/main.py` | FastAPI: sim loop, WebSocket, REST, SQLite |
| Dashboard | `frontend/index.html` | Single-file vanilla JS + Chart.js SPA |

---

## What Is Simulated vs. What Would Be Real

### Simulated (Software-Only)

| Element | How It's Faked | Real Equivalent |
|---|---|---|
| Engine telemetry | Physics-informed ODE model with Rotax 914/915iS thermal constants | Physical sensors (CHT probes, oil transducers, EGT thermocouples, accelerometers) |
| Mission flight phases | Hard-coded phase sequence with realistic durations | UAV autopilot flight plan via MAVLink/FADEC |
| Sensor noise | Gaussian noise fitted to real sensor spec sheets | ADC quantization, cable EMI, vibration-induced drift |
| Fault injection | Mathematical perturbations applied to simulator state | Physical failures (oil leak, spark plug fouling, valve sticking) |
| Training data | 50 simulated missions x 1550 ticks = 77,500 labeled samples | Logged flight data from test-bench or operational UAV fleet |
| Degradation / wear | Slow accumulation variable driving RUL label | Borescope inspections, oil analysis, overhaul records |

### AI/ML — Genuinely Real

- All three ML models trained from scratch during setup; saved as `.joblib` artifacts
- SHAP explanations run real TreeExplainer on the actual trained Random Forest
- Corroboration logic implements genuine multi-sensor cross-validation
- RUL regression: **MAE ~2.7 hours, R2 ~0.958** on held-out test data

### Architectural Separation

`digital_twin.py` is intentionally a *separate, simplified physics model* from the simulator. It does not have privileged access to simulator internals — it only sees what a real edge computer would see (throttle command, altitude, ambient temp, RPM). The residual = actual - expected is a genuine anomaly signal, not a tautological comparison.

---

## API Reference

### WebSocket

`ws://localhost:8000/ws/telemetry` — one JSON message per second:

```json
{
  "mission_id": "mission_1790535226",
  "tick_number": 42,
  "telemetry": { "phase": "cruise", "rpm": 4210, "cht_c": 148.2, ... },
  "residuals": { "cht_c": 1.2, "oil_press_bar": 0.04, ... },
  "anomaly_score": 0.12,
  "is_anomaly": false,
  "fault": {
    "predicted_fault": "none",
    "confidence": 0.99,
    "explanation_text": "No fault detected — all parameters within expected range."
  },
  "rul_hours": 245.5,
  "corroboration": { "verdict": "nominal", "severity_level": "none" },
  "recommendation": "continue"
}
```

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/mission/start` | Start new simulated mission |
| POST | `/api/mission/stop` | Stop current mission |
| POST | `/api/fault/inject` | Body: `{"fault_type": "overheating_trend", "severity": 0.7}` |
| POST | `/api/fault/clear` | Remove injected fault |
| GET | `/api/missions` | List all past missions |
| GET | `/api/missions/{id}` | Full telemetry history for replay |
| GET | `/api/missions/{id}/report.pdf` | Download one-page Mission Health Debrief PDF |
| GET | `/api/missions/{id}/report.csv` | Download full tick-by-tick CSV telemetry export |

**Fault types (8 classes matching DRDO PS 26054):**
- `sensor_drift`
- `overheating_trend`
- `oil_pressure_loss`
- `misfire_condition`
- `injector_abnormality`
- `combustion_instability`
- `cooling_degradation`
- `lubrication_degradation`

**Mission recommendations:**
- `abort` → severity critical OR rul_hours < 1
- `divert` → severity warning OR rul_hours < 5
- `monitor` → severity advisory
- `continue` → all nominal

---

## ML Models

### Anomaly Detector (Isolation Forest)
- Inputs: 7 residual features
- Training: Unsupervised on healthy data only (contamination=2%)
- Output: Anomaly score [0,1] + binary flag

### Fault Classifier (Random Forest, 300 trees)
- Inputs: 7 residuals + throttle + RPM = 9 features
- Classes: `none`, `sensor_drift`, `overheating_trend`, `oil_pressure_loss`, `misfire_condition`
- Performance: F1 = 1.00 on test set
- Top features: oil pressure residual (27%), CHT residual (27%), vibration residual (16%)

### RUL Predictor (Gradient Boosting)
- Inputs: 5 residuals + degradation + throttle + RPM = 8 features
- Output: Estimated remaining hours (capped at 500)
- Performance: MAE ~2.7 hrs, R2 ~0.958

### Corroboration Check (Rule-Based)
- Only CHT flagged + oil_temp/vibration normal → `likely_instrumentation_fault` (advisory)
- 1 sensor → `single_sensor_anomaly` (advisory)
- 2-3 sensors → `corroborated_anomaly` (warning)
- 4+ sensors → `corroborated_anomaly` (critical)

---

## Integration Readiness & Telemetry Abstraction (`models/telemetry_source.py`)

AeroTwin defines a unified abstraction layer (`TelemetrySource`) decoupling the AI diagnostics and FastAPI backend from the physical data acquisition hardware:

```
               ┌──────────────────────────┐
               │ TelemetrySource Interface│
               └─────────────┬────────────┘
                             │
     ┌───────────────────────┼────────────────────────┐
     │                       │                        │
┌────▼────────────────┐ ┌────▼────────────────┐ ┌─────▼────────────────┐
│SimulatedTelemetry-  │ │CANBusTelemetrySource│ │EdgeECUTelemetrySource│
│Source (Phase 1: Now)│ │(Phase 2: SocketCAN) │ │(Phase 3: FADEC / UDP)│
└─────────────────────┘ └─────────────────────┘ └──────────────────────┘
```

- **Phase 1 (Active/Working):** `SimulatedTelemetrySource` wraps `EngineSimulator`, auto-advancing mission phases and generating 1 Hz synthetic telemetry.
- **Phase 2 (Hardware-in-the-Loop):** `CANBusTelemetrySource` interfaces directly with Linux `SocketCAN` (`can0` interface). `models/telemetry_source.py` includes `CAN_FRAME_MAP`, which specifies the exact bit offsets, lengths, scale factors, and PGNs for all 7 primary engine sensors compatible with Rotax 915iS / SAE J1939.
- **Phase 3 (Edge ECU / UDP):** `EdgeECUTelemetrySource` accepts high-rate telemetry frames from onboard FADEC / mission computers over serial/UDP without altering downstream digital twin logic.

---

## Deployment Roadmap — Simulation to Real Hardware

### Phase 1 — Edge Hardware Integration

**CAN Bus / SocketCAN**

Real MALE UAV engines expose data via CAN (SAE J1939 or proprietary FADEC CAN). Replace `EngineSimulator.step()`:

```python
import can
bus = can.interface.Bus(channel='can0', bustype='socketcan')
while True:
    msg = bus.recv()
    sample = decode_j1939_frame(msg)   # maps PGN -> {rpm, cht_c, ...}
    pipeline.process_telemetry_tick(sample)
```

**FADEC Integration (Rotax 915iS)**
The FADEC outputs: RPM via Hall-effect sensors, MAP, CHT via Type-K thermocouples, EGT probes (one per cylinder), oil pressure transducer (0-10 bar, 4-20 mA). All map 1-to-1 to simulator fields with unit conversions.

**Edge Computing Platform**
Recommended: NVIDIA Jetson Orin NX (16 GB) or ARM64 SBC with:
- SocketCAN interface (MCP2515 or USB-CAN adapter)
- Ubuntu 22.04 + Python 3.11
- All ML models: <5 ms inference on CPU
- FastAPI backend as a systemd service
- Dashboard served via onboard WiFi AP to GCS tablet

### Phase 2 — Sensor Calibration and Model Adaptation

**Fine-Tuning on Real Hardware**
1. Run 10-20 nominal flights, log raw telemetry and residuals
2. Fit new StandardScaler on real healthy-engine residuals
3. Retrain Fault Classifier on any real labeled fault events

**Sensor Redundancy**
Real installations use dual-redundant CHT probes (per EASA CS-23). The corroboration check is designed for this — pass the average of both probes; add cross-probe delta as an extra residual feature.

### Phase 3 — Fleet-Level Prognostics Upgrade

**LSTM for RUL**
Replace Gradient Boosting with an LSTM trained on full degradation trajectories, validated against NASA C-MAPSS (turbofan) dataset (methodology from NASA/CR-2007-214341). Captures temporal patterns the per-tick model misses.

**Fleet Telemetry**
Replace SQLite with TimescaleDB or InfluxDB for:
- Multi-engine fleet health dashboard
- Anomaly correlation across tail numbers
- Automated maintenance scheduling from fleet-aggregated RUL predictions

**Ground Data Link**
For beyond-LOS MALE UAV operations:
- Stream compressed telemetry (~500 bytes/sec) over satellite (Iridium/VSAT)
- Ground AeroTwin backend processes full pipeline
- On-board edge runs lightweight anomaly-only check; full classification deferred to ground when link is available

### Phase 4 — Certification Path

For operational deployment in defence aviation:
- DGCA / MIL-HDBK-516 compliance for avionics software
- DO-178C Level C software assurance
- Ground-truth validation against engine test-cell runs
- Human-in-the-loop: AI recommendations are advisory only; final authority remains with GCS operator

---

## Project Structure

```
aero_twinengine/
├── aerotwin_ml/                    # ML package
│   ├── simulator/engine_simulator.py
│   ├── models/
│   │   ├── digital_twin.py
│   │   ├── anomaly_detector.py
│   │   ├── fault_classifier.py
│   │   ├── rul_predictor.py
│   │   ├── explainability.py
│   │   └── artifacts/              # Trained .joblib files (generated)
│   ├── data/
│   │   ├── generate_dataset.py
│   │   └── aerotwin_dataset.csv    # 77,500 samples (generated)
│   └── requirements.txt
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt
│   └── flight_data.db              # SQLite (generated)
├── frontend/
│   └── index.html                  # Single-page dashboard
└── README.md
```

---

## Why These Design Choices

**Isolation Forest for anomaly detection** — In real deployment, labeled fault data is scarce. Isolation Forest trains on normal engine behavior only, enabling detection of novel faults not in the training set — the dominant failure mode in field deployments.

**Random Forest for fault classification** — <1 ms inference, well-calibrated probabilities, resistant to overfitting on tabular data, directly compatible with SHAP TreeExplainer.

**The corroboration check as a safety layer** — A pure ML pipeline would occasionally flag single-sensor noise as a critical fault. The rule-based corroboration check sits between residuals and ML as a sanity filter: if CHT spikes 120°C but oil temperature and vibration are normal, the thermal physics simply don't support a real overheating event — preventing false ABORT calls from a loose thermocouple wire.

---

*AeroTwin — DRDO PS 26054 / SIH 2026*
