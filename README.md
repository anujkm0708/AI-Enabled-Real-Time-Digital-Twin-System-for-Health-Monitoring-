# AeroTwin — AI-Enabled Real-Time Digital Twin System
### DRDO Problem Statement 26054 · Smart India Hackathon 2026

Real-time health monitoring, fault detection, and remaining-life estimation for **aero piston engines (Rotax-class, turbocharged 4-stroke)** used in MALE UAVs. The system is fully software-simulated and demonstrates the pipeline from sensor to cockpit decision.

> **Status:** v2.1 — calibrated, stateful digital twin with a Kalman state estimator. See [CHANGELOG.md](CHANGELOG.md) and [Known Limitations](#known-limitations).

---

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r aerotwin_ml/requirements.txt
pip3 install fastapi 'uvicorn[standard]' aiosqlite

# 2. Generate training data, calibrate the twin, and train the ML models
cd aerotwin_ml
PYTHONPATH=. python3 data/generate_dataset.py
PYTHONPATH=. python3 models/anomaly_detector.py
PYTHONPATH=. python3 models/fault_classifier.py
PYTHONPATH=. python3 models/rul_predictor.py
cd ..

# 3. Run the test suite (13 tests)
PYTHONPATH=aerotwin_ml:. python3 -m pytest tests -q

# 4. Start the backend
PYTHONPATH=aerotwin_ml:. python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 5. Open the dashboard
open frontend/index.html
# or: python3 -m http.server 8080 --directory frontend
```

Notes:
- `scipy`, `pytest` and `reportlab` are listed in `aerotwin_ml/requirements.txt`. `scipy` is needed by the twin calibration, `reportlab` by the PDF report.
- The dataset CSV, SQLite DB and compiled files are not stored in git. Step 2 regenerates the dataset.
- Twin calibration files (`twin_params.json`, `twin_thresholds.json`) are in `aerotwin_ml/models/artifacts/`. If missing, `calibrate_params()` and `calibrate_thresholds()` in `digital_twin.py` recreate them.

> **Demo flow:** start the backend, open the dashboard, click Start Mission, then inject a fault and watch the AI respond within about 10 s at high severity.

---

## System Architecture

```
+--------------------------------------------------------------------------+
|                         FRONTEND (Vanilla JS)                            |
|   Live Gauges . Trend Charts . Fault Buttons . Replay Mode               |
+----------------------------------+---------------------------------------+
                  WebSocket /ws/telemetry  +  REST /api/*
+----------------------------------v---------------------------------------+
|                        BACKEND (FastAPI / Python)                        |
|                                                                          |
| EngineSimulator --> _InputValidator --> DigitalTwin --> residuals        |
|   (1 Hz tick)        (range / rate /     (calibrated,      |             |
|   + fault inject      RPM-vs-model)       stateful)        v             |
|                                                    EngineStateEstimator  |
|                                                    (Kalman: bias, drift) |
|                                                             |            |
|                          Corroborator  <--------------------+            |
|                          (3 of last 5 ticks)                |            |
|                                                             v            |
|                     AI pipeline: AnomalyDetector, FaultClassifier,       |
|                     RULPredictor, SHAP FaultExplainer, HealthIndex       |
|                                                                          |
|                     SQLite  ->  /api/missions/{id}, report.pdf / .csv    |
+--------------------------------------------------------------------------+
                      Combined JSON -> WebSocket broadcast
```

### Component Map

| Component | File | Purpose |
|---|---|---|
| Engine Simulator | `aerotwin_ml/simulator/engine_simulator.py` | Physics-based synthetic telemetry and fault injection |
| Digital Twin | `aerotwin_ml/models/digital_twin.py` | Calibrated, stateful "healthy engine" reference model |
| Engine State Estimator | `aerotwin_ml/models/digital_twin.py` | Kalman filters on residuals; wear, cooling and lubrication indices |
| Input Validator | `aerotwin_ml/models/digital_twin.py` | Checks throttle/RPM inputs before the twin trusts them |
| Corroborator | `aerotwin_ml/models/digital_twin.py` | Persistence + multi-sensor cross-check |
| Anomaly Detector | `aerotwin_ml/models/anomaly_detector.py` | Isolation Forest on residuals (unsupervised) |
| Fault Classifier | `aerotwin_ml/models/fault_classifier.py` | Random Forest, 9 classes (healthy + 8 faults) |
| RUL Predictor | `aerotwin_ml/models/rul_predictor.py` | Gradient Boosting regression (hours remaining) |
| SHAP Explainer | `aerotwin_ml/models/explainability.py` | Plain-English reason per prediction |
| Health Index | `aerotwin_ml/models/health_index.py` | Subsystem health scores |
| Mission Report | `aerotwin_ml/models/mission_report.py` | PDF / CSV debrief export |
| Telemetry Source | `aerotwin_ml/models/telemetry_source.py` | Abstraction for simulator / CAN / ECU input |
| Backend API | `backend/main.py` | FastAPI: sim loop, WebSocket, REST, SQLite |
| Dashboard | `frontend/index.html` | Single-file vanilla JS + Chart.js SPA |
| Tests | `tests/test_digital_twin.py` | 13 pytest acceptance tests for the twin |

---

## Digital Twin Core (v2.1)

**Calibrated physics model.** `digital_twin.py` has no `simulator.*` imports at module level. Its parameters (`twin_params.json`) are fitted by `calibrate_params()` using `scipy.optimize.least_squares` on healthy simulator runs, including the time constants. CHT and oil temperature use a heat-balance ODE: `C·dT/dt = k_q·throttle·alt_factor − h·(T − T_amb)`.

**Stateful dynamics.** A `DigitalTwin` instance holds `cht_hat`, `oil_temp_hat` and `rpm_hat`, updated with exact first-order discretisation `α = 1 − exp(−dt/τ)`. One fresh instance is created per mission, and `reset()` clears state. The old stateless `compute_residuals()` remains as a deprecated wrapper.

**Engine state estimation.** `EngineStateEstimator` runs a 2-state Kalman filter (bias, drift) per residual channel and outputs `bias`, `drift` and `time_to_threshold_s`. Three health indices in [0, 1] are derived from the filtered states: `wear_index`, `cooling_efficiency` and `lube_health`. Filters are frozen while the input validator reports a fault.

**Input validation.** `_InputValidator` applies range and rate-of-change checks, and flags RPM when it disagrees with the twin's `rpm_hat` by more than 4σ for 3 or more consecutive ticks. On an RPM fault the twin substitutes its own estimate for the measured RPM.

**Calibrated thresholds.** `calibrate_thresholds(n_missions=30, k=4.0)` runs healthy missions, discards the first 60 s, and sets `threshold = max(k·σ, floor)` per channel (`twin_thresholds.json`). The `Corroborator` flags a channel only if it exceeds its threshold on at least 3 of the last 5 ticks.

---

## What Is Simulated vs. What Would Be Real

| Element | How it is faked | Real equivalent |
|---|---|---|
| Engine telemetry | Physics-informed ODE model with Rotax 914/915iS-like constants | CHT probes, oil transducers, EGT thermocouples, accelerometers |
| Mission flight phases | Hard-coded phase sequence | UAV autopilot / FADEC flight plan |
| Sensor noise | Gaussian noise | ADC quantization, EMI, vibration-induced drift |
| Fault injection | Mathematical perturbations of simulator state | Physical failures (oil leak, fouling, valve sticking) |
| Training data | Simulated missions, labeled automatically | Test-bench or fleet flight logs |
| Degradation / wear | Hidden slow-accumulating simulator variable | Borescope, oil analysis, overhaul records |

The ML models are genuinely trained (`.joblib` artifacts), SHAP runs a real `TreeExplainer`, and the corroboration logic is real multi-sensor cross-validation. **All data they learn from and are tested on comes from the simulator.**

---

## Results (measured on simulated data)

| Check | Result |
|---|---|
| Test suite | 13 / 13 pytest tests pass |
| Fault detection latency, severity 0.7, cruise, all 8 fault types, 5 seeds | 5–10 s |
| Fault detection latency, severity 0.3 | up to about 20 s |
| Healthy cruise, corroborator false flags after 60 s warm-up, 5 seeds | 0 |
| Input fault (RPM +800) | detected within 3 ticks (test_5) |
| `wear_index` vs simulator degradation over a long cruise | strong correlation, asserted in test_4 |

**Not yet re-validated:** the classifier F1 and RUL scores reported in earlier versions (F1 = 1.00, RUL MAE ~2.7 h, R² ~0.958) came from random row-level train/test splits. The RUL model also used the simulator's hidden `degradation` value as an input. Treat those figures as optimistic until they are re-measured with mission-grouped cross-validation and a leak-free RUL input (see Known Limitations).

---

## API Reference

### WebSocket

`ws://localhost:8000/ws/telemetry` — one JSON message per second:

```json
{
  "mission_id": "mission_1790535226",
  "tick_number": 42,
  "telemetry": { "phase": "cruise", "rpm": 4210, "cht_c": 148.2 },
  "residuals": { "cht_c": 1.2, "oil_press_bar": 0.04 },
  "anomaly_score": 0.12,
  "is_anomaly": false,
  "fault": {
    "predicted_fault": "none",
    "confidence": 0.99,
    "explanation_text": "No fault detected — all parameters within expected range."
  },
  "rul_hours": 245.5,
  "corroboration": { "verdict": "nominal", "severity_level": "none" },
  "health_indices": {},
  "engine_state": {
    "wear_index": 0.03,
    "cooling_efficiency": 0.98,
    "lube_health": 0.99,
    "drift": {}
  },
  "input_fault": { "flag": false, "channel": null, "reason": null },
  "recommendation": "continue"
}
```

`engine_state` and `input_fault` were added in v2.1. They are additive, and all earlier keys are unchanged. Values above are illustrative.

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/mission/start` | Start new simulated mission |
| POST | `/api/mission/stop` | Stop current mission |
| POST | `/api/fault/inject` | Body: `{"fault_type": "overheating_trend", "severity": 0.7}` |
| POST | `/api/fault/clear` | Remove injected fault |
| GET | `/api/missions` | List past missions |
| GET | `/api/missions/{id}` | Full telemetry history for replay |
| GET | `/api/missions/{id}/report.pdf` | One-page Mission Health Debrief PDF |
| GET | `/api/missions/{id}/report.csv` | Tick-by-tick CSV export |

**Fault types (8, matching DRDO PS 26054):** `sensor_drift`, `overheating_trend`, `oil_pressure_loss`, `misfire_condition`, `injector_abnormality`, `combustion_instability`, `cooling_degradation`, `lubrication_degradation`.

**Recommendations:** `abort` (severity critical or RUL < 1 h), `divert` (severity warning or RUL < 5 h), `monitor` (severity advisory), `continue` (all nominal). Recommendations are advisory only.

---

## ML Models

**Anomaly Detector (Isolation Forest)** — 7 residual features, trained unsupervised on healthy data only (contamination 2%). Outputs a score in [0, 1] and a binary flag.

**Fault Classifier (Random Forest, 300 trees)** — 9 features (7 residuals, throttle, RPM). Classes: `none` plus the 8 fault types above. Trained on labeled simulator runs; ramp-in samples with low severity are currently excluded from training.

**RUL Predictor (Gradient Boosting)** — 8 features (5 residuals, degradation, throttle, RPM); output capped at 500 h. **Known issue:** `degradation` is a simulator-internal value with no real-engine equivalent. The planned fix is to use the estimator's `wear_index` instead.

**Corroboration (rule-based)** — a channel counts only if it exceeds its calibrated threshold on at least 3 of the last 5 ticks. Only CHT flagged while oil temperature and vibration are normal gives `likely_instrumentation_fault` (advisory). One sensor gives `single_sensor_anomaly` (advisory). Two to three sensors give `corroborated_anomaly` (warning). Four or more give `corroborated_anomaly` (critical).

---

## Known Limitations

- **Simulator-only validation.** The twin is calibrated on simulator runs and everything is tested against the same simulator. No real engine data has been used.
- **RUL input leak.** The RUL model takes the simulator's hidden `degradation` value as an input. RUL metrics are therefore not meaningful yet.
- **Random-row splits.** The classifier and RUL predictor were evaluated with random row-level splits, so neighbouring ticks from one mission appear in both train and test. Scores are optimistic.
- **Weak faults under-tested.** The classifier trains only on higher-severity samples, so early, low-severity faults are not well covered.
- **Limited test conditions.** Detection tests use few seeds, one altitude and mostly one severity. Sensor dropout, heavier noise and wider ambient ranges are not covered yet.
- **Demo-grade backend.** CORS is open (`allow_origins=["*"]`) and there is no authentication.

---

## Integration Readiness (`models/telemetry_source.py`)

`TelemetrySource` decouples the diagnostics from the data acquisition hardware:

- **Phase 1 (working):** `SimulatedTelemetrySource` wraps `EngineSimulator` and produces 1 Hz synthetic telemetry.
- **Phase 2 (HIL):** `CANBusTelemetrySource` for Linux SocketCAN, with a `CAN_FRAME_MAP` of offsets, scales and PGNs for the 7 primary sensors (Rotax 915iS / SAE J1939 style).
- **Phase 3 (edge ECU):** `EdgeECUTelemetrySource` for FADEC / mission-computer frames over serial or UDP.

The CAN and ECU sources are interfaces prepared for later hardware work and have not been tested against real hardware.

---

## Deployment Roadmap — Simulation to Real Hardware

1. **Edge hardware:** read CAN / FADEC data (RPM, MAP, CHT, EGT, oil pressure) through SocketCAN on an ARM64 edge computer such as a Jetson Orin NX, running the backend as a systemd service and serving the dashboard over onboard WiFi.
2. **Calibration on real data:** log 10–20 nominal flights, refit twin parameters and thresholds with `calibrate_params()` / `calibrate_thresholds()` on real healthy data, and retrain the classifier on any real labeled fault events. Use dual-redundant CHT probes with the corroboration check.
3. **Fleet prognostics:** replace Gradient Boosting with a sequence model (e.g. LSTM) for RUL, move from SQLite to TimescaleDB or InfluxDB for fleet dashboards, and support a ground data link for beyond-line-of-sight operations.
4. **Certification path:** DGCA / MIL-HDBK-516 compliance, DO-178C software assurance, test-cell ground-truth validation, and human-in-the-loop operation with the GCS operator holding final authority.

---

## Project Structure

```
aero_twinengine/
├── aerotwin_ml/
│   ├── simulator/engine_simulator.py
│   ├── models/
│   │   ├── digital_twin.py            # twin, estimator, validator, corroborator
│   │   ├── anomaly_detector.py
│   │   ├── fault_classifier.py
│   │   ├── rul_predictor.py
│   │   ├── explainability.py
│   │   ├── health_index.py
│   │   ├── mission_report.py
│   │   ├── telemetry_source.py
│   │   └── artifacts/                 # .joblib models + twin_params/thresholds.json
│   ├── data/generate_dataset.py       # dataset CSV is generated, not stored in git
│   └── requirements.txt
├── backend/main.py                    # FastAPI application
├── frontend/index.html                # single-page dashboard
├── tests/test_digital_twin.py         # 13 pytest tests
├── CHANGELOG.md
└── README.md
```

---

## Why These Design Choices

**Isolation Forest for anomaly detection** — labeled fault data is scarce in the field. Training on healthy behavior only can flag faults the classifier has never seen.

**Random Forest for fault classification** — fast inference, well-behaved probabilities on tabular data, and direct compatibility with SHAP `TreeExplainer`.

**A calibrated, stateful twin** — a twin that tracks its own thermal and RPM state avoids false alarms during transients, and calibrated thresholds replace hand-picked ones.

**Corroboration as a safety layer** — if CHT spikes but oil temperature and vibration stay normal, the physics does not support a real overheating event. This prevents a false ABORT from a loose thermocouple wire.

---

*AeroTwin — DRDO PS 26054 / SIH 2026*
