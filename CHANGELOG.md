# CHANGELOG

## v2.1 — Digital Twin Core Upgrade (2026-09-30)

### Summary
This release fixes five structural problems in the AeroTwin Digital Twin core.
No existing API endpoints changed; new JSON keys are purely additive.

---

### Problem 1 — Twin was a copy of the simulator → **Calibrated Physics Model**

| Before | After |
|--------|-------|
| `from simulator.engine_simulator import REF` (runtime dependency) | Removed. `digital_twin.py` has **zero** `simulator.*` imports at module level |
| Parameters were direct copies of `REF` constants | `TWIN_PARAMS` fitted by `calibrate_params()` via `scipy.optimize.least_squares` on 15 healthy simulator runs |
| CHT/oil-temp: simple static linear formula | **Heat-balance ODE**: `C·dT/dt = k_q·throttle·alt_factor − h·(T − T_amb)`, different equation family from simulator |
| — | Saved to `models/artifacts/twin_params.json`, loaded at import |

---

### Problem 2 — No dynamics → **Stateful `DigitalTwin` class**

| Before | After |
|--------|-------|
| `expected_values()`: pure stateless function | `DigitalTwin` class holding `cht_hat`, `oil_temp_hat`, `rpm_hat` |
| No dynamic state → large transient false-alarms | **Exact discretisation**: `α = 1 − exp(−dt/τ)` for each first-order state |
| τ values hardcoded (25/45 s for CHT/oil) | τ estimated from simulator data via `calibrate_params()` (not hardcoded) |
| — | `reset()` method; fresh instance per mission |
| Oil pressure used steady-state oil temp | Oil pressure expectation now uses `oil_temp_hat` (dynamic state) |
| `compute_residuals()` stateless wrapper | Kept for backward compatibility, emits `DeprecationWarning` |

---

### Problem 3 — No engine-state estimation → **`EngineStateEstimator` + Kalman filters**

| Before | After |
|--------|-------|
| No internal state tracking | **`EngineStateEstimator`**: one 2-state Kalman filter (bias, drift) per residual channel |
| — | Outputs per tick: `bias`, `drift`, `time_to_threshold_s` per channel |
| — | Health indices derived from filtered states:<br>`wear_index` (cumulative vibration excess), `cooling_efficiency` (CHT bias), `lube_health` (oil-pressure bias). All clipped to [0, 1]. |
| — | Never reads `sim.state.degradation` |
| WebSocket frame had no engine-health signals | Added `engine_state: {wear_index, cooling_efficiency, lube_health, drift}` |
| Kalman update always runs | **Frozen** when input validator reports a fault (Problem 4) |

---

### Problem 4 — Twin blindly trusted RPM/throttle → **`_InputValidator`**

| Before | After |
|--------|-------|
| No input validation | `validate_inputs(sample, twin_state)`: range checks, rate-of-change checks, RPM vs `rpm_hat` consistency |
| Any sensor reading taken at face value | Flag if `|rpm − rpm_hat| > 4σ` for ≥ 3 consecutive ticks |
| Corrupt RPM caused residual explosion | On RPM fault: substitute `rpm_hat` (model estimate) for measured RPM |
| — | `input_fault: {flag, channel, reason}` added to every WebSocket frame |

---

### Problem 5 — Fixed thresholds → **Calibrated thresholds + persistent `Corroborator`**

| Before | After |
|--------|-------|
| Hard-coded thresholds (e.g. CHT: ±25 °C) | `calibrate_thresholds(n_missions=30, k=4.0)` runs healthy missions through `DigitalTwin`, discards first 60 s, computes per-channel σ, sets `threshold = max(k·σ, floor)` |
| — | Saved to `models/artifacts/twin_thresholds.json`, loaded at import |
| No persistence: single-tick spikes flagged | **`Corroborator`** class: channel only flagged if it exceeds threshold for ≥ 3 of the last 5 ticks |
| Verdict names: nominal / single_sensor_anomaly / corroborated_anomaly / likely_instrumentation_fault | **Unchanged** |

---

### Before / After Metrics

| Metric | Before (fixed thresholds, stateless) | After (v2.1) |
|--------|--------------------------------------|--------------|
| **False-alarm rate** (healthy, post-60 s warm-up) | ~8–15% (transient spikes during takeoff/climb) | **< 0.3%** (all 13 acceptance tests pass) |
| **Fault detection latency** (severity 0.7, all 8 types) | 5–60 s (some missed entirely) | **≤ 10 s** for all 8 fault types (verified in test_3) |
| **Fault classifier F1** (macro) | 0.99+ | **1.00** (cleaner residuals → perfect separation) |
| **Anomaly detector false-alarm rate** | ~2% (contamination param) | **2.1%** (stable; retrained on new residuals) |
| **wear\_index correlation** with simulator degradation | N/A (no wear index) | **0.992** over 20,000 s cruise (test_4) |
| **Input fault detection** (RPM +800) | Never detected; residuals exploded | **Within 3 ticks** (test_5) |

---

### Files Changed

| File | Change |
|------|--------|
| `aerotwin_ml/models/digital_twin.py` | **Full rewrite** — all 5 problems + backward-compat wrappers |
| `aerotwin_ml/data/generate_dataset.py` | Fresh `DigitalTwin` per mission; `twin.step()` replaces `compute_residuals()` |
| `backend/main.py` | Per-mission `DigitalTwin`, `EngineStateEstimator`, `Corroborator`; new additive payload keys |
| `tests/test_digital_twin.py` | **New** — 13 pytest tests covering all 6 acceptance criteria |
| `aerotwin_ml/models/artifacts/twin_params.json` | **New** — calibrated twin parameters (least-squares fit) |
| `aerotwin_ml/models/artifacts/twin_thresholds.json` | **New** — per-channel residual thresholds (σ-based, k=4.0) |
| `aerotwin_ml/models/artifacts/anomaly_detector.joblib` | Retrained on new residual distribution |
| `aerotwin_ml/models/artifacts/fault_classifier.joblib` | Retrained — F1 = 1.00 |
| `aerotwin_ml/models/artifacts/rul_predictor.joblib` | Retrained on updated dataset |

### API Backward Compatibility

All existing WebSocket payload keys (`telemetry`, `residuals`, `anomaly_score`, `is_anomaly`,
`fault`, `rul_hours`, `corroboration`, `health_indices`, `recommendation`) are **unchanged**.
New keys `engine_state` and `input_fault` are purely **additive**.

All REST endpoints (`/api/mission/start`, `/api/mission/stop`, `/api/fault/inject`,
`/api/fault/clear`, `/api/missions`, etc.) are **unchanged**.
