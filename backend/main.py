"""
AeroTwin Backend — FastAPI
Changes v2:
  - SimulatedTelemetrySource replaces direct EngineSimulator usage
  - health_index.py wired into every tick
  - 8-fault taxonomy (FAULT_TYPES from updated simulator)
  - PDF + CSV report endpoints
"""
import sys, os, asyncio, json, time, io
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import aiosqlite

AEROTWIN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aerotwin_ml')
if AEROTWIN_PATH not in sys.path:
    sys.path.insert(0, AEROTWIN_PATH)

from models.telemetry_source import SimulatedTelemetrySource
from simulator.engine_simulator import FlightPhase, FAULT_TYPES
from models.digital_twin import compute_residuals, corroboration_check
from models.anomaly_detector import AnomalyDetector
from models.fault_classifier import FaultClassifier, FEATURE_COLS as FC_FEATURE_COLS
from models.rul_predictor import RULPredictor
from models.explainability import FaultExplainer
from models.health_index import compute_health_indices
from models.mission_report import export_pdf, export_csv

app = FastAPI(title="AeroTwin Backend v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ────────────────────────────────────────────────────────────

class MissionState:
    def __init__(self):
        self.is_running: bool = False
        self.mission_id: Optional[str] = None
        self.source: Optional[SimulatedTelemetrySource] = None
        self.tick_number: int = 0
        self.phase_idx: int = 0
        self.ticks_in_phase: int = 0
        self.task: Optional[asyncio.Task] = None

mission_state = MissionState()
active_websockets: List[WebSocket] = []

DB_PATH = os.path.join(os.path.dirname(__file__), "flight_data.db")
MODELS_DIR = os.path.join(AEROTWIN_PATH, "models", "artifacts")

detector = None
clf      = None
rul_model = None
explainer = None

# ── Model loading ────────────────────────────────────────────────────────────

def load_models():
    global detector, clf, rul_model, explainer
    try:
        detector  = AnomalyDetector.load(os.path.join(MODELS_DIR, "anomaly_detector.joblib"))
        clf       = FaultClassifier.load(os.path.join(MODELS_DIR, "fault_classifier.joblib"))
        rul_model = RULPredictor.load(os.path.join(MODELS_DIR, "rul_predictor.joblib"))
        explainer = FaultExplainer(clf)
        print(f"Models loaded. Fault classes: {list(clf.classes_)}")
    except Exception as e:
        print(f"Warning: Models could not be loaded: {e}")

# ── Database ─────────────────────────────────────────────────────────────────

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS missions (
                id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                status TEXT
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS telemetry_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT,
                tick_number INTEGER,
                timestamp TEXT,
                data_json TEXT
            )
        ''')
        await db.commit()

async def save_tick_to_db(mission_id: str, tick_number: int, timestamp: str, data_json: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO telemetry_ticks (mission_id, tick_number, timestamp, data_json) VALUES (?,?,?,?)",
            (mission_id, tick_number, timestamp, data_json)
        )
        await db.commit()

async def update_mission_status(mission_id: str, status: str):
    async with aiosqlite.connect(DB_PATH) as db:
        if status != 'running':
            await db.execute(
                "UPDATE missions SET status=?, end_time=? WHERE id=?",
                (status, datetime.utcnow().isoformat(), mission_id)
            )
        else:
            await db.execute("UPDATE missions SET status=? WHERE id=?", (status, mission_id))
        await db.commit()

async def fetch_mission_ticks(mission_id: str) -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT data_json FROM telemetry_ticks WHERE mission_id=? ORDER BY tick_number ASC",
            (mission_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [json.loads(r["data_json"]) for r in rows]

# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_numpy(obj):
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

async def broadcast(payload: dict) -> str:
    clean = convert_numpy(payload)
    msg = json.dumps(clean)
    dead = []
    for ws in active_websockets:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_websockets.remove(ws)
    return msg

# ── Simulation loop ───────────────────────────────────────────────────────────

PHASES = [
    (FlightPhase.GROUND,  60),
    (FlightPhase.TAKEOFF, 30),
    (FlightPhase.CLIMB,   300),
    (FlightPhase.CRUISE,  900),
    (FlightPhase.DESCENT, 200),
    (FlightPhase.LANDING, 60),
]

AD_COLS  = ['resid_cht_c','resid_egt_c','resid_oil_temp_c','resid_oil_press_bar',
            'resid_fuel_flow_lph','resid_vibration_g','resid_battery_v']
RUL_COLS = ['resid_cht_c','resid_egt_c','resid_oil_temp_c','resid_oil_press_bar',
            'resid_vibration_g','degradation','throttle','rpm']

async def simulation_loop():
    try:
        while mission_state.is_running and mission_state.phase_idx < len(PHASES):
            current_phase, duration = PHASES[mission_state.phase_idx]

            # 1. Read telemetry from SimulatedTelemetrySource
            sample = mission_state.source.read_sample(phase=current_phase)

            # 2. Digital twin residuals + corroboration
            residuals = compute_residuals(sample)
            corrob    = corroboration_check(residuals)

            # 3. Build feature DataFrame
            row_dict = {
                'resid_cht_c':       residuals['cht_c'],
                'resid_egt_c':       residuals['egt_c'],
                'resid_oil_temp_c':  residuals['oil_temp_c'],
                'resid_oil_press_bar': residuals['oil_press_bar'],
                'resid_fuel_flow_lph': residuals['fuel_flow_lph'],
                'resid_vibration_g': residuals['vibration_g'],
                'resid_battery_v':   residuals['battery_v'],
                'throttle':          sample['throttle'],
                'rpm':               sample['rpm'],
                'degradation':       sample['degradation'],
            }
            df_row = pd.DataFrame([row_dict])

            # 4. Anomaly detection
            anomaly_score = 0.0
            is_anomaly    = False
            if detector and detector.model:
                X   = df_row[AD_COLS].values
                Xs  = detector.scaler.transform(X)
                raw = detector.model.decision_function(Xs)[0]
                anomaly_score = float(max(0.0, min(1.0, 0.5 - raw)))
                is_anomaly    = bool(detector.predict_is_anomaly(df_row[AD_COLS])[0])

            # 5. Fault classification + SHAP explanation
            fault_info = {
                "predicted_fault":  "none",
                "confidence":       1.0,
                "top_reasons":      [],
                "explanation_text": "No fault detected — all monitored parameters are within expected range.",
            }
            if clf:
                try:
                    fault_info = explainer.explain(df_row[FC_FEATURE_COLS], top_k=3)
                except Exception:
                    pred  = clf.predict(df_row[FC_FEATURE_COLS])[0]
                    probs = clf.predict_proba(df_row[FC_FEATURE_COLS])[0]
                    fault_info = {
                        "predicted_fault": str(pred),
                        "confidence":      float(max(probs)),
                        "top_reasons":     [],
                        "explanation_text": "",
                    }

            # 6. RUL prediction
            rul_hours = 100.0
            if rul_model:
                rul_hours = float(rul_model.predict(df_row[RUL_COLS])[0])

            # 7. Composite Health Index (NEW)
            fault_label      = fault_info.get("predicted_fault", "none")
            fault_confidence = float(fault_info.get("confidence", 0.0))
            health_indices   = compute_health_indices(
                residuals, anomaly_score, fault_label, fault_confidence
            )

            # 8. Mission recommendation
            sev = corrob.get('severity_level', 'none')
            if sev == 'critical' or rul_hours < 1:
                recommendation = 'abort'
            elif sev == 'warning' or rul_hours < 5:
                recommendation = 'divert'
            elif sev == 'advisory':
                recommendation = 'monitor'
            else:
                recommendation = 'continue'

            timestamp = datetime.utcnow().isoformat()

            payload = {
                "mission_id":     mission_state.mission_id,
                "tick_number":    mission_state.tick_number,
                "timestamp":      timestamp,
                "telemetry":      sample,
                "residuals":      residuals,
                "anomaly_score":  anomaly_score,
                "is_anomaly":     is_anomaly,
                "fault":          fault_info,
                "rul_hours":      rul_hours,
                "corroboration":  corrob,
                "health_indices": health_indices,
                "recommendation": recommendation,
            }

            msg_str = await broadcast(payload)
            await save_tick_to_db(
                mission_state.mission_id,
                mission_state.tick_number,
                timestamp,
                msg_str,
            )

            mission_state.tick_number     += 1
            mission_state.ticks_in_phase  += 1
            if mission_state.ticks_in_phase >= duration:
                mission_state.phase_idx      += 1
                mission_state.ticks_in_phase  = 0

            await asyncio.sleep(1.0)

        if mission_state.is_running:
            await update_mission_status(mission_state.mission_id, "completed")
        mission_state.is_running = False

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        print(f"Simulation loop error: {e}\n{traceback.format_exc()}")
        if mission_state.mission_id:
            await update_mission_status(mission_state.mission_id, "error")
        mission_state.is_running = False

# ── Startup / WebSocket ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    await init_db()
    load_models()

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

# ── REST — Mission control ────────────────────────────────────────────────────

class FaultInjectRequest(BaseModel):
    fault_type: str
    severity:   float = 0.7

@app.post("/api/mission/start")
async def start_mission():
    if mission_state.is_running:
        raise HTTPException(400, "Mission already running")
    mission_id = f"mission_{int(time.time())}"
    mission_state.is_running     = True
    mission_state.mission_id     = mission_id
    mission_state.source         = SimulatedTelemetrySource(
        ambient_temp_c=25.0, altitude_ft=10000.0, dt_s=1.0
    )
    mission_state.tick_number    = 0
    mission_state.phase_idx      = 0
    mission_state.ticks_in_phase = 0
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO missions (id, start_time, status) VALUES (?,?,?)",
            (mission_id, datetime.utcnow().isoformat(), "running")
        )
        await db.commit()
    mission_state.task = asyncio.create_task(simulation_loop())
    return {"status": "started", "mission_id": mission_id}

@app.post("/api/mission/stop")
async def stop_mission():
    if not mission_state.is_running:
        raise HTTPException(400, "No mission running")
    mission_state.is_running = False
    if mission_state.task:
        mission_state.task.cancel()
    await update_mission_status(mission_state.mission_id, "stopped")
    return {"status": "stopped", "mission_id": mission_state.mission_id}

@app.post("/api/fault/inject")
async def inject_fault(req: FaultInjectRequest):
    if not mission_state.is_running or not mission_state.source:
        raise HTTPException(400, "No mission running")
    if req.fault_type not in FAULT_TYPES:
        raise HTTPException(400, f"Invalid fault type. Valid: {FAULT_TYPES}")
    mission_state.source.inject_fault(req.fault_type, severity=req.severity)
    return {"status": "fault_injected", "type": req.fault_type, "severity": req.severity}

@app.post("/api/fault/clear")
async def clear_fault():
    if not mission_state.is_running or not mission_state.source:
        raise HTTPException(400, "No mission running")
    mission_state.source.clear_fault()
    return {"status": "fault_cleared"}

# ── REST — Mission history ────────────────────────────────────────────────────

@app.get("/api/missions")
async def list_missions():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute('''
            SELECT m.id, m.start_time, m.status, count(t.id) as total_ticks
            FROM missions m
            LEFT JOIN telemetry_ticks t ON m.id = t.mission_id
            GROUP BY m.id ORDER BY m.start_time DESC
        ''') as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]

@app.get("/api/missions/{mission_id}")
async def get_mission_data(mission_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM missions WHERE id=?", (mission_id,)) as cur:
            mission = await cur.fetchone()
        if not mission:
            raise HTTPException(404, "Mission not found")
        async with db.execute(
            "SELECT data_json FROM telemetry_ticks WHERE mission_id=? ORDER BY tick_number ASC",
            (mission_id,)
        ) as cur:
            ticks = await cur.fetchall()
    return {
        "mission":   dict(mission),
        "telemetry": [json.loads(r["data_json"]) for r in ticks],
    }

# ── REST — Reports (PDF + CSV) ────────────────────────────────────────────────

@app.get("/api/missions/{mission_id}/report.pdf")
async def mission_report_pdf(mission_id: str):
    ticks = await fetch_mission_ticks(mission_id)
    if not ticks:
        raise HTTPException(404, f"No ticks found for mission {mission_id}")
    try:
        pdf_bytes = export_pdf(mission_id, ticks)
    except ImportError as e:
        raise HTTPException(500, f"PDF export requires reportlab: {e}")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{mission_id}_report.pdf"'},
    )

@app.get("/api/missions/{mission_id}/report.csv")
async def mission_report_csv(mission_id: str):
    ticks = await fetch_mission_ticks(mission_id)
    if not ticks:
        raise HTTPException(404, f"No ticks found for mission {mission_id}")
    csv_bytes = export_csv(ticks)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{mission_id}_telemetry.csv"'},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
