from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .analyzers.autoencoder import analyze_video_autoencoder
from .analyzers.optical_flow import analyze_video_optical_flow
from .models import RiskLevel
from .storage import AlertStore, LocationStore

app = FastAPI(title="Crowd Risk API", version="0.1.0")
store = AlertStore()
location_store = LocationStore()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    userEmail: str = Form(...),
    location: str = Form("Kandavli"),
    includeLosses: bool = Form(False),
    analyzer: str = Form("optical_flow"),
    sampleEverySeconds: float = Form(0.2),
    thresholdLow: float = Form(0.0008),
    thresholdMedium: float = Form(0.0012),
    thresholdHigh: float = Form(0.0016),
    processFps: float = Form(5.0),
    minConsecutive: int = Form(1),
    zLow: float = Form(3.0),
    zMed: float = Form(5.0),
    zHigh: float = Form(7.0),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    safe_name = Path(file.filename).name
    out_path = UPLOAD_DIR / f"{int(datetime.now().timestamp())}_{safe_name}"

    try:
        contents = await file.read()
        out_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    analyzer_norm = (analyzer or "").strip().lower()

    try:
        if analyzer_norm in {"optical_flow", "flow", "optical"}:
            of = analyze_video_optical_flow(
                video_path=str(out_path),
                process_fps=float(processFps),
                min_consecutive=int(minConsecutive),
                z_low=float(zLow),
                z_med=float(zMed),
                z_high=float(zHigh),
                stop_on_high=True,
            )
            first_alert = next(
                (s for s in of.samples if s.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}),
                None,
            )
            risk_level = of.risk_level
            event_time_seconds = float(first_alert.time_seconds) if first_alert else 0.0
            result_payload = {
                "analyzer": "optical_flow",
                "riskLevel": risk_level.value,
                "riskScore": float(max((s.z_score for s in of.samples), default=0.0)),
                "eventTimeSeconds": event_time_seconds,
                "summary": {
                    "processFps": float(processFps),
                    "minConsecutive": int(minConsecutive),
                    "zLow": float(zLow),
                    "zMed": float(zMed),
                    "zHigh": float(zHigh),
                    "counts": of.counts,
                    "samples": len(of.samples),
                },
                "samples": [
                    {
                        "riskLevel": s.risk_level.value,
                        "timeSeconds": s.time_seconds,
                        "meanFlowMag": s.mean_flow_mag,
                        "zScore": s.z_score,
                        "activeRatio": s.active_ratio,
                        "cause": s.cause,
                    }
                    for s in of.samples
                ],
            }

        elif analyzer_norm in {"autoencoder", "ae"}:
            ae = analyze_video_autoencoder(
                video_path=str(out_path),
                include_losses=bool(includeLosses),
                sample_every_seconds=float(sampleEverySeconds),
                threshold_low=float(thresholdLow),
                threshold_medium=float(thresholdMedium),
                threshold_high=float(thresholdHigh),
                stop_on_high=True,
            )
            result_payload = {
                "analyzer": "autoencoder",
                "riskLevel": ae.risk_level.value,
                "riskScore": ae.risk_score,
                "maxLoss": ae.max_loss,
                "meanLoss": ae.mean_loss,
                "eventTimeSeconds": ae.event_time_seconds,
                "sampleEverySeconds": float(sampleEverySeconds),
                "losses": ae.losses,
                "samples": ae.samples,
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid analyzer. Use 'optical_flow' or 'autoencoder'.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    alert_created = False
    alert = None

    if result_payload["riskLevel"] in {RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}:
        alert_created = True
        alert_obj = store.create_alert(
            user_email=userEmail,
            location=location or "Kandavli",
            risk_level=RiskLevel(result_payload["riskLevel"]),
            risk_score=float(result_payload.get("riskScore", 0.0)),
            file_name=safe_name,
            event_time_seconds=float(result_payload.get("eventTimeSeconds", 0.0)),
        )
        alert = {
            "id": alert_obj.id,
            "created_at": alert_obj.created_at.isoformat(),
            "user_email": alert_obj.user_email,
            "location": alert_obj.location,
            "risk_level": alert_obj.risk_level,
            "risk_score": alert_obj.risk_score,
            "file_name": alert_obj.file_name,
            "event_time_seconds": alert_obj.event_time_seconds,
        }

    return {
        "userEmail": userEmail,
        "location": location or "Kandivali",
        **result_payload,
        "alertCreated": alert_created,
        "alert": alert,
    }

@app.get("/api/alerts")
def list_alerts(includeAcknowledged: bool = True):
    return {"alerts": store.list_alerts(include_acknowledged=includeAcknowledged)}

@app.post("/api/alerts/{alert_id}/ack")
def acknowledge(alert_id: str):
    updated = store.acknowledge(alert_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return updated

@app.post("/api/location")
def update_location(
    userEmail: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    loc = location_store.update_location(userEmail, latitude, longitude)
    return {
        "status": "ok", 
        "location": {
            "user_email": loc.user_email,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "timestamp": loc.timestamp.isoformat()
        }
    }

@app.get("/api/locations")
def get_locations():
    return {"locations": location_store.get_active_locations()}
