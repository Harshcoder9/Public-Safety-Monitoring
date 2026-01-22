from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
from .analyzers.autoencoder import analyze_video_autoencoder
from .analyzers.optical_flow import analyze_video_optical_flow
from .analyzers.yolo_crowd_detector import analyze_video_yolo
from .models import RiskLevel
from .storage import AlertStore, LocationStore
from .stream_manager import stream_manager
from .realtime_analyzer_fast import analyze_video_realtime

app = FastAPI(title="Crowd Risk API", version="0.1.0")
store = AlertStore()
location_store = LocationStore()

cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://192.168.0.106:5173,https://public-safety-monitoring.vercel.app",
)
allow_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
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
    location: str = Form("Kandivali"),
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
        if analyzer_norm in {"yolo", "yolov8"}:
            result = analyze_video_yolo(
                video_path=str(out_path),
                process_fps=float(processFps) or 10.0,
                confidence_threshold=0.4,
                risk_window=10,
                high_confirm_frames=3,
                medium_confirm_frames=2,
                cooldown_frames=5,
                require_agreement=3,
                stop_on_high=True,
            )
            risk_level = result.risk_level
            event_time_seconds = result.event_time_seconds
            result_payload = {
                "analyzer": "yolo",
                "riskLevel": risk_level.value,
                "riskScore": float(result.risk_score),
                "confidence": float(result.confidence),
                "primaryCause": result.primary_cause,
                "supportingFactors": result.supporting_factors,
                "explanation": result.explanation,
                "eventTimeSeconds": event_time_seconds,
                "summary": {
                    "processFps": float(processFps) or 10.0,
                    "samples": len(result.samples),
                    "riskEngine": {
                        "window": 10,
                        "highConfirmFrames": 3,
                        "mediumConfirmFrames": 2,
                        "cooldownFrames": 5,
                        "requireAgreement": 3,
                        "signals": [
                            "densityChangeRate",
                            "motionSpeed",
                            "directionalChaos",
                            "spread",
                            "persistence",
                        ],
                    },
                },
                "samples": result.samples,
            }
        
        elif analyzer_norm in {"optical_flow", "flow", "optical"}:
            of = analyze_video_optical_flow(
                video_path=str(out_path),
                process_fps=float(processFps),
                min_consecutive=int(minConsecutive),
                z_low=float(zLow),
                z_med=float(zMed),
                z_high=float(zHigh),
                stop_on_high=True,
                high_confirm_frames=3,
                medium_confirm_frames=2,
                cooldown_frames=5,
                risk_window=7,
                require_agreement=2,
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
                "riskScore": float(of.risk_score),
                "confidence": float(max((s.confidence for s in of.samples), default=0.0)),
                "primaryCause": (first_alert.primary_cause if first_alert else ""),
                "supportingFactors": (list(first_alert.supporting_factors or []) if first_alert else []),
                "explanation": (first_alert.explanation if first_alert else ""),
                "eventTimeSeconds": event_time_seconds,
                "summary": {
                    "processFps": float(processFps),
                    "minConsecutive": int(minConsecutive),
                    "zLow": float(zLow),
                    "zMed": float(zMed),
                    "zHigh": float(zHigh),
                    "counts": of.counts,
                    "samples": len(of.samples),
                    "riskEngine": {
                        "window": 7,
                        "highConfirmFrames": 3,
                        "mediumConfirmFrames": 2,
                        "cooldownFrames": 5,
                        "requireAgreement": 2,
                        "signals": [
                            "densityChangeRate",
                            "motionSpeed",
                            "directionalChaos",
                            "persistence",
                            "spread",
                        ],
                    },
                },
                "samples": [
                    {
                        "riskLevel": s.risk_level.value,
                        "timeSeconds": s.time_seconds,
                        "meanFlowMag": s.mean_flow_mag,
                        "zScore": s.z_score,
                        "activeRatio": s.active_ratio,
                        "cause": s.cause,
                        "riskScore": s.risk_score_0_100,
                        "confidence": s.confidence,
                        "primaryCause": s.primary_cause,
                        "supportingFactors": list(s.supporting_factors or []),
                        "explanation": s.explanation,
                        "signals": dict(s.signals or {}),
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
                high_confirm_frames=3,
                medium_confirm_frames=2,
                cooldown_frames=5,
                risk_window=7,
                require_agreement=2,
            )
            result_payload = {
                "analyzer": "autoencoder",
                "riskLevel": ae.risk_level.value,
                "riskScore": ae.risk_score,
                "maxLoss": ae.max_loss,
                "meanLoss": ae.mean_loss,
                "eventTimeSeconds": ae.event_time_seconds,
                "confidence": float(ae.confidence),
                "primaryCause": str(ae.primary_cause),
                "supportingFactors": list(ae.supporting_factors or []),
                "explanation": str(ae.explanation),
                "sampleEverySeconds": float(sampleEverySeconds),
                "losses": ae.losses,
                "samples": ae.samples,
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid analyzer. Use 'yolo', 'optical_flow' or 'autoencoder'.")

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
            confidence=float(result_payload.get("confidence", 0.0) or 0.0),
            primary_cause=str(result_payload.get("primaryCause", "") or ""),
            supporting_factors=list(result_payload.get("supportingFactors") or []),
            explanation=str(result_payload.get("explanation", "") or ""),
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
            "confidence": alert_obj.confidence,
            "primary_cause": alert_obj.primary_cause,
            "supporting_factors": list(alert_obj.supporting_factors or []),
            "explanation": alert_obj.explanation,
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
    return {"locations": location_store.get_active_locations(max_age_seconds=60)}


@app.post("/api/location/stop")
def stop_location(userEmail: str = Form(...)):
    email = (userEmail or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Missing userEmail")
    location_store.remove_location(email)
    return {"status": "ok"}

@app.post("/api/streams/add")
async def add_stream(
    camera_id: str = Form(...),
    rtsp_url: str = Form(...),
    location: str = Form("Unknown"),
):
    def handle_stream_alert(data: Dict):
        if data.get("risk_level") in {"MEDIUM", "HIGH"}:
            try:
                store.create_alert(
                    user_email=f"camera_{camera_id}",
                    location=location,
                    risk_level=RiskLevel.from_str(data["risk_level"]),
                    risk_score=float(data.get("risk_score", 0.0)),
                    file_name=f"stream_{camera_id}",
                    event_time_seconds=float(data.get("time_seconds", 0.0)),
                    confidence=float(data.get("confidence", 0.0)),
                    primary_cause=str(data.get("primary_cause", "")),
                    supporting_factors=list(data.get("supporting_factors", [])),
                    explanation=str(data.get("explanation", "")),
                )
            except Exception as e:
                print(f"Failed to create alert from stream: {e}")
    
    success = stream_manager.add_stream(camera_id, rtsp_url, location, handle_stream_alert)
    if not success:
        raise HTTPException(status_code=400, detail="Stream already exists")
    
    return {"status": "ok", "camera_id": camera_id}

@app.delete("/api/streams/{camera_id}")
async def remove_stream(camera_id: str):
    success = stream_manager.remove_stream(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Stream not found")
    return {"status": "ok"}

@app.get("/api/streams/{camera_id}/frame")
async def get_stream_frame(camera_id: str):
    data = stream_manager.get_latest_frame(camera_id)
    if not data:
        raise HTTPException(status_code=404, detail="No frame available")
    return data

@app.get("/api/streams/all")
async def get_all_streams():
    return stream_manager.get_all_latest_frames()

async def event_generator():
    while True:
        await asyncio.sleep(0.5)
        frames = stream_manager.get_all_latest_frames()
        if frames:
            yield f"data: {json.dumps(frames)}\n\n"

@app.get("/api/streams/sse")
async def stream_sse():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/api/analyze/realtime")
async def analyze_realtime(
    file: UploadFile = File(...),
    processFps: float = Form(10.0),
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
    
    return StreamingResponse(
        analyze_video_realtime(
            video_path=str(out_path),
            process_fps=float(processFps),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
