from __future__ import annotations

import asyncio
import base64
import cv2
import json
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from .analyzers.yolo_crowd_detector import _load_yolo_model

_executor = ThreadPoolExecutor(max_workers=6)
_detection_cache = {"count": 0, "frame_num": 0}

def _fast_detect_and_encode(frame, model, confidence_threshold, quality, frame_idx, detect_every):
    global _detection_cache
    
    should_detect = (frame_idx % detect_every) == 0
    
    if should_detect:
        small = cv2.resize(frame, (480, 270))
        results = model.track(small, persist=True, conf=confidence_threshold, verbose=False, classes=[0], imgsz=480)
        
        if results and len(results) > 0 and results[0].boxes is not None:
            person_count = len(results[0].boxes.xyxy)
            annotated = results[0].plot()
            annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))
            _detection_cache["count"] = person_count
            _detection_cache["frame_num"] = frame_idx
        else:
            person_count = 0
            annotated = frame
            _detection_cache["count"] = 0
    else:
        annotated = frame
        person_count = _detection_cache["count"]
    
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return frame_base64, person_count

async def analyze_video_realtime(
    video_path: str,
    process_fps: float = 25.0,
    confidence_threshold: float = 0.5,
) -> AsyncGenerator[str, None]:
    
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(_executor, _load_yolo_model)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"data: {json.dumps({'error': 'Could not open video'})}\n\n"
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    yield f"data: {json.dumps({'type': 'info', 'fps': fps, 'total_frames': total_frames, 'process_fps': process_fps})}\n\n"
    
    frame_idx = 0
    detect_every = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_idx / fps
        progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
        
        try:
            frame_base64, person_count = await loop.run_in_executor(
                _executor,
                _fast_detect_and_encode,
                frame,
                model,
                confidence_threshold,
                50,
                frame_idx,
                detect_every
            )
            
            if person_count > 20:
                risk_level = "HIGH"
                risk_score = min(100.0, person_count * 2.5)
            elif person_count > 10:
                risk_level = "MEDIUM"
                risk_score = person_count * 2.0
            else:
                risk_level = "LOW"
                risk_score = person_count * 1.0
            
            data = {
                "type": "frame",
                "frame": frame_base64,
                "time_seconds": float(time_seconds),
                "progress": float(progress),
                "person_count": int(person_count),
                "risk_level": str(risk_level),
                "risk_score": float(risk_score),
                "confidence": 0.85,
                "primary_cause": "Crowd density" if person_count > 10 else "Normal activity",
                "supporting_factors": [f"{person_count} people detected"],
                "explanation": f"Real-time: {person_count} people in frame",
                "density_score": float(person_count / 50.0),
                "motion_speed": 0.0,
                "direction_entropy": 0.0,
                "spatial_spread": 0.0,
            }
            
            yield f"data: {json.dumps(data)}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        frame_idx += 1
    
    cap.release()
    yield f"data: {json.dumps({'type': 'complete', 'message': 'Analysis complete'})}\n\n"
