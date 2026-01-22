from __future__ import annotations

import asyncio
import base64
import cv2
import json
import numpy as np
from pathlib import Path
from typing import AsyncGenerator, Dict
from concurrent.futures import ThreadPoolExecutor
from .analyzers.yolo_crowd_detector import _load_yolo_model, _calculate_density_score, _calculate_motion_speed, _calculate_direction_entropy, _calculate_spatial_spread
from .risk_reasoning import RiskReasoningEngine, RiskSignals

_executor = ThreadPoolExecutor(max_workers=4)

def _quick_detect(frame, model, confidence_threshold):
    small_frame = cv2.resize(frame, (640, 360))
    results = model.track(small_frame, persist=True, conf=confidence_threshold, verbose=False, classes=[0], imgsz=640)
    
    if not results or len(results) == 0 or results[0].boxes is None:
        return None, 0
    
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy()
    annotated = results[0].plot()
    annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))
    
    return annotated, len(detections)

def _process_frame(frame, model, prev_centroids, density_history, metrics_history, reasoning, frame_idx, fps, total_frames, confidence_threshold):
    time_seconds = frame_idx / fps
    progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
    
    results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False, classes=[0])
    
    if not results or len(results) == 0 or results[0].boxes is None:
        return None
    
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(len(detections))
    
    annotated_frame = results[0].plot()
    
    frame_area = frame.shape[0] * frame.shape[1]
    density_score = _calculate_density_score(detections, frame_area)
    
    curr_centroids = {}
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        track_id = int(track_ids[i]) if i < len(track_ids) else i
        curr_centroids[track_id] = (cx, cy)
    
    motion_speed = _calculate_motion_speed(prev_centroids, curr_centroids, fps)
    direction_entropy = _calculate_direction_entropy(prev_centroids, curr_centroids)
    spatial_spread = _calculate_spatial_spread(detections, frame.shape[:2])
    
    density_history.append(density_score)
    if len(density_history) > 30:
        density_history.pop(0)
    
    density_change_rate = 0.0
    if len(density_history) >= 2:
        density_change_rate = abs(density_history[-1] - density_history[-2])
    
    normalized_speed = motion_speed / 100.0 if motion_speed > 0 else 0.0
    
    persistence_frames = len([m for m in metrics_history[-10:] if m.get("person_count", 0) > 10])
    
    signals = RiskSignals(
        density_change_rate=float(density_change_rate),
        motion_speed=float(normalized_speed),
        directional_chaos=float(direction_entropy),
        spread=float(spatial_spread),
        persistence_frames=int(persistence_frames),
    )
    
    decision = reasoning.decide(time_seconds=time_seconds, signals=signals)
    
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    data = {
        "type": "frame",
        "frame": frame_base64,
        "time_seconds": float(time_seconds),
        "progress": float(progress),
        "person_count": int(len(detections)),
        "risk_level": str(decision.risk_level),
        "risk_score": float(decision.risk_score),
        "confidence": float(decision.confidence),
        "primary_cause": str(decision.primary_cause),
        "supporting_factors": [str(f) for f in decision.supporting_factors],
        "explanation": str(decision.explanation),
        "density_score": float(density_score),
        "motion_speed": float(motion_speed),
        "direction_entropy": float(direction_entropy),
        "spatial_spread": float(spatial_spread),
    }
    
    metrics_history.append({
        "person_count": len(detections),
        "time_seconds": time_seconds
    })
    if len(metrics_history) > 30:
        metrics_history.pop(0)
    
    return data, curr_centroids

async def analyze_video_realtime(
    video_path: str,
    process_fps: float = 10.0,
    confidence_threshold: float = 0.4,
) -> AsyncGenerator[str, None]:
    
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(_executor, _load_yolo_model)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"data: {json.dumps({'error': 'Could not open video'})}\n\n"
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps / process_fps))
    
    yield f"data: {json.dumps({'type': 'info', 'fps': fps, 'total_frames': total_frames, 'process_fps': process_fps})}\n\n"
    
    reasoning = RiskReasoningEngine(
        window=10,
        require_agreement=3,
        high_confirm_frames=3,
        medium_confirm_frames=2,
        cooldown_frames=5,
    )
    
    frame_idx = 0
    prev_centroids = {}
    density_history = []
    metrics_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        
        try:
            result = await loop.run_in_executor(
                _executor,
                _process_frame,
                frame,
                model,
                prev_centroids,
                density_history,
                metrics_history,
                reasoning,
                frame_idx,
                fps,
                total_frames,
                confidence_threshold
            )
            
            if result:
                data, prev_centroids = result
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        frame_idx += 1
    
    cap.release()
    
    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
