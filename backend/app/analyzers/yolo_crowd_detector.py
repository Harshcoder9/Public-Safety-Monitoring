from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from threading import Lock
from ..models import RiskLevel
from ..risk_reasoning import RiskDecision, RiskReasoningEngine, RiskSignals

@dataclass
class CrowdMetrics:
    person_count: int
    density_score: float
    avg_speed: float
    motion_variance: float
    direction_entropy: float
    spatial_spread: float
    timestamp: float

@dataclass
class YOLOAnalysisResult:
    risk_level: RiskLevel
    risk_score: float
    event_time_seconds: float
    confidence: float
    primary_cause: str
    supporting_factors: List[str]
    explanation: str
    samples: List[Dict]
    metrics_history: List[CrowdMetrics]

_model_lock = Lock()
_yolo_model = None

def _load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO('yolov8n.pt')
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    return _yolo_model

def _calculate_density_score(detections: np.ndarray, frame_area: float) -> float:
    if len(detections) == 0:
        return 0.0
    
    total_bbox_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        total_bbox_area += (x2 - x1) * (y2 - y1)
    
    density = total_bbox_area / frame_area
    return min(density, 1.0)

def _calculate_motion_speed(prev_centroids: Dict, curr_centroids: Dict, fps: float) -> float:
    if not prev_centroids or not curr_centroids:
        return 0.0
    
    speeds = []
    for track_id, curr_pos in curr_centroids.items():
        if track_id in prev_centroids:
            prev_pos = prev_centroids[track_id]
            dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            speed = dist * fps
            speeds.append(speed)
    
    return float(np.mean(speeds)) if speeds else 0.0

def _calculate_direction_entropy(prev_centroids: Dict, curr_centroids: Dict) -> float:
    if not prev_centroids or not curr_centroids:
        return 0.0
    
    directions = []
    for track_id, curr_pos in curr_centroids.items():
        if track_id in prev_centroids:
            prev_pos = prev_centroids[track_id]
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            angle = np.arctan2(dy, dx)
            directions.append(angle)
    
    if not directions:
        return 0.0
    
    hist, _ = np.histogram(directions, bins=16, range=(-np.pi, np.pi))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(16)
    
    return float(entropy / max_entropy)

def _calculate_spatial_spread(detections: np.ndarray, frame_shape: Tuple) -> float:
    if len(detections) == 0:
        return 0.0
    
    centroids = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append([cx, cy])
    
    centroids = np.array(centroids)
    std_x = np.std(centroids[:, 0])
    std_y = np.std(centroids[:, 1])
    
    frame_diag = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
    spread_score = (std_x + std_y) / frame_diag
    
    return float(min(spread_score, 1.0))

def analyze_video_yolo(
    *,
    video_path: str,
    process_fps: float = 10.0,
    confidence_threshold: float = 0.4,
    risk_window: int = 10,
    high_confirm_frames: int = 3,
    medium_confirm_frames: int = 2,
    cooldown_frames: int = 5,
    require_agreement: int = 3,
    stop_on_high: bool = True,
    density_weight: float = 0.25,
    speed_weight: float = 0.30,
    chaos_weight: float = 0.25,
    spread_weight: float = 0.20,
) -> YOLOAnalysisResult:
    
    with _model_lock:
        model = _load_yolo_model()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps / process_fps))
    
    frame_idx = 0
    time_seconds = 0.0
    
    metrics_history: List[CrowdMetrics] = []
    prev_centroids: Dict = {}
    
    reasoning = RiskReasoningEngine(
        window=int(risk_window),
        require_agreement=int(require_agreement),
        high_confirm_frames=int(high_confirm_frames),
        medium_confirm_frames=int(medium_confirm_frames),
        cooldown_frames=int(cooldown_frames),
        weights={
            "density_change_rate": density_weight,
            "motion_speed": speed_weight,
            "directional_chaos": chaos_weight,
            "spread": spread_weight,
            "persistence": 0.0,
        },
    )
    
    samples: List[Dict] = []
    first_alert_time = 0.0
    max_risk_level = RiskLevel.NONE
    max_risk_score = 0.0
    
    density_history = []
    speed_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        
        time_seconds = frame_idx / fps
        
        results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False, classes=[0])
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detections = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(len(detections))
            
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
            
            motion_variance = np.std([motion_speed]) if motion_speed > 0 else 0.0
            
            metrics = CrowdMetrics(
                person_count=len(detections),
                density_score=density_score,
                avg_speed=motion_speed,
                motion_variance=motion_variance,
                direction_entropy=direction_entropy,
                spatial_spread=spatial_spread,
                timestamp=time_seconds
            )
            metrics_history.append(metrics)
            
            density_history.append(density_score)
            speed_history.append(motion_speed)
            
            density_change_rate = 0.0
            if len(density_history) >= 2:
                density_change_rate = abs(density_history[-1] - density_history[-2])
            
            normalized_speed = motion_speed / 100.0 if motion_speed > 0 else 0.0
            
            persistence_frames = 0
            if len(metrics_history) >= 2:
                for i in range(len(metrics_history) - 1, -1, -1):
                    if metrics_history[i].person_count > 10:
                        persistence_frames += 1
                    else:
                        break
            
            signals = RiskSignals(
                density_change_rate=float(density_change_rate),
                motion_speed=float(normalized_speed),
                directional_chaos=float(direction_entropy),
                spread=float(spatial_spread),
                persistence_frames=int(persistence_frames),
            )
            
            decision = reasoning.decide(time_seconds=time_seconds, signals=signals)
            
            sample_dict = {
                "time_seconds": decision.time_seconds,
                "risk_level": decision.risk_level,
                "risk_score": decision.risk_score,
                "confidence": decision.confidence,
                "primary_cause": decision.primary_cause,
                "supporting_factors": decision.supporting_factors,
                "explanation": decision.explanation,
                "person_count": len(detections),
                "density_score": density_score,
                "motion_speed": motion_speed,
                "direction_entropy": direction_entropy,
                "spatial_spread": spatial_spread,
            }
            samples.append(sample_dict)
            
            risk_enum = RiskLevel.from_str(decision.risk_level)
            
            if risk_enum.value > max_risk_level.value:
                max_risk_level = risk_enum
                max_risk_score = decision.risk_score
                first_alert_time = time_seconds
            
            if stop_on_high and risk_enum == RiskLevel.HIGH:
                break
            
            prev_centroids = curr_centroids
        
        frame_idx += 1
        
        if frame_idx >= total_frames:
            break
    
    cap.release()
    
    final_decision = samples[-1] if samples else None
    
    if not final_decision:
        final_decision = {
            "risk_level": "NONE",
            "risk_score": 0.0,
            "confidence": 0.0,
            "primary_cause": "No crowd detected",
            "supporting_factors": [],
            "explanation": "Video analysis completed with no significant crowd activity detected."
        }
    
    return YOLOAnalysisResult(
        risk_level=max_risk_level,
        risk_score=max_risk_score,
        event_time_seconds=first_alert_time,
        confidence=final_decision.get("confidence", 0.0),
        primary_cause=final_decision.get("primary_cause", ""),
        supporting_factors=final_decision.get("supporting_factors", []),
        explanation=final_decision.get("explanation", ""),
        samples=samples,
        metrics_history=metrics_history,
    )

def analyze_rtsp_stream(
    *,
    rtsp_url: str,
    callback,
    process_fps: float = 10.0,
    confidence_threshold: float = 0.4,
    risk_window: int = 10,
    high_confirm_frames: int = 3,
    medium_confirm_frames: int = 2,
    cooldown_frames: int = 5,
    require_agreement: int = 3,
):
    
    with _model_lock:
        model = _load_yolo_model()
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps / process_fps))
    
    frame_idx = 0
    prev_centroids = {}
    density_history = []
    speed_history = []
    metrics_history = []
    
    reasoning = RiskReasoningEngine(
        window=int(risk_window),
        require_agreement=int(require_agreement),
        high_confirm_frames=int(high_confirm_frames),
        medium_confirm_frames=int(medium_confirm_frames),
        cooldown_frames=int(cooldown_frames),
        weights={
            "density_change_rate": 0.25,
            "motion_speed": 0.30,
            "directional_chaos": 0.25,
            "spread": 0.20,
            "persistence": 0.0,
        },
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        
        time_seconds = frame_idx / fps
        
        results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False, classes=[0])
        
        if results and len(results) > 0 and results[0].boxes is not None:
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
            speed_history.append(motion_speed)
            
            density_change_rate = 0.0
            if len(density_history) >= 2:
                density_change_rate = abs(density_history[-1] - density_history[-2])
            
            normalized_speed = motion_speed / 100.0 if motion_speed > 0 else 0.0
            
            persistence_frames = len([m for m in metrics_history[-10:] if m.person_count > 10])
            
            signals = RiskSignals(
                density_change_rate=float(density_change_rate),
                motion_speed=float(normalized_speed),
                directional_chaos=float(direction_entropy),
                spread=float(spatial_spread),
                persistence_frames=int(persistence_frames),
            )
            
            decision = reasoning.decide(time_seconds=time_seconds, signals=signals)
            
            data = {
                "frame": annotated_frame,
                "time_seconds": time_seconds,
                "person_count": len(detections),
                "risk_level": decision.risk_level,
                "risk_score": decision.risk_score,
                "confidence": decision.confidence,
                "primary_cause": decision.primary_cause,
                "supporting_factors": decision.supporting_factors,
                "explanation": decision.explanation,
                "density_score": density_score,
                "motion_speed": motion_speed,
            }
            
            callback(data)
            
            prev_centroids = curr_centroids
        
        frame_idx += 1
    
    cap.release()
