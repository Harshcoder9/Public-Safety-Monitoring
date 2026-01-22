from __future__ import annotations

import asyncio
import base64
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np

@dataclass
class CameraStream:
    camera_id: str
    rtsp_url: str
    location: str
    active: bool = False
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    frame_queue: Queue = field(default_factory=lambda: Queue(maxsize=2))
    latest_data: Dict = field(default_factory=dict)
    subscribers: List[Callable] = field(default_factory=list)

class StreamManager:
    def __init__(self):
        self.streams: Dict[str, CameraStream] = {}
        self._lock = threading.Lock()
    
    def add_stream(self, camera_id: str, rtsp_url: str, location: str, analyzer_callback: Callable) -> bool:
        with self._lock:
            if camera_id in self.streams:
                return False
            
            stream = CameraStream(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                location=location
            )
            stream.subscribers.append(analyzer_callback)
            self.streams[camera_id] = stream
            
            thread = threading.Thread(
                target=self._stream_worker,
                args=(camera_id,),
                daemon=True
            )
            stream.thread = thread
            stream.active = True
            thread.start()
            
            return True
    
    def remove_stream(self, camera_id: str) -> bool:
        with self._lock:
            if camera_id not in self.streams:
                return False
            
            stream = self.streams[camera_id]
            stream.stop_event.set()
            stream.active = False
            
            if stream.thread and stream.thread.is_alive():
                stream.thread.join(timeout=2.0)
            
            del self.streams[camera_id]
            return True
    
    def get_latest_frame(self, camera_id: str) -> Optional[Dict]:
        with self._lock:
            if camera_id not in self.streams:
                return None
            return self.streams[camera_id].latest_data.copy() if self.streams[camera_id].latest_data else None
    
    def get_all_latest_frames(self) -> Dict[str, Dict]:
        with self._lock:
            result = {}
            for cam_id, stream in self.streams.items():
                if stream.latest_data:
                    result[cam_id] = stream.latest_data.copy()
            return result
    
    def subscribe(self, camera_id: str, callback: Callable) -> bool:
        with self._lock:
            if camera_id not in self.streams:
                return False
            self.streams[camera_id].subscribers.append(callback)
            return True
    
    def _stream_worker(self, camera_id: str):
        stream = self.streams.get(camera_id)
        if not stream:
            return
        
        from .analyzers.yolo_crowd_detector import _load_yolo_model, _calculate_density_score, _calculate_motion_speed, _calculate_direction_entropy, _calculate_spatial_spread
        from .risk_reasoning import RiskReasoningEngine, RiskSignals
        
        try:
            model = _load_yolo_model()
        except Exception as e:
            print(f"Failed to load YOLO model for camera {camera_id}: {e}")
            return
        
        cap = cv2.VideoCapture(stream.rtsp_url)
        if not cap.isOpened():
            print(f"Failed to open stream for camera {camera_id}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        process_fps = 10.0
        frame_interval = max(1, int(fps / process_fps))
        
        frame_idx = 0
        prev_centroids = {}
        density_history = []
        metrics_history = []
        
        reasoning = RiskReasoningEngine(
            window=10,
            require_agreement=3,
            high_confirm_frames=3,
            medium_confirm_frames=2,
            cooldown_frames=5,
            weights={
                "density_change_rate": 0.25,
                "motion_speed": 0.30,
                "directional_chaos": 0.25,
                "spread": 0.20,
                "persistence": 0.0,
            },
        )
        
        while not stream.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            time_seconds = frame_idx / fps
            
            try:
                results = model.track(frame, persist=True, conf=0.4, verbose=False, classes=[0])
                
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
                    
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    data = {
                        "camera_id": camera_id,
                        "location": stream.location,
                        "frame": frame_base64,
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
                        "timestamp": time.time(),
                    }
                    
                    stream.latest_data = data
                    
                    metrics_history.append({
                        "person_count": len(detections),
                        "time_seconds": time_seconds
                    })
                    if len(metrics_history) > 30:
                        metrics_history.pop(0)
                    
                    for callback in stream.subscribers:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"Callback error for camera {camera_id}: {e}")
                    
                    prev_centroids = curr_centroids
                
            except Exception as e:
                print(f"Frame processing error for camera {camera_id}: {e}")
            
            frame_idx += 1
        
        cap.release()
        print(f"Stream worker stopped for camera {camera_id}")

stream_manager = StreamManager()
