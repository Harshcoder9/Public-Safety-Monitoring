from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4
from .models import Alert, RiskLevel, UserLocation

class LocationStore:
    def __init__(self, *, file_path: Optional[str] = None) -> None:
        self._lock = Lock()
        self._locations: Dict[str, UserLocation] = {}

        if file_path is None:
            here = os.path.dirname(os.path.abspath(__file__)) 
            backend_dir = os.path.dirname(here)  
            file_path = os.path.join(backend_dir, "data", "locations.json")
        
        self._file_path = file_path
        self._load_from_disk()

    def _serialize_location(self, loc: UserLocation) -> dict:
        return {
            "user_email": loc.user_email,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "timestamp": loc.timestamp.isoformat(),
            "active": loc.active,
        }

    def _deserialize_location(self, d: dict) -> UserLocation:
        return UserLocation(
            user_email=str(d["user_email"]),
            latitude=float(d["latitude"]),
            longitude=float(d["longitude"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            active=bool(d["active"]),
        )

    def _load_from_disk(self) -> None:
        try:
            if not os.path.exists(self._file_path):
                return
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                return
            for item in raw:
                try:
                    loc = self._deserialize_location(item)
                    self._locations[loc.user_email] = loc
                except Exception:
                    continue
        except Exception:
            return

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        payload = [self._serialize_location(loc) for loc in self._locations.values()]
        tmp = f"{self._file_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self._file_path)

    def update_location(self, user_email: str, latitude: float, longitude: float) -> UserLocation:
        with self._lock:
            loc = UserLocation(
                user_email=user_email,
                latitude=latitude,
                longitude=longitude,
                timestamp=datetime.now(timezone.utc),
                active=True
            )
            self._locations[user_email] = loc
            self._save_to_disk()
            return loc

    def get_active_locations(self) -> List[dict]:
        with self._lock:
            # Filter distinct users, just returning list of their latest location
            # Since we store by user_email in dict, it's already latest per user.
            active = [
                {
                    "user_email": loc.user_email,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "timestamp": loc.timestamp.isoformat(),
                    "active": loc.active
                }
                for loc in self._locations.values() if loc.active
            ]
            return active

    def remove_location(self, user_email: str) -> None:
         with self._lock:
            if user_email in self._locations:
                del self._locations[user_email]
                self._save_to_disk()

class AlertStore:
    def __init__(self, *, file_path: Optional[str] = None) -> None:
        self._lock = Lock()
        self._alerts: Dict[str, Alert] = {}

        if file_path is None:
            here = os.path.dirname(os.path.abspath(__file__)) 
            backend_dir = os.path.dirname(here)  
            file_path = os.path.join(backend_dir, "data", "alerts.json")

        self._file_path = file_path
        self._load_from_disk()

    def _serialize_alert(self, a: Alert) -> dict:
        d = asdict(a)
        d["created_at"] = a.created_at.isoformat()
        d["acknowledged_at"] = a.acknowledged_at.isoformat() if a.acknowledged_at else None
        return d

    def _deserialize_alert(self, d: dict) -> Alert:
        created_at = datetime.fromisoformat(d["created_at"])
        ack_raw = d.get("acknowledged_at")
        acknowledged_at = datetime.fromisoformat(ack_raw) if ack_raw else None
        return Alert(
            id=str(d["id"]),
            created_at=created_at,
            user_email=str(d["user_email"]),
            location=str(d["location"]),
            risk_level=RiskLevel(str(d["risk_level"])),
            risk_score=float(d["risk_score"]),
            file_name=str(d.get("file_name", "")),
            event_time_seconds=float(d.get("event_time_seconds", 0.0)),
            acknowledged_at=acknowledged_at,
        )

    def _load_from_disk(self) -> None:
        try:
            if not os.path.exists(self._file_path):
                return
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                return
            for item in raw:
                try:
                    alert = self._deserialize_alert(item)
                    self._alerts[alert.id] = alert
                except Exception:
                    continue
        except Exception:
            return

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        payload = [self._serialize_alert(a) for a in self._alerts.values()]
        tmp = f"{self._file_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self._file_path)

    def create_alert(
        self,
        *,
        user_email: str,
        location: str,
        risk_level: RiskLevel,
        risk_score: float,
        file_name: str,
        event_time_seconds: float,
    ) -> Alert:
        alert = Alert(
            id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            user_email=user_email,
            location=location,
            risk_level=risk_level,
            risk_score=float(risk_score),
            file_name=file_name,
            event_time_seconds=float(event_time_seconds),
        )
        with self._lock:
            self._alerts[alert.id] = alert
            self._save_to_disk()
        return alert

    def list_alerts(self, *, include_acknowledged: bool = True) -> List[dict]:
        with self._lock:
            alerts = list(self._alerts.values())

        def _key(a: Alert):
            return a.created_at

        alerts.sort(key=_key, reverse=True)
        if not include_acknowledged:
            alerts = [a for a in alerts if a.acknowledged_at is None]

        return [self._serialize_alert(a) for a in alerts]

    def acknowledge(self, alert_id: str) -> Optional[dict]:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            if alert.acknowledged_at is None:
                alert.acknowledged_at = datetime.now(timezone.utc)
            self._save_to_disk()
            return self._serialize_alert(alert)
