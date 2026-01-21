from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class RiskLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class Alert:
    id: str
    created_at: datetime
    user_email: str
    location: str
    risk_level: RiskLevel
    risk_score: float
    file_name: str
    event_time_seconds: float
    confidence: float = 0.0
    primary_cause: str = ""
    supporting_factors: Optional[list[str]] = None
    explanation: str = ""
    acknowledged_at: Optional[datetime] = None

@dataclass
class UserLocation:
    user_email: str
    latitude: float
    longitude: float
    timestamp: datetime
    active: bool = True
