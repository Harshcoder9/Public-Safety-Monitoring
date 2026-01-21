from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional
import numpy as np
from ..models import RiskLevel
from ..path_setup import ensure_workspace_on_path
from ..risk_reasoning import RiskDecision, RiskReasoningEngine, RiskSignals

@dataclass
class AnalysisResult:
    risk_level: RiskLevel
    risk_score: float
    max_loss: float
    mean_loss: float
    event_time_seconds: float
    confidence: float = 0.0
    primary_cause: str = ""
    supporting_factors: Optional[List[str]] = None
    explanation: str = ""
    samples: Optional[List[dict]] = None
    losses: Optional[List[float]] = None


_model_lock = Lock()
_model = None

def _get_default_model_path() -> str:
    root_dir = ensure_workspace_on_path()
    return os.path.join(root_dir, "Crowd_Anomaly_Detection", "AnomalyDetector.h5")

def analyze_video_autoencoder(
    *,
    video_path: str,
    sample_every_seconds: float = 0.2,
    threshold_low: float = 0.0008,
    threshold_medium: float = 0.0012,
    threshold_high: float = 0.0016,
    include_losses: bool = False,
    stop_on_high: bool = True,
    model_path: Optional[str] = None,
    # Risk reasoning configuration
    risk_window: int = 7,
    high_confirm_frames: int = 3,
    medium_confirm_frames: int = 2,
    cooldown_frames: int = 5,
    require_agreement: int = 2,
) -> AnalysisResult:
    global _model
    ensure_workspace_on_path()

    from Crowd_Anomaly_Detection.run_video_risk_alerts import (
        _classify_risk,
        _extract_sampled_grayscale_frames,
        _load_model,
        _mean_euclidean_loss,
        _preprocess_to_model_tensor,
    )

    if model_path is None:
        model_path = _get_default_model_path()

    frames_gray = _extract_sampled_grayscale_frames(video_path, sample_every_seconds)
    bunches, _usable_frames = _preprocess_to_model_tensor(frames_gray)

    with _model_lock:
        if _model is None:
            _model = _load_model(model_path)
        model = _model

    losses: List[float] = []
    first_alert_bunch_idx: Optional[int] = None
    samples: List[dict] = []

    reasoning = RiskReasoningEngine(
        window=int(risk_window),
        require_agreement=int(require_agreement),
        high_confirm_frames=int(high_confirm_frames),
        medium_confirm_frames=int(medium_confirm_frames),
        cooldown_frames=int(cooldown_frames),
        # AE has fewer direct motion cues; upweight persistence a bit.
        weights={
            "motion_speed": 0.35,
            "density_change_rate": 0.25,
            "persistence": 0.25,
            "spread": 0.10,
            "directional_chaos": 0.05,
        },
    )

    consec = 0
    last_loss: Optional[float] = None

    for bunch in bunches:
        n_bunch = np.expand_dims(bunch, axis=0)
        reconstructed = model.predict(n_bunch, verbose=0)
        loss = _mean_euclidean_loss(n_bunch, reconstructed)
        losses.append(loss)

        # AE-derived signals:
        # - motion_speed: normalize loss into [0..1] using thresholds
        # - density_change_rate: rate-of-change of loss (spike/shift proxy)
        # - persistence: consecutive above-threshold
        if float(loss) >= float(threshold_low):
            consec += 1
        else:
            consec = 0

        if last_loss is None:
            d_loss = 0.0
        else:
            d_loss = float(loss - last_loss)
        last_loss = float(loss)

        tl = float(max(1e-9, threshold_low))
        th = float(max(tl + 1e-9, threshold_high))
        motion_speed = float(np.clip((float(loss) - tl) / (th - tl), 0.0, 1.0))

        seconds_per_bunch = 10.0 * float(sample_every_seconds)
        density_change_rate = float(
            np.clip((max(0.0, d_loss) / max(1e-6, seconds_per_bunch)) / (3.0 * (th - tl)), 0.0, 1.0)
        )
        spread = float(np.clip(motion_speed, 0.0, 1.0))
        directional_chaos = 0.0

        decision: RiskDecision = reasoning.update(
            time_seconds=float((len(losses) - 1) * seconds_per_bunch),
            signals=RiskSignals(
                density_change_rate=density_change_rate,
                motion_speed=motion_speed,
                directional_chaos=directional_chaos,
                spread=spread,
                persistence_frames=consec,
            ),
        )

        bunch_idx = len(losses) - 1
        t_sec = float(bunch_idx * seconds_per_bunch)
        risk_str = _classify_risk(loss, threshold_low, threshold_medium, threshold_high)
        risk_level = RiskLevel(risk_str)
        cause = (
            "Motion pattern anomaly detected: spatiotemporal reconstruction error exceeded threshold."
            if risk_level != RiskLevel.NONE
            else "Normal scene motion."
        )

        # Replace coarse label with decision-grade risk level once we have reasoning.
        risk_level = (
            RiskLevel(decision.risk_level)
            if decision.risk_level in {"NONE", "LOW", "MEDIUM", "HIGH"}
            else risk_level
        )
        if risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            cause = decision.primary_cause
        samples.append(
            {
                "riskLevel": risk_level.value,
                "timeSeconds": t_sec,
                "loss": float(loss),
                "cause": cause,
                "riskScore": float(decision.risk_score),
                "confidence": float(decision.confidence),
                "primaryCause": decision.primary_cause,
                "supportingFactors": list(decision.supporting_factors),
                "explanation": decision.explanation,
                "signals": dict(decision.signals),
            }
        )

        if first_alert_bunch_idx is None and risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            first_alert_bunch_idx = bunch_idx

        if stop_on_high and risk_level == RiskLevel.HIGH:
            break

    max_loss = float(np.max(losses)) if losses else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0

    seconds_per_bunch = 10.0 * float(sample_every_seconds)
    event_time_seconds = 0.0
    if first_alert_bunch_idx is not None:
        event_time_seconds = float(first_alert_bunch_idx * seconds_per_bunch)

    # Overall decision-grade result: take max decision score/level from computed samples.
    overall_score = float(max((float(s.get("riskScore", 0.0)) for s in samples), default=0.0))
    overall_level = RiskLevel(_classify_risk(max_loss, threshold_low, threshold_medium, threshold_high))
    if overall_score <= 0:
        overall_level = RiskLevel.NONE
    elif overall_score <= 30:
        overall_level = RiskLevel.LOW
    elif overall_score <= 70:
        overall_level = RiskLevel.MEDIUM
    else:
        overall_level = RiskLevel.HIGH

    rep = next((s for s in samples if s.get("riskLevel") in {"MEDIUM", "HIGH"}), None)

    return AnalysisResult(
        risk_level=overall_level,
        risk_score=overall_score,
        max_loss=max_loss,
        mean_loss=mean_loss,
        event_time_seconds=event_time_seconds,
        confidence=float(rep.get("confidence", 0.0)) if rep else 0.0,
        primary_cause=str(rep.get("primaryCause", "")) if rep else "",
        supporting_factors=list(rep.get("supportingFactors") or []) if rep else [],
        explanation=str(rep.get("explanation", "")) if rep else "",
        samples=samples,
        losses=losses if include_losses else None,
    )
