from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..models import RiskLevel
from ..risk_reasoning import RiskDecision, RiskReasoningEngine, RiskSignals

@dataclass
class RiskSample:
    time_seconds: float
    risk_level: RiskLevel
    mean_flow_mag: float
    z_score: float
    active_ratio: float
    cause: str

    # Decision-grade fields
    risk_score_0_100: float = 0.0
    confidence: float = 0.0
    primary_cause: str = ""
    supporting_factors: Optional[List[str]] = None
    explanation: str = ""
    signals: Optional[Dict[str, float]] = None

@dataclass
class OpticalFlowAnalysisResult:
    risk_level: RiskLevel
    risk_score: float
    event_time_seconds: float
    samples: List[RiskSample]
    counts: dict

def _rolling_median_mad(values: list[float], window: int) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    w = values[-window:] if len(values) > window else values
    med = float(np.median(w))
    mad = float(np.median(np.abs(np.array(w) - med)))
    return med, mad

def _risk_from_z(z: float, z_low: float, z_med: float, z_high: float) -> RiskLevel:
    if z > z_high:
        return RiskLevel.HIGH
    if z > z_med:
        return RiskLevel.MEDIUM
    if z > z_low:
        return RiskLevel.LOW
    return RiskLevel.NONE

def _cause_for(risk: RiskLevel, *, z: float, active_ratio: float) -> str:
    if risk == RiskLevel.NONE:
        return "Normal scene motion."

    widespread = active_ratio >= 0.18

    if risk == RiskLevel.HIGH:
        if widespread:
            return "Sudden crowd acceleration combined with density spike (widespread scene-level motion)."
        return "Sudden crowd acceleration detected (optical-flow spike)."

    if risk == RiskLevel.MEDIUM:
        if widespread:
            return "Elevated crowd motion with widespread movement."
        return "Elevated crowd motion detected."

    if widespread:
        return "Noticeable motion increase across the scene."
    return "Noticeable motion spike detected."

def analyze_video_optical_flow(
    *,
    video_path: str,
    process_fps: float = 5.0,  
    resize_width: int = 320,
    mad_window: int = 30,
    z_low: float = 3.0,
    z_med: float = 5.0,
    z_high: float = 7.0,
    min_consecutive: int = 1,
    stop_on_high: bool = True,
    active_mag_threshold: float = 1.0,
    # Risk reasoning configuration
    risk_window: int = 7,
    high_confirm_frames: int = 3,
    medium_confirm_frames: int = 2,
    cooldown_frames: int = 5,
    require_agreement: int = 2,
) -> OpticalFlowAnalysisResult:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(process_fps, 0.1))))
    prev_gray = None
    mags: list[float] = []
    active_ratios: list[float] = []
    samples: List[RiskSample] = []
    counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
    overall_risk = RiskLevel.NONE
    first_high_time: Optional[float] = None
    consec = 0
    frame_idx = 0

    reasoning = RiskReasoningEngine(
        window=int(risk_window),
        require_agreement=int(require_agreement),
        high_confirm_frames=int(high_confirm_frames),
        medium_confirm_frames=int(medium_confirm_frames),
        cooldown_frames=int(cooldown_frames),
    )
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            t_sec = float(frame_idx / fps)

            h, w = frame_bgr.shape[:2]
            if w > 0 and resize_width > 0 and w != resize_width:
                new_w = int(resize_width)
                new_h = max(1, int(h * (new_w / w)))
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                frame_idx += 1
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            active_ratio = float(np.mean(mag > float(active_mag_threshold)))
            mags.append(mean_mag)
            active_ratios.append(active_ratio)
            med, mad = _rolling_median_mad(mags[:-1], window=mad_window)
            denom = (mad * 1.4826) + 1e-6
            z = (mean_mag - med) / denom if len(mags) > 2 else 0.0

            # Directional chaos: circular std of angles on active motion pixels (normalized to [0..1]).
            active_mask = mag > float(active_mag_threshold)
            if np.any(active_mask):
                ang_active = ang[active_mask]
                sin_m = float(np.mean(np.sin(ang_active)))
                cos_m = float(np.mean(np.cos(ang_active)))
                R = float(np.hypot(sin_m, cos_m))
                circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-6)))))
                directional_chaos = float(np.clip(circ_std / np.pi, 0.0, 1.0))
            else:
                directional_chaos = 0.0

            # Density change rate proxy: rate-of-change of moving-area ratio.
            if len(active_ratios) >= 2:
                dt = max(1e-6, float(step / fps))
                d_ar = float(active_ratios[-1] - active_ratios[-2])
                # Normalize: 0.0 at no increase; saturate at ~0.15 increase per second.
                density_change_rate = float(np.clip((d_ar / dt) / 0.15, 0.0, 1.0))
            else:
                density_change_rate = 0.0

            # Motion speed proxy: map z-score (robust) into [0..1] using provided z thresholds.
            # <= z_low -> 0, >= z_high -> 1, linearly in between.
            z_low_eff = float(max(1e-6, z_low))
            z_high_eff = float(max(z_low_eff + 1e-6, z_high))
            motion_speed = float(np.clip((float(z) - z_low_eff) / (z_high_eff - z_low_eff), 0.0, 1.0))

            spread = float(np.clip(active_ratio / 0.35, 0.0, 1.0))

            # Persistence is consecutive anomaly-like frames (based on motion_speed). 0.35 ~= mild anomaly.
            if motion_speed >= 0.35:
                consec += 1
            else:
                consec = 0

            decision: RiskDecision = reasoning.update(
                time_seconds=t_sec,
                signals=RiskSignals(
                    density_change_rate=density_change_rate,
                    motion_speed=motion_speed,
                    directional_chaos=directional_chaos,
                    spread=spread,
                    persistence_frames=consec,
                ),
            )

            risk_level = (
                RiskLevel(decision.risk_level)
                if decision.risk_level in {"NONE", "LOW", "MEDIUM", "HIGH"}
                else RiskLevel.NONE
            )

            # Keep existing min_consecutive behavior as an extra false-alarm guard.
            if risk_level != RiskLevel.NONE and consec < max(1, int(min_consecutive)):
                risk_level = RiskLevel.NONE

            cause = _cause_for(risk_level, z=float(z), active_ratio=active_ratio)
            if risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
                cause = decision.primary_cause

            samples.append(
                RiskSample(
                    time_seconds=t_sec,
                    risk_level=risk_level,
                    mean_flow_mag=mean_mag,
                    z_score=float(z),
                    active_ratio=active_ratio,
                    cause=cause,
                    risk_score_0_100=float(decision.risk_score),
                    confidence=float(decision.confidence),
                    primary_cause=decision.primary_cause,
                    supporting_factors=list(decision.supporting_factors),
                    explanation=decision.explanation,
                    signals=dict(decision.signals),
                )
            )

            counts[risk_level.value] = counts.get(risk_level.value, 0) + 1
            overall_risk = max(
                overall_risk,
                risk_level,
                key=lambda r: ["NONE", "LOW", "MEDIUM", "HIGH"].index(r.value),
            )

            if risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH} and first_high_time is None:
                first_high_time = t_sec

            if risk_level == RiskLevel.HIGH and stop_on_high:
                # Stop only after decision engine confirms HIGH.
                break

            prev_gray = gray
            frame_idx += 1

    finally:
        cap.release()

    event_time_seconds = float(first_high_time or 0.0)
    # Decision-grade risk score: max smoothed score across samples.
    risk_score = float(max((s.risk_score_0_100 for s in samples), default=0.0))

    return OpticalFlowAnalysisResult(
        risk_level=overall_risk,
        risk_score=risk_score,
        event_time_seconds=event_time_seconds,
        samples=samples,
        counts=counts,
    )
