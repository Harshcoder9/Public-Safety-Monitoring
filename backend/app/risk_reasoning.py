from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RiskSignals:
    density_change_rate: float  # proxy: rate-of-change of moving-area ratio (or loss-change for AE)
    motion_speed: float  # proxy: normalized motion intensity (or loss z-score for AE)
    directional_chaos: float  # [0..1] proxy: circular spread of movement directions
    spread: float  # [0..1] proxy: fraction of frame affected
    persistence_frames: int  # consecutive anomaly-like frames


@dataclass(frozen=True)
class RiskDecision:
    time_seconds: float
    risk_score: float  # 0..100
    risk_level: str  # LOW/MEDIUM/HIGH/NONE
    confidence: float  # 0..1
    primary_cause: str
    supporting_factors: List[str]
    explanation: str
    signals: Dict[str, float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _sec_to_hhmmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _risk_level_from_score(score: float) -> str:
    if score <= 0:
        return "NONE"
    if score <= 30:
        return "LOW"
    if score <= 70:
        return "MEDIUM"
    return "HIGH"


class RiskReasoningEngine:

    def __init__(
        self,
        *,
        window: int = 10,
        require_agreement: int = 3,
        high_confirm_frames: int = 3,
        medium_confirm_frames: int = 2,
        cooldown_frames: int = 5,
        deescalate_frames: int = 3,
        low_confirm_frames: int = 1,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.window = max(3, int(window))
        self.require_agreement = max(1, int(require_agreement))
        self.high_confirm_frames = max(1, int(high_confirm_frames))
        self.medium_confirm_frames = max(1, int(medium_confirm_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.deescalate_frames = max(1, int(deescalate_frames))
        self.low_confirm_frames = max(1, int(low_confirm_frames))

        self.weights = weights or {
            "motion_speed": 0.30,
            "directional_chaos": 0.25,
            "density_change_rate": 0.25,
            "spread": 0.20,
            "persistence": 0.0,
        }

        self._scores: List[float] = []
        self._raw_levels: List[str] = []
        self._stable_level: str = "NONE"
        self._level_hold: int = 0
        self._consec_high: int = 0
        self._consec_med: int = 0
        self._consec_low: int = 0
        self._below_med: int = 0
        self._below_high: int = 0

    def _smooth_score(self) -> float:
        if not self._scores:
            return 0.0
        w = self._scores[-self.window :]
        return float(sum(w) / max(1, len(w)))

    def _window_variance(self) -> float:
        w = self._scores[-self.window :]
        if len(w) < 2:
            return 0.0
        mean = sum(w) / len(w)
        return float(sum((x - mean) ** 2 for x in w) / (len(w) - 1))

    def _normalize_signals(self, signals: RiskSignals) -> Dict[str, float]:
        # Normalize into [0..1]. These mappings are not hard-coded scores;
        # they convert measurable magnitudes into comparable contributions.
        density = _clamp(signals.density_change_rate, 0.0, 1.0)
        speed = _clamp(signals.motion_speed, 0.0, 1.0)
        chaos = _clamp(signals.directional_chaos, 0.0, 1.0)
        spread = _clamp(signals.spread, 0.0, 1.0)

        # Persistence saturates (short spikes are cheap, sustained anomalies matter)
        pers = _clamp(signals.persistence_frames / 10.0, 0.0, 1.0)

        return {
            "density_change_rate": density,
            "motion_speed": speed,
            "directional_chaos": chaos,
            "spread": spread,
            "persistence": pers,
        }

    def _agreement_count(self, norm: Dict[str, float]) -> int:
        # Count strong signals (>= 0.55) as agreeing evidence.
        return sum(1 for v in norm.values() if v >= 0.55)

    def _compute_score(self, norm: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        total_w = 0.0
        weighted = 0.0
        contributions: Dict[str, float] = {}

        for key, value in norm.items():
            w = float(self.weights.get(key, 0.0))
            if w <= 0:
                continue
            total_w += w
            c = w * float(value)
            weighted += c
            contributions[key] = c

        if total_w <= 0:
            return 0.0, {k: 0.0 for k in norm.keys()}

        score = 100.0 * (weighted / total_w)
        return float(_clamp(score, 0.0, 100.0)), contributions

    def _stabilize_level(self, raw_level: str, smooth_score: float) -> str:
        # Temporal stability:
        # - HIGH requires N consecutive frames
        # - MEDIUM requires M consecutive frames
        # - cooldown to prevent rapid de-escalation
        # - de-escalate only after sustained below-threshold

        if self._level_hold > 0:
            self._level_hold -= 1
            return self._stable_level

        if raw_level == "HIGH":
            self._consec_high += 1
        else:
            self._consec_high = 0

        if raw_level in {"HIGH", "MEDIUM"}:
            self._consec_med += 1
        else:
            self._consec_med = 0

        if raw_level in {"HIGH", "MEDIUM", "LOW"}:
            self._consec_low += 1
        else:
            self._consec_low = 0

        if smooth_score <= 70:
            self._below_high += 1
        else:
            self._below_high = 0

        if smooth_score <= 30:
            self._below_med += 1
        else:
            self._below_med = 0

        if self._stable_level != "HIGH" and self._consec_high >= self.high_confirm_frames:
            self._stable_level = "HIGH"
            self._level_hold = self.cooldown_frames
            return self._stable_level

        if self._stable_level == "NONE" and raw_level == "LOW" and self._consec_low >= self.low_confirm_frames:
            self._stable_level = "LOW"
            self._level_hold = self.cooldown_frames
            return self._stable_level

        if self._stable_level == "NONE" and self._consec_med >= self.medium_confirm_frames:
            self._stable_level = "MEDIUM"
            self._level_hold = self.cooldown_frames
            return self._stable_level

        if self._stable_level == "LOW" and self._consec_med >= self.medium_confirm_frames:
            self._stable_level = "MEDIUM"
            self._level_hold = self.cooldown_frames
            return self._stable_level

        # De-escalation with persistence
        if self._stable_level == "HIGH" and self._below_high >= self.deescalate_frames:
            self._stable_level = "MEDIUM" if smooth_score > 30 else "LOW" if smooth_score > 0 else "NONE"
            self._level_hold = self.cooldown_frames
            return self._stable_level

        if self._stable_level == "MEDIUM" and self._below_med >= self.deescalate_frames:
            self._stable_level = "LOW" if smooth_score > 0 else "NONE"
            self._level_hold = self.cooldown_frames
            return self._stable_level

        if self._stable_level == "LOW" and smooth_score <= 0:
            self._stable_level = "NONE"
            return self._stable_level

        return self._stable_level

    def _explain(
        self,
        *,
        time_seconds: float,
        score: float,
        level: str,
        confidence: float,
        norm: Dict[str, float],
        contributions: Dict[str, float],
    ) -> Tuple[str, List[str], str]:
        def fmt_pct(x: float) -> str:
            return f"{x * 100.0:.0f}%"

        ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        primary_key = ranked[0][0] if ranked else "motion_speed"

        cause_map = {
            "motion_speed": "Rapid crowd acceleration / deceleration",
            "directional_chaos": "Directional instability (movement chaos)",
            "density_change_rate": "Rapid change in crowd density proxy",
            "spread": "Large affected area (widespread motion)",
            "persistence": "Sustained anomaly persistence",
        }
        primary_cause = cause_map.get(primary_key, primary_key)

        def supporting_line(key: str, value: float) -> str:
            # Keep language close to the prompt while staying truthful.
            if key == "density_change_rate":
                return (
                    "Density increase over threshold"
                    if value >= 0.55
                    else "Density increase below threshold"
                )
            if key == "directional_chaos":
                return (
                    "Directional instability detected"
                    if value >= 0.55
                    else "Directional instability below threshold"
                )
            if key == "motion_speed":
                return (
                    "Rapid crowd acceleration detected"
                    if value >= 0.55
                    else "Crowd acceleration not dominant"
                )
            if key == "spread":
                return (
                    "Widespread area affected"
                    if value >= 0.55
                    else "Localized impact (spread below threshold)"
                )
            if key == "persistence":
                return (
                    "Anomaly persistence sustained"
                    if value >= 0.55
                    else "Anomaly persistence short-lived"
                )
            return f"{cause_map.get(key, key)}: {fmt_pct(value)}"

        # Always print supporting factors (even if they are below threshold) to satisfy
        # auditability; the line text explicitly indicates whether it supported or not.
        ranked_norm = sorted(
            [(k, float(v)) for k, v in norm.items() if k != primary_key],
            key=lambda kv: kv[1],
            reverse=True,
        )
        # Report top 3 secondary signals for consistency.
        supporting: List[str] = [supporting_line(k, v) for k, v in ranked_norm[:3]]

        # Always reflect actual computed signals (no copy-paste).
        explanation = (
            f"Time: {_sec_to_hhmmss(time_seconds)}\n"
            f"Risk Score: {score:.0f}\n"
            f"Risk Level: {level}\n"
            f"Primary Cause: {primary_cause}\n"
            f"Supporting Factors:\n" + ("\n".join(f"- {s}" for s in supporting) if supporting else "- None") + "\n"
            f"Confidence: {confidence:.2f}"
        )

        return primary_cause, supporting, explanation

    def update(self, *, time_seconds: float, signals: RiskSignals) -> RiskDecision:
        return self.decide(time_seconds=time_seconds, signals=signals)

    def decide(self, *, time_seconds: float, signals: RiskSignals) -> RiskDecision:
        norm = self._normalize_signals(signals)

        # False-alarm awareness: require multi-signal agreement for MEDIUM/HIGH.
        agreement = self._agreement_count(norm)

        raw_score, contributions = self._compute_score(norm)

        # Spike suppression: if only 1 strong signal and not persistent, damp the score.
        if agreement < self.require_agreement and signals.persistence_frames < max(2, self.medium_confirm_frames):
            raw_score *= 0.55

        raw_score = float(_clamp(raw_score, 0.0, 100.0))
        raw_level = _risk_level_from_score(raw_score)

        self._scores.append(raw_score)
        self._raw_levels.append(raw_level)

        smooth_score = self._smooth_score()
        smooth_level = _risk_level_from_score(smooth_score)
        stable_level = self._stabilize_level(smooth_level, smooth_score)

        # Confidence: higher when (a) stronger score, (b) more signal agreement, (c) stable window.
        variance = self._window_variance()
        # normalize variance: 0..~(30^2) typical; clamp to prevent over-penalizing.
        stability = 1.0 - _clamp(variance / (30.0 * 30.0), 0.0, 1.0)
        agree_factor = _clamp(agreement / max(2.0, float(len(norm))), 0.0, 1.0)
        pers_factor = _clamp(signals.persistence_frames / 6.0, 0.0, 1.0)

        base = _clamp(smooth_score / 100.0, 0.0, 1.0)
        confidence = _clamp((0.55 * base) + (0.20 * agree_factor) + (0.15 * stability) + (0.10 * pers_factor), 0.0, 1.0)

        primary_cause, supporting, explanation = self._explain(
            time_seconds=time_seconds,
            score=smooth_score,
            level=stable_level if stable_level != "NONE" else smooth_level,
            confidence=confidence,
            norm=norm,
            contributions=contributions,
        )

        return RiskDecision(
            time_seconds=float(time_seconds),
            risk_score=float(_clamp(smooth_score, 0.0, 100.0)),
            risk_level=stable_level if stable_level != "NONE" else smooth_level,
            confidence=float(confidence),
            primary_cause=primary_cause,
            supporting_factors=supporting,
            explanation=explanation,
            signals={
                "densityChangeRate": float(norm["density_change_rate"]),
                "motionSpeed": float(norm["motion_speed"]),
                "directionalChaos": float(norm["directional_chaos"]),
                "spread": float(norm["spread"]),
                "persistence": float(norm["persistence"]),
                "signalAgreementCount": float(agreement),
            },
        )
