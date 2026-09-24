"""
MECHGUARD V2 — Maintenance Intelligence Engine

Generates structured maintenance intelligence from accumulated
machine observations.

This engine does NOT diagnose physical faults.
It summarises trends and suggests inspections.
All recommendations use cautious language:
    "recommend", "inspect", "review", "possible contributor".
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Optional


# ============================================================
# MAINTENANCE ENGINE
# ============================================================

class MaintenanceEngine:
    """
    Accumulates observation events and generates a structured
    maintenance intelligence report.

    Call record() after every HealthEngine + InterventionEngine
    evaluation.  Call report() to retrieve the current report.
    """

    def __init__(
        self,
        machine_id:      str = "MG-M1",
        history_maxlen:  int = 200,
    ):
        self.machine_id     = machine_id
        self._started_at    = datetime.now()
        self._history_maxlen = history_maxlen

        # Aggregate counters
        self._total_samples      = 0
        self._anomaly_events     = 0
        self._warning_events     = 0
        self._intervention_events = 0
        self._recovery_attempts  = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._critical_events    = 0
        self._shutdown_events    = 0

        # Rolling window of (timestamp, health_score, anomaly_score)
        # used for trend judgement
        self._health_window: deque[float] = deque(maxlen=50)
        self._anomaly_window: deque[float] = deque(maxlen=50)

        # Risk accumulation (for dominant risk calculation)
        self._temp_risk_sum    = 0.0
        self._current_risk_sum = 0.0
        self._vib_risk_sum     = 0.0

        # Last recorded values
        self._last_health:        Optional[float] = None
        self._last_anomaly:       Optional[float] = None
        self._last_health_status: str             = "GOOD"
        self._last_intervention_state: str        = "NORMAL"

    # =========================================================
    # RECORD
    # =========================================================

    def record(
        self,
        health_dict:       dict,
        intervention_dict: dict,
    ) -> None:
        """
        Record one observation cycle.

        health_dict       — from HealthEngine.evaluate()
        intervention_dict — from InterventionEngine.evaluate()
        """

        self._total_samples += 1

        health_score:  float = health_dict.get("health_score",  100.0)
        anomaly_score: float = health_dict.get("anomaly_score", 0.0)
        state:         str   = health_dict.get("state",         "NORMAL")
        temp_risk:     float = health_dict.get("temperature_risk", 0.0)
        curr_risk:     float = health_dict.get("current_risk",    0.0)
        vib_risk:      float = health_dict.get("vibration_risk",  0.0)

        intervention_state: str = intervention_dict.get(
            "intervention_state", "NORMAL"
        )
        recovery_status: str = intervention_dict.get(
            "recovery_status", "NONE"
        )

        self._health_window.append(health_score)
        self._anomaly_window.append(anomaly_score)

        self._temp_risk_sum    += temp_risk
        self._current_risk_sum += curr_risk
        self._vib_risk_sum     += vib_risk

        if anomaly_score > 0.25:
            self._anomaly_events += 1

        if state == "WARNING" and self._last_health_status != "WARNING":
            self._warning_events += 1

        if intervention_state == "INTERVENTION" and self._last_intervention_state != "INTERVENTION":
            self._intervention_events += 1

        if intervention_state == "RECOVERY_CHECK" and self._last_intervention_state != "RECOVERY_CHECK":
            self._recovery_attempts += 1

        if intervention_state == "RECOVERED" and self._last_intervention_state != "RECOVERED":
            self._successful_recoveries += 1

        if intervention_state == "FAILED_RECOVERY" and self._last_intervention_state != "FAILED_RECOVERY":
            self._failed_recoveries += 1

        if state == "CRITICAL" and self._last_health_status != "CRITICAL":
            self._critical_events += 1

        if intervention_state == "PROTECTIVE_SHUTDOWN" and self._last_intervention_state != "PROTECTIVE_SHUTDOWN":
            self._shutdown_events += 1

        self._last_health             = health_score
        self._last_anomaly            = anomaly_score
        self._last_health_status      = health_dict.get(
            "health_status", "GOOD"
        )
        self._last_intervention_state = intervention_state

    # =========================================================
    # REPORT
    # =========================================================

    def report(self) -> dict:
        """
        Return the current maintenance intelligence report.
        """

        elapsed = datetime.now() - self._started_at
        minutes = int(elapsed.total_seconds() / 60)
        monitoring_period = (
            f"{minutes} minute{'s' if minutes != 1 else ''}"
            if minutes > 0
            else "< 1 minute"
        )

        health_trend = _health_trend(self._health_window)
        dominant_risk = _dominant_risk(
            self._temp_risk_sum,
            self._current_risk_sum,
            self._vib_risk_sum,
            self._total_samples,
        )
        priority = _maintenance_priority(
            self._critical_events,
            self._warning_events,
            self._intervention_events,
            self._anomaly_events,
            self._total_samples,
            health_trend,
        )
        recommendation = _recommendation(
            dominant_risk,
            priority,
            self._shutdown_events,
        )
        summary = _summary(
            self._total_samples,
            self._anomaly_events,
            self._intervention_events,
            self._successful_recoveries,
            self._critical_events,
            priority,
            health_trend,
            dominant_risk,
        )

        return {
            "machine_id":          self.machine_id,
            "monitoring_period":   monitoring_period,
            "total_samples":       self._total_samples,
            "current_health":      self._last_health,
            "current_health_status": self._last_health_status,
            "health_trend":        health_trend,
            "anomaly_events":      self._anomaly_events,
            "warning_events":      self._warning_events,
            "intervention_events": self._intervention_events,
            "recovery_attempts":   self._recovery_attempts,
            "successful_recoveries": self._successful_recoveries,
            "failed_recoveries":   self._failed_recoveries,
            "recovery_events":     self._successful_recoveries,
            "critical_events":     self._critical_events,
            "shutdown_events":     self._shutdown_events,
            "dominant_risk":       dominant_risk,
            "maintenance_priority": priority,
            "recommended_action":  recommendation,
            "maintenance_summary": summary,
            "generated_at":        datetime.now().isoformat(),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:
        """Reset all counters (e.g. after operator maintenance)."""
        self._started_at    = datetime.now()
        self._total_samples      = 0
        self._anomaly_events     = 0
        self._warning_events     = 0
        self._intervention_events = 0
        self._recovery_attempts  = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._critical_events    = 0
        self._shutdown_events    = 0
        self._health_window.clear()
        self._anomaly_window.clear()
        self._temp_risk_sum    = 0.0
        self._current_risk_sum = 0.0
        self._vib_risk_sum     = 0.0
        self._last_health             = None
        self._last_anomaly            = None
        self._last_health_status      = "GOOD"
        self._last_intervention_state = "NORMAL"


# ============================================================
# PURE HELPERS
# ============================================================

def _health_trend(window: deque) -> str:
    if len(window) < 5:
        return "INSUFFICIENT_DATA"
    recent = list(window)[-5:]
    first  = sum(list(window)[:5]) / 5 if len(window) >= 5 else recent[0]
    last   = sum(recent) / len(recent)
    delta  = last - first
    if delta >  2.0:
        return "IMPROVING"
    if delta < -2.0:
        return "DETERIORATING"
    return "STABLE"


def _dominant_risk(
    temp_sum:    float,
    current_sum: float,
    vib_sum:     float,
    samples:     int,
) -> str:
    if samples == 0:
        return "unknown"
    t = temp_sum    / samples
    c = current_sum / samples
    v = vib_sum     / samples
    best = max(t, c, v)
    if best < 0.05:
        return "none"
    if best == t:
        return "temperature"
    if best == c:
        return "current"
    return "vibration"


def _maintenance_priority(
    critical_events:      int,
    warning_events:       int,
    intervention_events:  int,
    anomaly_events:       int,
    total_samples:        int,
    health_trend:         str,
) -> str:
    if total_samples == 0:
        return "LOW"

    if critical_events > 0 or intervention_events >= 3:
        return "CRITICAL"

    warning_rate = warning_events / max(total_samples, 1)
    anomaly_rate = anomaly_events / max(total_samples, 1)

    if warning_rate > 0.30 or anomaly_rate > 0.40:
        return "HIGH"

    if (
        warning_rate > 0.10
        or anomaly_rate > 0.20
        or health_trend == "DETERIORATING"
    ):
        return "MEDIUM"

    return "LOW"


def _recommendation(
    dominant_risk: str,
    priority:      str,
    shutdown_events: int,
) -> str:

    if shutdown_events > 0:
        return (
            "Protective shutdown has occurred. "
            "Operator inspection is required before restarting the machine. "
            "Review all sensor logs and inspect the machine thoroughly."
        )

    if priority == "CRITICAL":
        _map = {
            "temperature": (
                "Immediate inspection of the cooling system and "
                "thermal management is recommended."
            ),
            "current": (
                "Immediate inspection of motor load, electrical "
                "connections, and current draw is recommended."
            ),
            "vibration": (
                "Immediate inspection of mechanical mounting, "
                "bearings, and rotating components is recommended."
            ),
        }
        return _map.get(
            dominant_risk,
            "Immediate machine inspection is recommended.",
        )

    if priority == "HIGH":
        _map = {
            "temperature": (
                "Inspect cooling system during the next scheduled "
                "maintenance."
            ),
            "current": (
                "Inspect motor load / current draw and electrical "
                "connections during the next scheduled maintenance."
            ),
            "vibration": (
                "Inspect mechanical mounting, bearings or rotating "
                "components during scheduled maintenance."
            ),
        }
        return _map.get(
            dominant_risk,
            "Review machine performance at next scheduled maintenance.",
        )

    if priority == "MEDIUM":
        return (
            "Monitor machine performance closely. "
            "Review operating conditions at next scheduled maintenance."
        )

    return "No immediate maintenance action required. Continue monitoring."


def _summary(
    total:         int,
    anomalies:     int,
    interventions: int,
    recoveries:    int,
    criticals:     int,
    priority:      str,
    health_trend:  str,
    dominant_risk: str,
) -> str:
    if total == 0:
        return "No observations recorded yet."

    parts = [
        f"MECHGUARD has monitored {total} observations.",
    ]

    if anomalies > 0:
        parts.append(
            f"{anomalies} anomalous reading"
            f"{'s were' if anomalies > 1 else ' was'} detected."
        )
    else:
        parts.append("No significant anomalies detected.")

    if interventions > 0:
        parts.append(
            f"{interventions} intervention recommendation"
            f"{'s were' if interventions > 1 else ' was'} issued."
        )

    if recoveries > 0:
        parts.append(
            f"Machine recovered successfully "
            f"{recoveries} time{'s' if recoveries > 1 else ''}."
        )

    if criticals > 0:
        parts.append(
            f"{criticals} critical condition"
            f"{'s were' if criticals > 1 else ' was'} observed."
        )

    trend_phrase = {
        "IMPROVING":       "Overall health trend is improving.",
        "DETERIORATING":   "Overall health trend is deteriorating.",
        "STABLE":          "Overall health trend is stable.",
        "INSUFFICIENT_DATA": "",
    }.get(health_trend, "")

    if trend_phrase:
        parts.append(trend_phrase)

    if dominant_risk not in ("none", "unknown"):
        parts.append(
            f"The primary possible contributor to observed risk is "
            f"{dominant_risk}."
        )

    parts.append(f"Maintenance priority: {priority}.")

    return " ".join(parts)
