"""
MECHGUARD V2 — Machine State Store

Holds the current MachineState and a bounded deque of
HistoryRecord observations.

History is capped at MAX_HISTORY entries to prevent unbounded
memory growth during continuous operation.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Optional

from backend.schemas.machine import (
    HistoryRecord,
    MachineHealth,
    MachineState,
)
from backend.config import SENSOR_TIMEOUT_SECONDS

MAX_HISTORY = 200


class MachineStateStore:

    def __init__(
        self,
        history_maxlen: int = MAX_HISTORY,
        sensor_timeout_seconds: float = SENSOR_TIMEOUT_SECONDS,
    ):
        self.sensor_timeout_seconds = sensor_timeout_seconds
        self.state = MachineState(
            machine_id="MG-M1",
            connected=False,
        )
        self._history: deque[HistoryRecord] = deque(
            maxlen=history_maxlen
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        temperature_c:          float,
        current_a:              float,
        vibration_detected:     bool,
        vibration_event_count:  int,
        vibration_frequency_hz: float,
        health:                 dict,
        relay_active:           bool  = False,
        intervention:           Optional[dict] = None,
        trend:                  Optional[dict] = None,
        learning:               Optional[dict] = None,
    ) -> MachineState:

        now = datetime.now()

        self.state.connected    = True
        self.state.last_update  = now

        self.state.temperature_c          = temperature_c
        self.state.current_a              = current_a
        self.state.vibration_detected     = vibration_detected
        self.state.vibration_event_count  = vibration_event_count
        self.state.vibration_frequency_hz = vibration_frequency_hz
        self.state.relay_active           = relay_active

        # Store structured dicts for intervention / trend / learning
        self.state.intervention = intervention
        self.state.trend        = trend
        self.state.learning     = learning

        # Build MachineHealth from dict
        self.state.health = MachineHealth(
            health_score=health["health_score"],
            anomaly_score=health["anomaly_score"],
            state=health["state"],
            severity=health["severity"],

            # V2 fields (use .get with safe defaults)
            health_status=health.get("health_status", "GOOD"),
            condition=health.get("condition", "NORMAL"),
            persistence_state=health.get("persistence_state", "NORMAL"),
            raw_state=health.get("raw_state", health["state"]),

            temperature_risk=health["temperature_risk"],
            vibration_risk=health["vibration_risk"],
            current_risk=health["current_risk"],

            consecutive_warning=health.get("consecutive_warning",  0),
            consecutive_critical=health.get("consecutive_critical", 0),
            consecutive_normal=health.get("consecutive_normal",    0),

            online_learning_active=health.get(
                "online_learning_active", False
            ),
            model_samples=health.get("model_samples", 0),
            learned_baseline=health.get("learned_baseline", False),

            primary_contributor=health.get("primary_contributor", ""),
            trend_summary=health.get("trend_summary", ""),
            recommended_action=health.get("recommended_action", ""),
            recommended_maintenance=health.get(
                "recommended_maintenance", ""
            ),

            intervention_required=health.get(
                "intervention_required", False
            ),
            shutdown_allowed=health.get("shutdown_allowed", False),
        )

        # Append history record
        intervention_state = (
            intervention.get("intervention_state", "NORMAL")
            if intervention else "NORMAL"
        )
        action = (
            intervention.get("action", "MONITOR")
            if intervention else "MONITOR"
        )
        reason = (
            intervention.get("reason", "")
            if intervention else ""
        )

        self._history.append(
            HistoryRecord(
                timestamp=now,
                temperature_c=temperature_c,
                current_a=current_a,
                vibration_detected=vibration_detected,
                vibration_frequency_hz=vibration_frequency_hz,
                health_score=health["health_score"],
                anomaly_score=health["anomaly_score"],
                health_status=health.get("health_status", "GOOD"),
                intervention_state=intervention_state,
                action=action,
                reason=reason,
            )
        )

        return self.state

    # =========================================================
    # GET
    # =========================================================

    def get(self) -> MachineState:
        if self.state.last_update is not None:
            elapsed = (datetime.now() - self.state.last_update).total_seconds()
            if elapsed > self.sensor_timeout_seconds:
                self.state.connected = False
        return self.state

    # =========================================================
    # HISTORY
    # =========================================================

    def get_history(self, limit: int = 50) -> list[dict]:
        """
        Return the most recent `limit` history records as dicts.
        Most-recent first.
        """
        records = list(self._history)[-limit:]
        records.reverse()
        return [r.model_dump() for r in records]

    def history_len(self) -> int:
        return len(self._history)

    # =========================================================
    # DISCONNECT
    # =========================================================

    def set_disconnected(self) -> None:
        self.state.connected = False
