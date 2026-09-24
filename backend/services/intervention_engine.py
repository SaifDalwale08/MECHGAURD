"""
MECHGUARD V2 — Intervention Engine

Implements the SENSE → LEARN → ASSESS → INTERVENE → VERIFY →
PROTECT → REPORT decision pipeline.

State machine
─────────────
NORMAL            → Continue operation.
WATCH             → Continue with increased monitoring.
WARNING           → Recommend controlled intervention.
INTERVENTION      → Request load reduction; begin recovery window.
RECOVERY_CHECK    → Observe machine response after intervention.
RECOVERING        → Improvement is being confirmed across samples.
RECOVERED         → Machine improved; return to WATCH/NORMAL.
FAILED_RECOVERY   → Recovery failed; critical persistence is evaluated.
CRITICAL          → Severe persistent abnormal condition.
CRITICAL_PERSISTENT → Critical persistence is confirmed.
PROTECTIVE_SHUTDOWN → Only here is relay OFF generated.

Design principles
─────────────────
- A single abnormal sample does NOT change the state.
- Intervention is recommended BEFORE any shutdown consideration.
- Recovery monitoring verifies improvement before returning to normal.
- PROTECTIVE_SHUTDOWN requires CRITICAL state + failed recovery,
  OR an immediate CRITICAL condition that has persisted long enough.
- The relay command is only produced in PROTECTIVE_SHUTDOWN state.

Hardware note
─────────────
MECHGUARD does not have verified variable-speed motor hardware.
"REDUCE_LOAD" is a control recommendation, not an automated command.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


# ============================================================
# STATES
# ============================================================

STATE_NORMAL              = "NORMAL"
STATE_WATCH               = "WATCH"
STATE_WARNING             = "WARNING"
STATE_INTERVENTION        = "INTERVENTION"
STATE_RECOVERY_CHECK      = "RECOVERY_CHECK"
STATE_RECOVERING          = "RECOVERING"
STATE_RECOVERED           = "RECOVERED"
STATE_FAILED_RECOVERY     = "FAILED_RECOVERY"
STATE_CRITICAL            = "CRITICAL"
STATE_CRITICAL_PERSISTENT = "CRITICAL_PERSISTENT"
STATE_PROTECTIVE_SHUTDOWN = "PROTECTIVE_SHUTDOWN"

# ============================================================
# RECOVERY CLASSIFICATION
# ============================================================

RECOVERY_NONE     = "NONE"
RECOVERY_STARTED  = "STARTED"
RECOVERY_POSITIVE = "RECOVERING"
RECOVERY_SUCCESS  = "RECOVERED"
RECOVERY_FAILED   = "FAILED_RECOVERY"


# ============================================================
# INTERVENTION ENGINE
# ============================================================

class InterventionEngine:
    """
    Stateful intervention decision engine.

    Call evaluate() once per sensor observation (after HealthEngine).
    """

    def __init__(
        self,
        intervention_trigger_state:    str = "WARNING",
        recovery_window_samples:       int = 5,
        recovery_success_samples:      int = 3,
        immediate_shutdown_threshold:  float = 0.95,
    ):
        """
        Parameters
        ──────────
        intervention_trigger_state
            HealthEngine effective_state that triggers INTERVENTION.
            Default: "WARNING"

        recovery_window_samples
            How many samples to observe after intervention before
            deciding recovery succeeded or failed.

        recovery_success_samples
            Consecutive improved samples required to confirm RECOVERED.

        immediate_shutdown_threshold
            If combined_risk exceeds this in a CRITICAL state the
            engine may immediately allow shutdown without waiting
            for the full recovery cycle (e.g. thermal runaway).
        """

        self.intervention_trigger_state   = intervention_trigger_state
        self.recovery_window_samples      = recovery_window_samples
        self.recovery_success_samples     = recovery_success_samples
        self.immediate_shutdown_threshold = immediate_shutdown_threshold

        # Current state
        self._state              = STATE_NORMAL
        self._recovery_status    = RECOVERY_NONE
        self._intervention_at:   Optional[datetime] = None
        self._last_transition_at: Optional[datetime] = None

        # Baseline captured at intervention start
        self._baseline_health:   Optional[float] = None
        self._baseline_anomaly:  Optional[float] = None
        self._baseline_temp:     Optional[float] = None
        self._baseline_current:  Optional[float] = None
        self._baseline_vibration: Optional[float] = None

        # Recovery monitoring counters
        self._recovery_samples_seen     = 0
        self._consecutive_improved      = 0
        self._consecutive_deteriorating = 0

    # =========================================================
    # EVALUATE
    # =========================================================

    def evaluate(
        self,
        health_dict:            dict,
        temperature_c:          float,
        current_a:              float,
        vibration_frequency_hz: float,
        trend:                  Optional[dict] = None,
    ) -> dict:
        """
        Advance the intervention state machine for one observation.

        health_dict is the full dict returned by HealthEngine.evaluate().
        """

        effective_state:     str   = health_dict["state"]
        combined_risk:       float = health_dict["anomaly_score"]
        health_score:        float = health_dict["health_score"]
        shutdown_allowed:    bool  = health_dict["shutdown_allowed"]
        intervention_req:    bool  = health_dict["intervention_required"]

        now = datetime.now()

        # -------------------------------------------------
        # IMMEDIATE EXTREME PROTECTION
        # Bypass the full recovery cycle only when risk is
        # catastrophically high (configurable threshold).
        # -------------------------------------------------

        if (
            combined_risk >= self.immediate_shutdown_threshold
            and effective_state == "CRITICAL"
        ):
            self._transition(STATE_PROTECTIVE_SHUTDOWN, now)
            self._recovery_status = RECOVERY_NONE
            return self._result(
                action="PROTECTIVE_SHUTDOWN_IMMEDIATE",
                reason=(
                    f"Extreme risk detected "
                    f"(score {combined_risk:.3f} ≥ "
                    f"{self.immediate_shutdown_threshold}). "
                    "Immediate protective shutdown justified."
                ),
                relay_action="OFF",
                requires_operator=True,
            )

        # -------------------------------------------------
        # STATE MACHINE TRANSITIONS
        # -------------------------------------------------

        if self._state == STATE_NORMAL:
            self._state = self._from_normal(effective_state, now)

        elif self._state == STATE_WATCH:
            self._state = self._from_watch(
                effective_state, intervention_req, now
            )

        elif self._state == STATE_WARNING:
            self._state = self._from_warning(
                effective_state, intervention_req, now
            )

        elif self._state == STATE_INTERVENTION:
            self._state = self._from_intervention(
                effective_state,
                health_score,
                combined_risk,
                temperature_c,
                current_a,
                vibration_frequency_hz,
                now,
            )

        elif self._state == STATE_RECOVERY_CHECK:
            self._state = self._from_recovery_check(
                effective_state,
                health_score,
                combined_risk,
                temperature_c,
                current_a,
                vibration_frequency_hz,
                shutdown_allowed,
                now,
            )

        elif self._state == STATE_RECOVERING:
            self._state = self._from_recovery_check(
                effective_state,
                health_score,
                combined_risk,
                temperature_c,
                current_a,
                vibration_frequency_hz,
                shutdown_allowed,
                now,
            )

        elif self._state == STATE_RECOVERED:
            self._state = self._from_recovered(effective_state, now)

        elif self._state == STATE_CRITICAL:
            self._state = self._from_critical(
                effective_state, shutdown_allowed, now
            )

        elif self._state == STATE_FAILED_RECOVERY:
            self._transition(STATE_CRITICAL, now)

        elif self._state == STATE_CRITICAL_PERSISTENT:
            if shutdown_allowed:
                self._transition(STATE_PROTECTIVE_SHUTDOWN, now)

        elif self._state == STATE_PROTECTIVE_SHUTDOWN:
            # Latch until operator resets
            pass

        # -------------------------------------------------
        # BUILD RESULT
        # -------------------------------------------------

        return self._build_result(
            effective_state,
            combined_risk,
            trend,
        )

    # =========================================================
    # STATE TRANSITION METHODS
    # =========================================================

    def _from_normal(self, effective_state: str, now: datetime) -> str:
        if effective_state == "WATCH":
            self._transition(STATE_WATCH, now)
            return STATE_WATCH
        if effective_state in ("WARNING", "CRITICAL"):
            self._transition(STATE_WARNING, now)
            return STATE_WARNING
        return STATE_NORMAL

    def _from_watch(
        self,
        effective_state: str,
        intervention_req: bool,
        now: datetime,
    ) -> str:
        if effective_state == "NORMAL":
            self._transition(STATE_NORMAL, now)
            return STATE_NORMAL
        if intervention_req:
            self._transition(STATE_WARNING, now)
            return STATE_WARNING
        return STATE_WATCH

    def _from_warning(
        self,
        effective_state:  str,
        intervention_req: bool,
        now:              datetime,
    ) -> str:
        if effective_state == "NORMAL":
            self._transition(STATE_NORMAL, now)
            return STATE_NORMAL
        if effective_state == "WATCH":
            self._transition(STATE_WATCH, now)
            return STATE_WATCH
        if effective_state == "CRITICAL":
            self._transition(STATE_CRITICAL, now)
            return STATE_CRITICAL
        # Stays WARNING → escalate to INTERVENTION
        self._transition(STATE_INTERVENTION, now)
        return STATE_INTERVENTION

    def _from_intervention(
        self,
        effective_state: str,
        health_score:    float,
        combined_risk:   float,
        temperature_c:   float,
        current_a:       float,
        vibration_frequency_hz: float,
        now:             datetime,
    ) -> str:
        # Capture baseline on first entry
        if self._baseline_health is None:
            self._baseline_health  = health_score
            self._baseline_anomaly = combined_risk
            self._baseline_temp    = temperature_c
            self._baseline_current = current_a
            self._baseline_vibration = vibration_frequency_hz
            self._recovery_samples_seen     = 0
            self._consecutive_improved      = 0
            self._consecutive_deteriorating = 0
            self._recovery_status = RECOVERY_STARTED

        # After one observation in INTERVENTION, move to RECOVERY_CHECK
        self._transition(STATE_RECOVERY_CHECK, now)
        return STATE_RECOVERY_CHECK

    def _from_recovery_check(
        self,
        effective_state: str,
        health_score:    float,
        combined_risk:   float,
        temperature_c:   float,
        current_a:       float,
        vibration_frequency_hz: float,
        shutdown_allowed: bool,
        now:              datetime,
    ) -> str:
        self._recovery_samples_seen += 1

        baseline_health = self._baseline_health or 0.0
        baseline_anomaly = self._baseline_anomaly or 1.0
        improvements = sum((
            health_score > baseline_health,
            combined_risk < baseline_anomaly,
            temperature_c < (self._baseline_temp or temperature_c),
            current_a < (self._baseline_current or current_a),
            vibration_frequency_hz < (self._baseline_vibration or vibration_frequency_hz),
        ))
        improved = health_score > baseline_health and improvements >= 2

        if improved:
            self._consecutive_improved      += 1
            self._consecutive_deteriorating  = 0
            self._recovery_status = RECOVERY_POSITIVE
        else:
            self._consecutive_deteriorating += 1
            self._consecutive_improved       = 0

        # Success condition
        if self._consecutive_improved >= self.recovery_success_samples:
            self._recovery_status = RECOVERY_SUCCESS
            self._clear_baseline()
            self._transition(STATE_RECOVERED, now)
            return STATE_RECOVERED

        if improved:
            self._transition(STATE_RECOVERING, now)
            return STATE_RECOVERING

        # Failure condition
        if (
            self._consecutive_deteriorating >= self.recovery_window_samples
            or (shutdown_allowed and self._consecutive_deteriorating >= 2)
        ):
            self._recovery_status = RECOVERY_FAILED
            self._transition(STATE_FAILED_RECOVERY, now)
            return STATE_FAILED_RECOVERY

        # Still in window
        if effective_state == "NORMAL":
            self._recovery_status = RECOVERY_POSITIVE

        return STATE_RECOVERY_CHECK

    def _from_recovered(self, effective_state: str, now: datetime) -> str:
        if effective_state in ("WARNING", "CRITICAL"):
            self._transition(STATE_WARNING, now)
            return STATE_WARNING
        if effective_state == "WATCH":
            self._transition(STATE_WATCH, now)
            return STATE_WATCH
        self._transition(STATE_NORMAL, now)
        return STATE_NORMAL

    def _from_critical(
        self,
        effective_state: str,
        shutdown_allowed: bool,
        now:              datetime,
    ) -> str:
        if shutdown_allowed:
            self._transition(STATE_CRITICAL_PERSISTENT, now)
            return STATE_CRITICAL_PERSISTENT
        return STATE_CRITICAL

    # =========================================================
    # RESULT BUILDER
    # =========================================================

    def _build_result(
        self,
        effective_state: str,
        combined_risk:   float,
        trend:           Optional[dict],
    ) -> dict:

        action, reason, relay_action, requires_operator = (
            _action_for_state(
                self._state,
                effective_state,
                combined_risk,
                self._recovery_status,
                trend,
            )
        )

        return self._result(
            action=action,
            reason=reason,
            relay_action=relay_action,
            requires_operator=requires_operator,
        )

    def _result(
        self,
        action:            str,
        reason:            str,
        relay_action:      str,
        requires_operator: bool,
    ) -> dict:
        return {
            "intervention_state":       self._state,
            "action":                   action,
            "reason":                   reason,
            "severity":                 _severity_for_state(self._state),
            "requires_operator":        requires_operator,
            "relay_action":             relay_action,
            "recovery_check_required":  (
                self._state == STATE_RECOVERY_CHECK
            ),
            "recovery_status":          self._recovery_status,
            "intervention_started_at":  (
                self._intervention_at.isoformat()
                if self._intervention_at else None
            ),
            "last_transition_at": (
                self._last_transition_at.isoformat()
                if self._last_transition_at else None
            ),
        }

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _transition(self, new_state: str, now: datetime) -> None:
        if self._state != new_state:
            self._last_transition_at = now
            if new_state == STATE_INTERVENTION:
                self._intervention_at = now
        self._state = new_state

    def _clear_baseline(self) -> None:
        self._baseline_health  = None
        self._baseline_anomaly = None
        self._baseline_temp    = None
        self._baseline_current = None
        self._baseline_vibration = None
        self._recovery_samples_seen     = 0
        self._consecutive_improved      = 0
        self._consecutive_deteriorating = 0

    # =========================================================
    # OPERATOR RESET
    # =========================================================

    def reset(self) -> None:
        """
        Reset to NORMAL. Call after operator clears a shutdown.
        """
        self._state              = STATE_NORMAL
        self._recovery_status    = RECOVERY_NONE
        self._intervention_at    = None
        self._last_transition_at = None
        self._clear_baseline()

    # =========================================================
    # PROPERTIES
    # =========================================================

    @property
    def state(self) -> str:
        return self._state

    @property
    def recovery_status(self) -> str:
        return self._recovery_status


# ============================================================
# PURE HELPERS
# ============================================================

def _action_for_state(
    state:           str,
    health_state:    str,
    combined_risk:   float,
    recovery_status: str,
    trend:           Optional[dict],
) -> tuple[str, str, str, bool]:
    """
    Returns (action, reason, relay_action, requires_operator).
    relay_action is "OFF" only for PROTECTIVE_SHUTDOWN.
    """

    if state == STATE_NORMAL:
        return (
            "MONITOR",
            "Machine operating within learned normal parameters.",
            "NONE",
            False,
        )

    if state == STATE_WATCH:
        return (
            "INCREASED_MONITORING",
            "Machine behaviour slightly elevated. "
            "Continuing operation with increased monitoring.",
            "NONE",
            False,
        )

    if state == STATE_WARNING:
        return (
            "RECOMMEND_INSPECTION",
            "Persistent abnormal condition detected. "
            "Inspect machine and prepare for possible load reduction.",
            "NONE",
            True,
        )

    if state == STATE_INTERVENTION:
        return (
            "REDUCE_LOAD",
            "Intervention requested: reduce machine load and monitor "
            "response. (Load reduction is a recommendation; no "
            "automatic hardware control is available.)",
            "NONE",
            True,
        )

    if state == STATE_RECOVERY_CHECK:
        if recovery_status == RECOVERY_POSITIVE:
            msg = "Machine condition is improving after intervention."
        else:
            msg = (
                "Monitoring machine response after intervention. "
                "Awaiting improvement confirmation."
            )
        return ("OBSERVE_RECOVERY", msg, "NONE", True)

    if state == STATE_RECOVERED:
        return (
            "RESUME_NORMAL",
            "Machine has recovered. Resuming normal operation "
            "with continued monitoring.",
            "NONE",
            False,
        )

    if state == STATE_CRITICAL:
        if recovery_status == RECOVERY_FAILED:
            reason = (
                "Recovery failed; critical condition persists. "
                "Operator intervention required immediately."
            )
        else:
            reason = (
                "Critical abnormal condition detected and persisting. "
                "Operator intervention required."
            )
        return ("CRITICAL_ALERT", reason, "NONE", True)

    if state == STATE_FAILED_RECOVERY:
        return (
            "CRITICAL_ALERT",
            "Recovery failed; critical condition requires persistent confirmation.",
            "NONE",
            True,
        )

    if state == STATE_CRITICAL_PERSISTENT:
        return (
            "CRITICAL_ALERT",
            "Critical condition is persistent; protective shutdown is authorized.",
            "NONE",
            True,
        )

    if state == STATE_PROTECTIVE_SHUTDOWN:
        return (
            "PROTECTIVE_SHUTDOWN",
            "Critical persistent condition confirmed after failed "
            "recovery. Protective shutdown command issued.",
            "OFF",
            True,
        )

    return ("MONITOR", "State unknown.", "NONE", False)


def _severity_for_state(state: str) -> str:
    return {
        STATE_NORMAL:              "LOW",
        STATE_WATCH:               "LOW",
        STATE_WARNING:             "MEDIUM",
        STATE_INTERVENTION:        "HIGH",
        STATE_RECOVERY_CHECK:      "HIGH",
        STATE_RECOVERING:          "HIGH",
        STATE_RECOVERED:           "LOW",
        STATE_FAILED_RECOVERY:     "CRITICAL",
        STATE_CRITICAL:            "CRITICAL",
        STATE_CRITICAL_PERSISTENT: "CRITICAL",
        STATE_PROTECTIVE_SHUTDOWN: "CRITICAL",
    }.get(state, "LOW")
