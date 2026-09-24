"""
MECHGUARD V2 — Health Engine

Evaluates machine health from:
    - sensor risk scores (temperature, current, vibration)
    - online anomaly score
    - trend information
    - persistence of abnormal observations

Key design principles
─────────────────────
1. A single threshold crossing does NOT immediately indicate a problem.
2. Persistence counters determine whether a condition is TRANSIENT or PERSISTENT.
3. intervention_required is raised before shutdown_allowed.
4. shutdown_allowed is only set when a CRITICAL condition is PERSISTENT.

Persistence thresholds are passed in from config at construction time
so they remain configurable without touching engine logic.
"""

from __future__ import annotations

import math
from typing import Optional


# ============================================================
# SENSOR RISK BANDS
# ============================================================

_TEMP_NORMAL   = 40.0
_TEMP_WARNING  = 55.0
_TEMP_CRITICAL = 70.0

_CURRENT_NORMAL   = 5.0
_CURRENT_WARNING  = 8.0
_CURRENT_CRITICAL = 12.0

_VIBRATION_NORMAL   = 0.20
_VIBRATION_WARNING  = 0.50
_VIBRATION_CRITICAL = 0.80


# ============================================================
# COMBINED RISK → STATE BANDS
# ============================================================
# combined_risk is a [0, 1] weighted sum of sensor + anomaly
# NORMAL   < 0.25
# WATCH    < 0.40
# WARNING  < 0.65
# CRITICAL >= 0.65

_BAND_NORMAL   = 0.25
_BAND_WATCH    = 0.40
_BAND_WARNING  = 0.65


# ============================================================
# PERSISTENCE ANOMALY CLASSIFICATION
# ============================================================

_PERSISTENCE_STATE_NORMAL     = "NORMAL"
_PERSISTENCE_STATE_TRANSIENT  = "TRANSIENT_ANOMALY"
_PERSISTENCE_STATE_PERSISTENT = "PERSISTENT_ANOMALY"
_PERSISTENCE_STATE_CRITICAL   = "CRITICAL_PERSISTENT"
_PERSISTENCE_STATE_RECOVERING = "RECOVERING"
_PERSISTENCE_STATE_RECOVERED  = "RECOVERED"


# ============================================================
# HEALTH ENGINE
# ============================================================

class HealthEngine:
    """
    Stateful health assessment engine.

    Unlike V1 (pure function), V2 keeps internal persistence
    counters so transient spikes do not trigger intervention.
    """

    def __init__(
        self,
        warning_persistence_samples:  int = 3,
        critical_persistence_samples: int = 5,
        recovery_samples:             int = 3,
    ):
        self.warning_persistence_samples  = warning_persistence_samples
        self.critical_persistence_samples = critical_persistence_samples
        self.recovery_samples             = recovery_samples

        # Rolling counters
        self._consecutive_warning  = 0
        self._consecutive_critical = 0
        self._consecutive_normal   = 0

        # Last known raw state (before persistence adjustment)
        self._last_raw_state = "NORMAL"

    # =========================================================
    # EVALUATE
    # =========================================================

    def evaluate(
        self,
        temperature_c:          float,
        current_a:              float,
        vibration_frequency_hz: float,
        online_anomaly_score:   float = 0.0,
        online_learning_active: bool  = False,
        model_samples:          int   = 0,
        trend:                  Optional[dict] = None,
    ) -> dict:
        """
        Assess machine health for one observation.

        Returns a rich dict that feeds the InterventionEngine and
        the API response.
        """

        temperature_c = _finite(temperature_c, 0.0)
        current_a = max(0.0, _finite(current_a, 0.0))
        vibration_frequency_hz = max(
            0.0, _finite(vibration_frequency_hz, 0.0)
        )
        online_anomaly_score = min(
            1.0, max(0.0, _finite(online_anomaly_score, 0.0))
        )

        # -------------------------------------------------
        # INDIVIDUAL SENSOR RISKS [0, 1]
        # -------------------------------------------------

        temperature_risk = _risk(
            temperature_c,
            _TEMP_NORMAL,
            _TEMP_WARNING,
            _TEMP_CRITICAL,
        )
        current_risk = _risk(
            current_a,
            _CURRENT_NORMAL,
            _CURRENT_WARNING,
            _CURRENT_CRITICAL,
        )
        vibration_risk = _risk(
            vibration_frequency_hz,
            _VIBRATION_NORMAL,
            _VIBRATION_WARNING,
            _VIBRATION_CRITICAL,
        )

        # -------------------------------------------------
        # COMBINED SENSOR RISK
        # -------------------------------------------------

        sensor_risk = (
            temperature_risk * 0.35
            + current_risk   * 0.30
            + vibration_risk * 0.35
        )

        # -------------------------------------------------
        # ANOMALY WEIGHT  (only once model is initialised)
        # -------------------------------------------------

        anomaly_weight = 0.20 if online_learning_active else 0.0

        combined_risk = (
            sensor_risk * (1.0 - anomaly_weight)
            + online_anomaly_score * anomaly_weight
        )
        combined_risk = max(0.0, min(1.0, combined_risk))

        health_score  = round(100.0 * (1.0 - combined_risk), 2)

        # -------------------------------------------------
        # RAW STATE from instantaneous combined risk
        # -------------------------------------------------

        if   combined_risk < _BAND_NORMAL:  raw_state = "NORMAL"
        elif combined_risk < _BAND_WATCH:   raw_state = "WATCH"
        elif combined_risk < _BAND_WARNING: raw_state = "WARNING"
        else:                               raw_state = "CRITICAL"

        self._last_raw_state = raw_state

        # -------------------------------------------------
        # PERSISTENCE COUNTERS
        # -------------------------------------------------

        if raw_state in ("CRITICAL", "WARNING"):
            self._consecutive_normal = 0
            if raw_state == "CRITICAL":
                self._consecutive_critical += 1
                self._consecutive_warning  += 1   # WARNING also ticking
            else:
                self._consecutive_critical  = 0
                self._consecutive_warning  += 1
        elif raw_state == "WATCH":
            self._consecutive_critical  = 0
            self._consecutive_warning   = 0
            self._consecutive_normal   += 1
        else:  # NORMAL
            self._consecutive_critical  = 0
            self._consecutive_warning   = 0
            self._consecutive_normal   += 1

        # -------------------------------------------------
        # PERSISTENCE STATE
        # -------------------------------------------------

        if raw_state == "NORMAL" or raw_state == "WATCH":

            if self._consecutive_normal >= self.recovery_samples:
                persistence_state = _PERSISTENCE_STATE_RECOVERED
            else:
                persistence_state = _PERSISTENCE_STATE_NORMAL

        elif (
            self._consecutive_critical
            >= self.critical_persistence_samples
        ):
            persistence_state = _PERSISTENCE_STATE_CRITICAL

        elif self._consecutive_warning >= self.warning_persistence_samples:
            persistence_state = _PERSISTENCE_STATE_PERSISTENT

        else:
            persistence_state = _PERSISTENCE_STATE_TRANSIENT

        # -------------------------------------------------
        # EFFECTIVE STATE (persistence-adjusted)
        # -------------------------------------------------

        if persistence_state == _PERSISTENCE_STATE_CRITICAL:
            effective_state = "CRITICAL"
        elif persistence_state == _PERSISTENCE_STATE_PERSISTENT:
            effective_state = "WARNING"
        elif persistence_state == _PERSISTENCE_STATE_TRANSIENT:
            effective_state = "WATCH"
        elif persistence_state == _PERSISTENCE_STATE_RECOVERED:
            effective_state = "NORMAL"
        else:
            effective_state = raw_state

        # -------------------------------------------------
        # INTERVENTION / SHUTDOWN FLAGS
        # -------------------------------------------------

        intervention_required = effective_state in ("WARNING", "CRITICAL")
        shutdown_allowed      = (
            persistence_state == _PERSISTENCE_STATE_CRITICAL
        )

        # -------------------------------------------------
        # TREND SUMMARY
        # -------------------------------------------------

        trend_summary = _build_trend_summary(
            trend,
            temperature_risk,
            current_risk,
            vibration_risk,
        )

        # -------------------------------------------------
        # PRIMARY CONTRIBUTOR
        # -------------------------------------------------

        primary_contributor = _primary(
            temperature_risk,
            current_risk,
            vibration_risk,
            online_anomaly_score,
        )

        # -------------------------------------------------
        # RECOMMENDED ACTION & MAINTENANCE
        # -------------------------------------------------

        action, maintenance = _recommendations(
            effective_state,
            primary_contributor,
            trend,
        )

        # -------------------------------------------------
        # HEALTH STATUS / CONDITION STRINGS
        # -------------------------------------------------

        health_status = {
            "NORMAL":   "GOOD",
            "WATCH":    "FAIR",
            "WARNING":  "WARNING",
            "CRITICAL": "CRITICAL",
        }.get(effective_state, "GOOD")

        condition = {
            "NORMAL":   "NORMAL",
            "WATCH":    "ELEVATED",
            "WARNING":  "WARNING",
            "CRITICAL": "CRITICAL",
        }.get(effective_state, "NORMAL")

        severity = {
            "NORMAL":   "LOW",
            "WATCH":    "MEDIUM",
            "WARNING":  "HIGH",
            "CRITICAL": "CRITICAL",
        }.get(effective_state, "LOW")

        return {
            # Core scores
            "health_score":       health_score,
            "anomaly_score":      round(combined_risk, 4),

            # State strings
            "state":              effective_state,
            "raw_state":          raw_state,
            "health_status":      health_status,
            "condition":          condition,
            "severity":           severity,
            "persistence_state":  persistence_state,

            # Persistence counters (useful for debugging/API)
            "consecutive_warning":  self._consecutive_warning,
            "consecutive_critical": self._consecutive_critical,
            "consecutive_normal":   self._consecutive_normal,

            # Individual risks
            "temperature_risk":   round(temperature_risk, 4),
            "current_risk":       round(current_risk, 4),
            "vibration_risk":     round(vibration_risk, 4),

            # Intelligence
            "primary_contributor":     primary_contributor,
            "trend_summary":           trend_summary,
            "recommended_action":      action,
            "recommended_maintenance": maintenance,

            # Decision flags
            "intervention_required": intervention_required,
            "shutdown_allowed":      shutdown_allowed,

            # Online ML context
            "online_learning_active": online_learning_active,
            "model_samples":          model_samples,
            "learned_baseline":       model_samples >= 30,
        }

    # =========================================================
    # RESET PERSISTENCE
    # =========================================================

    def reset_persistence(self) -> None:
        """Call when intervention begins so recovery is tracked fresh."""
        self._consecutive_warning  = 0
        self._consecutive_critical = 0
        self._consecutive_normal   = 0


# ============================================================
# HELPERS
# ============================================================

def _risk(
    value:    float,
    normal:   float,
    warning:  float,
    critical: float,
) -> float:
    """Piecewise linear risk: 0 at normal, 1 at critical."""
    value = _finite(value, normal)
    if value <= normal:
        return 0.0
    if value >= critical:
        return 1.0
    if value <= warning:
        return 0.35 * (value - normal) / (warning - normal)
    return 0.35 + 0.65 * (value - warning) / (critical - warning)


def _finite(value: float, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _primary(
    temp_risk:    float,
    current_risk: float,
    vib_risk:     float,
    anomaly:      float,
) -> str:
    scores = {
        "temperature": temp_risk,
        "current":     current_risk,
        "vibration":   vib_risk,
        "anomaly":     anomaly * 0.5,   # normalised weight
    }
    return max(scores, key=lambda k: scores[k])


def _build_trend_summary(
    trend:        Optional[dict],
    temp_risk:    float,
    current_risk: float,
    vib_risk:     float,
) -> str:
    if trend is None:
        return "Trend data not yet available."

    parts: list[str] = []

    if trend.get("temperature_direction") == "RISING" and temp_risk > 0.1:
        parts.append(
            "Temperature is rising faster than the learned "
            "operating pattern."
        )
    if trend.get("current_direction") == "RISING" and current_risk > 0.1:
        parts.append(
            "Current deviation and thermal trend indicate "
            "increasing machine stress."
        )
    if trend.get("vibration_direction") == "RISING" and vib_risk > 0.1:
        parts.append(
            "Vibration events have remained elevated across "
            "multiple observations."
        )
    if (
        trend.get("health_direction") == "FALLING"
        and trend.get("anomaly_direction") == "RISING"
    ):
        parts.append(
            "Health score is falling while the anomaly score "
            "is increasing."
        )

    return " ".join(parts) if parts else "All monitored metrics are stable."


def _recommendations(
    state:       str,
    primary:     str,
    trend:       Optional[dict],
) -> tuple[str, str]:
    """Return (recommended_action, recommended_maintenance)."""

    if state == "NORMAL":
        return (
            "Continue normal operation.",
            "No immediate maintenance required.",
        )

    if state == "WATCH":
        return (
            "Continue operation with increased monitoring frequency.",
            "Review machine performance at next scheduled opportunity.",
        )

    # WARNING or CRITICAL — differentiate by primary contributor

    _action_map = {
        "temperature": (
            "Inspect cooling system and ventilation. "
            "Consider reducing machine load.",
            "Inspect cooling system during the next scheduled maintenance.",
        ),
        "current": (
            "Inspect motor load and electrical connections. "
            "Consider reducing machine load.",
            "Inspect motor load / current draw and electrical connections.",
        ),
        "vibration": (
            "Inspect mechanical mounting and rotating components. "
            "Consider reducing machine load.",
            "Inspect mechanical mounting, bearings or rotating components "
            "during scheduled maintenance.",
        ),
        "anomaly": (
            "Machine behaviour deviates from the learned operating pattern. "
            "Review operating conditions and consider reducing load.",
            "Review operating parameters and inspect machine at next "
            "scheduled maintenance.",
        ),
    }

    action, maintenance = _action_map.get(
        primary,
        (
            "Inspect machine and review abnormal sensor behaviour.",
            "Inspect machine at next scheduled maintenance.",
        ),
    )

    if state == "CRITICAL":
        action = (
            "CRITICAL: "
            + action
            + " Prepare for controlled shutdown if condition persists."
        )

    return action, maintenance
