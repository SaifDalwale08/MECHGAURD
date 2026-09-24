"""
MECHGUARD V2 — API Routes

All existing V1 endpoints are preserved unchanged.
V2 additions:
    GET /api/machine/history
    GET /api/machine/maintenance-report

The live endpoint GET /api/machines/{machine_id}/live is extended
with intervention, maintenance, trend, and operating_pattern fields
while retaining every existing field the frontend depends on.
"""

from datetime import datetime
import math
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.machine import MachineState
from backend.services.feature_engine import FeatureEngine
from backend.services.health_engine import HealthEngine
from backend.services.intervention_engine import InterventionEngine
from backend.services.machine_state import MachineStateStore
from backend.services.maintenance_engine import MaintenanceEngine
from backend.services.trend_engine import TrendEngine
from backend.config import SENSOR_SAMPLE_INTERVAL_SECONDS


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()


# ============================================================
# SHARED BACKEND COMPONENTS
# (These are used by the HTTP sensor-data ingestion path only.
#  MQTT ingestion uses its own instances wired in main.py.)
# ============================================================

feature_engine    = FeatureEngine(
    window_size=50,
    sample_interval_seconds=SENSOR_SAMPLE_INTERVAL_SECONDS,
)
health_engine     = HealthEngine()
machine_store     = MachineStateStore()
trend_engine      = TrendEngine()
intervention_engine = InterventionEngine()
maintenance_engine  = MaintenanceEngine(machine_id="MG-M1")


# ============================================================
# BASIC HEALTH
# ============================================================

@router.get("/api/health")
def api_health():
    return {
        "status":    "online",
        "service":   "MECHGUARD API",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# MACHINE STATE
# ============================================================

@router.get("/api/machine")
def get_machine():
    return machine_store.get()


# ============================================================
# MACHINE HEALTH
# ============================================================

@router.get("/api/machine/health")
def get_machine_health():
    state = machine_store.get()
    if state.health is None:
        return {
            "available": False,
            "message":   "No sensor data received yet.",
        }
    return state.health


# ============================================================
# MACHINE SENSOR DATA
# ============================================================

@router.get("/api/machine/sensors")
def get_machine_sensors():
    state = machine_store.get()
    return {
        "temperature_c":          state.temperature_c,
        "current_a":              state.current_a,
        "vibration_detected":     state.vibration_detected,
        "vibration_event_count":  state.vibration_event_count,
        "vibration_frequency_hz": state.vibration_frequency_hz,
        "timestamp": (
            state.last_update.isoformat()
            if state.last_update else None
        ),
    }


# ============================================================
# V2 — HISTORY
# ============================================================

@router.get("/api/machine/history")
def get_machine_history(limit: int = 50):
    """
    Return the most recent `limit` observation records.
    Records are ordered most-recent first.
    Maximum limit: 200 (bounded by MACHINE_HISTORY_MAXLEN).
    """
    limit = min(max(1, limit), 200)
    return {
        "machine_id": "MG-M1",
        "count":      machine_store.history_len(),
        "limit":      limit,
        "records":    machine_store.get_history(limit=limit),
    }


# ============================================================
# V2 — MAINTENANCE REPORT
# ============================================================

@router.get("/api/machine/maintenance-report")
def get_maintenance_report():
    """
    Return the current maintenance intelligence report.
    """
    return maintenance_engine.report()


# ============================================================
# MANUAL / HTTP SENSOR INGESTION
# ============================================================

@router.post("/api/sensor-data")
def receive_sensor_data(payload: dict[str, Any]):
    try:
        temperature_c = float(payload["temperature_c"])
        current_a     = float(payload["current_a"])
        if not math.isfinite(temperature_c) or not math.isfinite(current_a):
            raise ValueError("Sensor values must be finite")
        if not -50.0 <= temperature_c <= 150.0:
            raise ValueError("temperature_c is outside the supported range")
        if not 0.0 <= current_a <= 100.0:
            raise ValueError("current_a is outside the supported range")
        if not isinstance(payload["vibration_detected"], bool):
            raise ValueError("vibration_detected must be boolean")
        vibration_detected = _parse_bool(
            payload.get("vibration_detected", False)
        )

        # Feature extraction
        features = feature_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
        )

        # Trend (no online learning on the HTTP path)
        trend = trend_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features["vibration_frequency_hz"],
            health_score=50.0,   # placeholder before first health calc
            anomaly_score=0.0,
        )

        # Health assessment
        health_dict = health_engine.evaluate(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features["vibration_frequency_hz"],
            online_anomaly_score=0.0,
            online_learning_active=False,
            model_samples=0,
            trend=trend,
        )

        # Re-update trend with actual health score
        trend = trend_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features["vibration_frequency_hz"],
            health_score=health_dict["health_score"],
            anomaly_score=health_dict["anomaly_score"],
        )

        # Intervention decision
        intervention = intervention_engine.evaluate(
            health_dict=health_dict,
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features["vibration_frequency_hz"],
            trend=trend,
        )

        relay_active = (
            intervention["intervention_state"] == "PROTECTIVE_SHUTDOWN"
            and intervention["relay_action"] == "OFF"
        )

        # Update state
        state = machine_store.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
            vibration_event_count=features["vibration_event_count"],
            vibration_frequency_hz=features["vibration_frequency_hz"],
            health=health_dict,
            relay_active=relay_active,
            intervention=intervention,
            trend=trend,
        )

        # Record for maintenance intelligence
        maintenance_engine.record(
            health_dict=health_dict,
            intervention_dict=intervention,
        )

        return {"success": True, "machine": state}

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ============================================================
# FRONTEND LIVE ENDPOINT  — GET /api/machines/{machine_id}/live
# ============================================================

@router.get("/api/machines/{machine_id}/live")
def get_live_snapshot(machine_id: str):

    if machine_id != "MG-M1":
        raise HTTPException(status_code=404, detail="Machine not found")

    state     = machine_store.get()
    connected = state.connected
    health    = state.health
    iv        = state.intervention   # intervention dict or None
    trend     = state.trend          # trend dict or None
    learning  = state.learning       # online-learning dict or None

    # ----------------------------------------------------------
    # HEALTH FIELDS
    # ----------------------------------------------------------

    if health is None:
        health_score       = None
        health_status      = "GOOD"
        anomaly_score      = None
        condition          = "NORMAL"
        anomaly_status     = "LOW"
        operating_pattern  = "LEARNING_BASELINE"
        primary_contributor = ""
        trend_summary      = ""
        recommended_action = "Waiting for sensor data."
        temperature_risk   = 0.0
        vibration_risk     = 0.0
        current_risk       = 0.0
        intervention_required = False
        shutdown_allowed      = False
        health_trend_dir      = "STABLE"
    else:
        health_score         = health.health_score
        health_status        = health.health_status
        anomaly_score        = health.anomaly_score
        condition            = health.condition
        anomaly_status       = _map_anomaly_status(health.anomaly_score)
        operating_pattern    = (
            learning.get("operating_pattern", "LEARNING_BASELINE")
            if learning else "LEARNING_BASELINE"
        )
        primary_contributor  = health.primary_contributor
        trend_summary        = health.trend_summary
        recommended_action   = health.recommended_action
        temperature_risk     = health.temperature_risk
        vibration_risk       = health.vibration_risk
        current_risk         = health.current_risk
        intervention_required = health.intervention_required
        shutdown_allowed      = health.shutdown_allowed
        health_trend_dir = (
            trend.get("health_direction", "STABLE")
            if trend else "STABLE"
        )

    # ----------------------------------------------------------
    # SENSOR STATUS
    # ----------------------------------------------------------

    if connected:
        temperature_status = _temperature_sensor_status(
            state.temperature_c
        )
        current_status     = _current_sensor_status(state.current_a)
        vibration_status   = "OK"
        sw420_status       = "OK"
        acs712_status      = "OK"
        ds18b20_status     = "OK"
    else:
        temperature_status = "NO_DATA"
        current_status     = "NO_DATA"
        vibration_status   = "NO_DATA"
        sw420_status       = "NO_DATA"
        acs712_status      = "NO_DATA"
        ds18b20_status     = "NO_DATA"

    # ----------------------------------------------------------
    # VIBRATION STATE
    # ----------------------------------------------------------

    vibration_state = (
        "VIBRATION" if state.vibration_detected else "NO_VIBRATION"
    )

    # ----------------------------------------------------------
    # RELAY STATE
    # ----------------------------------------------------------

    if not connected:
        relay_state = "OFFLINE"
    elif state.relay_active:
        relay_state = "TRIGGERED"
    else:
        relay_state = "ARMED"

    # ----------------------------------------------------------
    # INTERVENTION BLOCK
    # ----------------------------------------------------------

    if iv:
        intervention_block = {
            "state":            iv.get("intervention_state", "NORMAL"),
            "action":           iv.get("action",             "MONITOR"),
            "reason":           iv.get("reason",             ""),
            "recovery_status":  iv.get("recovery_status",    "NONE"),
            "shutdown_allowed": shutdown_allowed,
            "relay_action":     iv.get("relay_action",       "NONE"),
            "requires_operator": iv.get("requires_operator", False),
        }
    else:
        intervention_block = {
            "state":            "NORMAL",
            "action":           "MONITOR",
            "reason":           "",
            "recovery_status":  "NONE",
            "shutdown_allowed": False,
            "relay_action":     "NONE",
            "requires_operator": False,
        }

    # ----------------------------------------------------------
    # TREND BLOCK
    # ----------------------------------------------------------

    if trend:
        trend_block = {
            "temperature_direction": trend.get(
                "temperature_direction", "STABLE"
            ),
            "current_direction": trend.get(
                "current_direction", "STABLE"
            ),
            "vibration_direction": trend.get(
                "vibration_direction", "STABLE"
            ),
            "health_direction": trend.get(
                "health_direction", "STABLE"
            ),
            "anomaly_direction": trend.get(
                "anomaly_direction", "STABLE"
            ),
            "summary": trend_summary,
        }
    else:
        trend_block = {
            "temperature_direction": "STABLE",
            "current_direction":     "STABLE",
            "vibration_direction":   "STABLE",
            "health_direction":      "STABLE",
            "anomaly_direction":     "STABLE",
            "summary":               "Trend data not yet available.",
        }

    # ----------------------------------------------------------
    # MAINTENANCE BLOCK
    # ----------------------------------------------------------

    maint_report = maintenance_engine.report()
    maintenance_block = {
        "priority":       maint_report["maintenance_priority"],
        "recommendation": maint_report["recommended_action"],
        "summary":        maint_report["maintenance_summary"],
    }

    # ----------------------------------------------------------
    # ALERTS
    # ----------------------------------------------------------

    alerts = []
    if health is not None:
        if health.state == "CRITICAL":
            alerts.append({
                "id":       "critical-machine-condition",
                "severity": "CRITICAL",
                "title":    "Critical machine condition",
                "message":  recommended_action,
            })
        elif health.state == "WARNING":
            alerts.append({
                "id":       "machine-warning",
                "severity": "WARNING",
                "title":    "Machine warning",
                "message":  recommended_action,
            })
        elif health.state == "WATCH":
            alerts.append({
                "id":       "machine-watch",
                "severity": "INFO",
                "title":    "Machine requires monitoring",
                "message":  recommended_action,
            })

    if iv and iv.get("intervention_state") == "INTERVENTION":
        alerts.append({
            "id":       "intervention-requested",
            "severity": "WARNING",
            "title":    "Controlled intervention recommended",
            "message":  iv.get("reason", ""),
        })

    if iv and iv.get("intervention_state") == "PROTECTIVE_SHUTDOWN":
        alerts.append({
            "id":       "protective-shutdown",
            "severity": "CRITICAL",
            "title":    "Protective shutdown",
            "message":  iv.get("reason", ""),
        })

    # ----------------------------------------------------------
    # FULL RESPONSE
    # ----------------------------------------------------------

    return {
        # ── existing V1 fields (unchanged) ──────────────────
        "machine": {
            "id":         "MG-M1",
            "name":       "Machine 1",
            "status":     "RUNNING" if connected else "STOPPED",
            "connection": "CONNECTED" if connected else "DISCONNECTED",
        },

        "sensors": {
            "temperature": {
                "value": state.temperature_c if connected else None,
                "status": temperature_status,
            },
            "current": {
                "value": state.current_a if connected else None,
                "status": current_status,
            },
            "vibration": {
                "state":       vibration_state,
                "event_count": state.vibration_event_count,
                "frequency_hz": state.vibration_frequency_hz,
                "status":      vibration_status,
            },
        },

        "health": {
            "score":  health_score,
            "status": health_status,
            # V2 additions
            "trend":  health_trend_dir,
            "reason": trend_summary,
        },

        "analytics": {
            "condition":     condition,
            "anomaly_score": anomaly_score,
            "anomaly_status": anomaly_status,
            # V2 additions
            "operating_pattern":  operating_pattern,
            "trend":              trend_block,
            "primary_contributor": primary_contributor,
        },

        "protection": {
            "relay":  relay_state,
            "mode":   "AUTO",
            "reason": (
                "Critical condition detected"
                if state.relay_active else None
            ),
            "triggeredBy": {
                "temperature": temperature_risk >= 0.80,
                "vibration":   vibration_risk   >= 0.80,
                "current":     current_risk     >= 0.80,
            },
            "lastChangedAt": (
                state.last_update.isoformat()
                if state.relay_active and state.last_update
                else None
            ),
        },

        "relayEvents": [],

        "alerts": alerts,

        "system": {
            "esp8266": "CONNECTED" if connected else "DISCONNECTED",
            "backend": "ONLINE",
            "api":     "CONNECTED",
            "relay":   relay_state,
            "sensors": {
                "sw420":   sw420_status,
                "acs712":  acs712_status,
                "ds18b20": ds18b20_status,
            },
        },

        "prediction": {
            "available":      False,
            "model":          None,
            "rul_normalized": None,
        },

        # ── V2 additions ─────────────────────────────────────
        "intervention": intervention_block,
        "maintenance":  maintenance_block,
        "history":      machine_store.get_history(limit=10),

        "timestamp": (
            state.last_update.isoformat()
            if state.last_update
            else datetime.now().isoformat()
        ),
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "1", "yes", "on", "high", "detected", "vibration",
        }
    return False


def _map_anomaly_status(anomaly_score: float | None) -> str:
    if anomaly_score is None:
        return "LOW"
    if anomaly_score < 0.25:
        return "LOW"
    if anomaly_score < 0.55:
        return "ELEVATED"
    return "HIGH"


def _map_health_status(state: str) -> str:
    return {
        "NORMAL":   "GOOD",
        "WATCH":    "FAIR",
        "WARNING":  "WARNING",
        "CRITICAL": "CRITICAL",
    }.get(state, "GOOD")


def _temperature_sensor_status(temperature: float) -> str:
    return "DEGRADED" if temperature >= 70 else "OK"


def _current_sensor_status(current: float) -> str:
    return "DEGRADED" if current >= 12 else "OK"
