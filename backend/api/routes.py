from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.machine import MachineState
from backend.services.feature_engine import FeatureEngine
from backend.services.health_engine import HealthEngine
from backend.services.machine_state import MachineStateStore


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()


# ============================================================
# SHARED BACKEND COMPONENTS
# ============================================================

feature_engine = FeatureEngine(
    window_size=50
)

health_engine = HealthEngine()

machine_store = MachineStateStore()


# ============================================================
# BASIC HEALTH
# ============================================================

@router.get("/api/health")
def api_health():
    return {
        "status": "online",
        "service": "MECHGUARD API",
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
            "message": "No sensor data received yet.",
        }

    return state.health


# ============================================================
# MACHINE SENSOR DATA
# ============================================================

@router.get("/api/machine/sensors")
def get_machine_sensors():

    state = machine_store.get()

    return {
        "temperature_c": state.temperature_c,
        "current_a": state.current_a,

        "vibration_detected": (
            state.vibration_detected
        ),

        "vibration_event_count": (
            state.vibration_event_count
        ),

        "vibration_frequency_hz": (
            state.vibration_frequency_hz
        ),

        "timestamp": (
            state.last_update.isoformat()
            if state.last_update
            else None
        ),
    }


# ============================================================
# MANUAL / HTTP SENSOR INGESTION
# ============================================================

@router.post("/api/sensor-data")
def receive_sensor_data(payload: dict[str, Any]):

    try:

        temperature_c = float(
            payload.get(
                "temperature_c",
                0.0,
            )
        )

        current_a = float(
            payload.get(
                "current_a",
                0.0,
            )
        )

        vibration_detected = _parse_bool(
            payload.get(
                "vibration_detected",
                False,
            )
        )

        # ----------------------------------------------------
        # FEATURE ENGINE
        # ----------------------------------------------------

        features = feature_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
        )

        # ----------------------------------------------------
        # HEALTH ENGINE
        #
        # HTTP ingestion does not directly own the online
        # learning engine. MQTT ingestion is the primary
        # adaptive-learning path.
        # ----------------------------------------------------

        health_dict = health_engine.evaluate(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features[
                "vibration_frequency_hz"
            ],
            online_anomaly_score=0.0,
            online_learning_active=False,
            model_samples=0,
        )

        # ----------------------------------------------------
        # RELAY DECISION
        # ----------------------------------------------------

        relay_active = (
            health_dict["state"] == "CRITICAL"
        )

        # ----------------------------------------------------
        # MACHINE STATE
        # ----------------------------------------------------

        state = machine_store.update(
            temperature_c=temperature_c,
            current_a=current_a,

            vibration_detected=(
                vibration_detected
            ),

            vibration_event_count=(
                features[
                    "vibration_event_count"
                ]
            ),

            vibration_frequency_hz=(
                features[
                    "vibration_frequency_hz"
                ]
            ),

            health=health_dict,

            relay_active=relay_active,
        )

        return {
            "success": True,
            "machine": state,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# FRONTEND LIVE ENDPOINT
# ============================================================

@router.get(
    "/api/machines/{machine_id}/live"
)
def get_live_snapshot(
    machine_id: str,
):

    # --------------------------------------------------------
    # MACHINE ID
    # --------------------------------------------------------

    if machine_id != "MG-M1":
        raise HTTPException(
            status_code=404,
            detail="Machine not found",
        )

    state = machine_store.get()

    connected = state.connected

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    health = state.health

    if health is None:

        health_score = None
        health_status = "GOOD"

        anomaly_score = None
        condition = "NORMAL"
        anomaly_status = "LOW"

        recommended_action = (
            "Waiting for sensor data."
        )

        temperature_risk = 0.0
        vibration_risk = 0.0
        current_risk = 0.0

    else:

        health_score = health.health_score

        health_status = (
            _map_health_status(
                health.state
            )
        )

        anomaly_score = health.anomaly_score

        condition = (
            _map_condition(
                health.state
            )
        )

        anomaly_status = (
            _map_anomaly_status(
                health.anomaly_score
            )
        )

        recommended_action = (
            health.recommended_action
        )

        temperature_risk = (
            health.temperature_risk
        )

        vibration_risk = (
            health.vibration_risk
        )

        current_risk = (
            health.current_risk
        )

    # --------------------------------------------------------
    # SENSOR STATUS
    # --------------------------------------------------------

    if connected:

        temperature_status = (
            _temperature_sensor_status(
                state.temperature_c
            )
        )

        current_status = (
            _current_sensor_status(
                state.current_a
            )
        )

        vibration_status = (
            "OK"
            if state.vibration_detected
            else "OK"
        )

        sw420_status = "OK"
        acs712_status = "OK"
        ds18b20_status = "OK"

    else:

        temperature_status = "NO_DATA"
        current_status = "NO_DATA"
        vibration_status = "NO_DATA"

        sw420_status = "NO_DATA"
        acs712_status = "NO_DATA"
        ds18b20_status = "NO_DATA"

    # --------------------------------------------------------
    # VIBRATION STATE
    # --------------------------------------------------------

    vibration_state = (
        "VIBRATION"
        if state.vibration_detected
        else "NO_VIBRATION"
    )

    # --------------------------------------------------------
    # RELAY STATE
    # --------------------------------------------------------

    if not connected:

        relay_state = "OFFLINE"

    elif state.relay_active:

        relay_state = "TRIGGERED"

    else:

        relay_state = "ARMED"

    # --------------------------------------------------------
    # ALERTS
    # --------------------------------------------------------

    alerts = []

    if health is not None:

        if health.state == "CRITICAL":

            alerts.append(
                {
                    "id": "critical-machine-condition",
                    "severity": "CRITICAL",
                    "title": "Critical machine condition",
                    "message": recommended_action,
                }
            )

        elif health.state == "WARNING":

            alerts.append(
                {
                    "id": "machine-warning",
                    "severity": "WARNING",
                    "title": "Machine warning",
                    "message": recommended_action,
                }
            )

        elif health.state == "WATCH":

            alerts.append(
                {
                    "id": "machine-watch",
                    "severity": "INFO",
                    "title": "Machine requires monitoring",
                    "message": recommended_action,
                }
            )

    # --------------------------------------------------------
    # FRONTEND RESPONSE
    # --------------------------------------------------------

    response = {
        "machine": {
            "id": "MG-M1",
            "name": "Machine 1",

            "status": (
                "RUNNING"
                if connected
                else "STOPPED"
            ),

            "connection": (
                "CONNECTED"
                if connected
                else "DISCONNECTED"
            ),
        },

        "sensors": {

            "temperature": {
                "value": (
                    state.temperature_c
                    if connected
                    else None
                ),
                "status": temperature_status,
            },

            "current": {
                "value": (
                    state.current_a
                    if connected
                    else None
                ),
                "status": current_status,
            },

            "vibration": {
                "state": vibration_state,

                "event_count": (
                    state.vibration_event_count
                ),

                "frequency_hz": (
                    state.vibration_frequency_hz
                ),

                "status": vibration_status,
            },
        },

        "health": {
            "score": health_score,
            "status": health_status,
        },

        "analytics": {
            "condition": condition,
            "anomaly_score": anomaly_score,
            "anomaly_status": anomaly_status,
        },

        "protection": {
            "relay": relay_state,
            "mode": "AUTO",

            "reason": (
                "Critical condition detected"
                if state.relay_active
                else None
            ),

            "triggeredBy": {
                "temperature": (
                    temperature_risk >= 0.80
                ),
                "vibration": (
                    vibration_risk >= 0.80
                ),
                "current": (
                    current_risk >= 0.80
                ),
            },

            "lastChangedAt": (
                state.last_update.isoformat()
                if state.relay_active
                and state.last_update
                else None
            ),
        },

        # No persistent relay-event database yet.
        "relayEvents": [],

        "alerts": alerts,

        "system": {

            "esp8266": (
                "CONNECTED"
                if connected
                else "DISCONNECTED"
            ),

            "backend": "ONLINE",

            "api": "CONNECTED",

            "relay": relay_state,

            "sensors": {
                "sw420": sw420_status,
                "acs712": acs712_status,
                "ds18b20": ds18b20_status,
            },
        },

        # Live RUL remains unavailable because the current
        # hardware does not produce the validated 120-feature
        # Piecuch model input.
        "prediction": {
            "available": False,
            "model": None,
            "rul_normalized": None,
        },

        "timestamp": (
            state.last_update.isoformat()
            if state.last_update
            else datetime.now().isoformat()
        ),
    }

    return response


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
            "true",
            "1",
            "yes",
            "on",
            "high",
            "detected",
            "vibration",
        }

    return False


def _map_health_status(
    state: str,
) -> str:

    mapping = {
        "NORMAL": "GOOD",
        "WATCH": "FAIR",
        "WARNING": "WARNING",
        "CRITICAL": "CRITICAL",
    }

    return mapping.get(
        state,
        "GOOD",
    )


def _map_condition(
    state: str,
) -> str:

    mapping = {
        "NORMAL": "NORMAL",
        "WATCH": "ELEVATED",
        "WARNING": "WARNING",
        "CRITICAL": "CRITICAL",
    }

    return mapping.get(
        state,
        "NORMAL",
    )


def _map_anomaly_status(
    anomaly_score: float | None,
) -> str:

    if anomaly_score is None:
        return "LOW"

    if anomaly_score < 0.25:
        return "LOW"

    if anomaly_score < 0.55:
        return "ELEVATED"

    return "HIGH"


def _temperature_sensor_status(
    temperature: float,
) -> str:

    if temperature >= 70:
        return "DEGRADED"

    return "OK"


def _current_sensor_status(
    current: float,
) -> str:

    if current >= 12:
        return "DEGRADED"

    return "OK"