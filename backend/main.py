"""
MECHGUARD V2 — Application Entry Point

Instantiates all V2 services, wires them together, and starts
the FastAPI application with MQTT ingestion.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import (
    router,
    feature_engine,
    health_engine,
    machine_store,
    intervention_engine,
    maintenance_engine,
)

from backend.config import (
    # MQTT
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_SENSOR_TOPIC,
    MQTT_RELAY_TOPIC,
    MQTT_QOS,
    # Online learning
    OL_WARMUP_SAMPLES,
    OL_N_CLUSTERS,
    # Trend
    TREND_WINDOW_SIZE,
)

from backend.services.mqtt_service import MQTTService
from backend.services.online_learning_engine import OnlineLearningEngine
from backend.services.trend_engine import TrendEngine


# ============================================================
# ONLINE LEARNING ENGINE
# ============================================================

online_learning_engine = OnlineLearningEngine(
    warmup_samples=OL_WARMUP_SAMPLES,
    n_clusters=OL_N_CLUSTERS,
)


# ============================================================
# MQTT TREND ENGINE
# (Separate instance from the HTTP-path one in routes.py
#  so MQTT and HTTP sensor flows do not share state.)
# ============================================================

mqtt_trend_engine = TrendEngine(window=TREND_WINDOW_SIZE)


# ============================================================
# MQTT SERVICE  (V2 — all engines wired)
# ============================================================

mqtt_service = MQTTService(
    broker_host=MQTT_BROKER_HOST,
    broker_port=MQTT_BROKER_PORT,
    sensor_topic=MQTT_SENSOR_TOPIC,
    relay_topic=MQTT_RELAY_TOPIC,

    feature_engine=feature_engine,
    health_engine=health_engine,
    machine_store=machine_store,
    online_learning_engine=online_learning_engine,
    trend_engine=mqtt_trend_engine,   # MQTT gets its own trend instance
    intervention_engine=intervention_engine,
    maintenance_engine=maintenance_engine,

    qos=MQTT_QOS,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("=" * 48)
    print("       MECHGUARD V2 BACKEND STARTING")
    print("=" * 48)
    print("  AI pipeline:")
    print("    FeatureEngine    → SW-420 vibration features")
    print("    OnlineLearning   → adaptive anomaly detection")
    print("    TrendEngine      → rolling slope analysis")
    print("    HealthEngine V2  → persistence-aware scoring")
    print("    InterventionEng  → 8-state decision machine")
    print("    MaintenanceEng   → intelligence accumulation")
    print("=" * 48)

    mqtt_service.start()

    yield

    print()
    print("=" * 48)
    print("       MECHGUARD V2 BACKEND STOPPING")
    print("=" * 48)

    mqtt_service.stop()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MECHGUARD V2 API",
    description=(
        "MECHGUARD V2 machine monitoring backend. "
        "Adaptive unsupervised anomaly detection, "
        "machine-health assessment, trend analysis, "
        "intervention engine with recovery verification, "
        "and maintenance intelligence. "
        "Sensors: SW-420 vibration, ACS712 current, "
        "DS18B20 temperature. "
        "Protective shutdown only after persistent critical "
        "condition and failed recovery."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(router)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "project":          "MECHGUARD",
        "version":          "2.0.0",
        "status":           "running",
        "machine":          "MG-M1",
        "mqtt":             "enabled",
        "online_learning":  "enabled",
        "trend_analysis":   "enabled",
        "intervention":     "enabled",
        "maintenance_intel": "enabled",
        "vibration_sensor": "SW-420",
        "ai_description": (
            "Adaptive unsupervised anomaly detection and "
            "machine-health assessment. "
            "Does not claim exact failure-time prediction."
        ),
    }
