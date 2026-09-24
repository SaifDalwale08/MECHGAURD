from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import (
    router,
    feature_engine,
    health_engine,
    machine_store,
)

from backend.config import (
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_SENSOR_TOPIC,
    MQTT_RELAY_TOPIC,
    MQTT_QOS,
)

from backend.services.mqtt_service import MQTTService
from backend.services.online_learning_engine import (
    OnlineLearningEngine,
)


# ============================================================
# ONLINE LEARNING ENGINE
# ============================================================

online_learning_engine = OnlineLearningEngine(
    warmup_samples=30,
    n_clusters=2,
)


# ============================================================
# MQTT SERVICE
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

    qos=MQTT_QOS,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("========================================")
    print("       MECHGUARD BACKEND STARTING")
    print("========================================")

    mqtt_service.start()

    yield

    print()
    print("========================================")
    print("       MECHGUARD BACKEND STOPPING")
    print("========================================")

    mqtt_service.stop()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MECHGUARD API",
    description=(
        "MECHGUARD machine monitoring backend "
        "with MQTT sensor ingestion, SW-420 vibration "
        "analysis, adaptive online learning, machine "
        "health scoring, and prototype relay protection."
    ),
    version="1.0.0",
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
# API ROUTES
# ============================================================

app.include_router(router)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "MECHGUARD",
        "status": "running",
        "machine": "Machine 1",
        "mqtt": "enabled",
        "online_learning": "enabled",
        "vibration_sensor": "SW-420",
    }