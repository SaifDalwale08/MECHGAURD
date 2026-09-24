from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROJECT_NAME = "MECHGUARD"
MACHINE_ID   = "MG-M1"


# ============================================================
# ESP8266 CONFIGURATION
# ============================================================

# Default ESP8266 access-point IP.
# Update this later if your actual hardware/network uses
# a different IP address.
ESP8266_IP = "192.168.4.1"

# Consider the machine disconnected if no sensor update
# is received within this time.
SENSOR_TIMEOUT_SECONDS = 10

# FeatureEngine receives one packet per configured sensor interval.  The
# SW-420 value is an event rate, not a physical vibration spectrum.
SENSOR_SAMPLE_INTERVAL_SECONDS = 1.0


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

API_HOST = "0.0.0.0"
API_PORT = 8000


# ============================================================
# MQTT CONFIGURATION
# ============================================================

# MQTT broker address.
#
# For local testing:
#   127.0.0.1
#
# If your teammate is running Mosquitto on another computer,
# replace this later with your teammate's LAN IP address.
MQTT_BROKER_HOST = "127.0.0.1"

# Standard MQTT port.
MQTT_BROKER_PORT = 1883

# Sensor data published by ESP8266
MQTT_SENSOR_TOPIC = "mechguard/machine/01/sensors"

# Relay commands published by MECHGUARD backend
MQTT_RELAY_TOPIC  = "mechguard/machine/01/relay"

# MQTT delivery quality.
# QoS 1 = message delivered at least once.
MQTT_QOS = 1


# ============================================================
# ONLINE LEARNING CONFIGURATION
# ============================================================

OL_WARMUP_SAMPLES = 30
OL_N_CLUSTERS     = 2


# ============================================================
# HEALTH ENGINE — PERSISTENCE CONFIGURATION
#
# These values control how many consecutive abnormal samples
# are required before the engine escalates its state.
#
# Increase these values to make the system less sensitive
# (more resistant to transient spikes).
# Decrease them to make it respond faster to sustained issues.
# ============================================================

# Number of consecutive samples in WARNING state before
# the engine considers the condition PERSISTENT (not transient).
HEALTH_WARNING_PERSISTENCE_SAMPLES = 3

# Number of consecutive samples in CRITICAL state before
# PROTECTIVE_SHUTDOWN is allowed.
HEALTH_CRITICAL_PERSISTENCE_SAMPLES = 5

# Number of consecutive NORMAL/WATCH samples required after
# a WARNING/CRITICAL before the engine considers the
# machine RECOVERED.
HEALTH_RECOVERY_SAMPLES = 3


# ============================================================
# INTERVENTION ENGINE CONFIGURATION
# ============================================================

# Number of samples to observe after intervention before
# deciding recovery succeeded or failed.
INTERVENTION_RECOVERY_WINDOW_SAMPLES = 5

# Consecutive improved samples required to confirm RECOVERED.
INTERVENTION_RECOVERY_SUCCESS_SAMPLES = 3

# If combined_risk exceeds this in CRITICAL state the engine
# may immediately allow shutdown (e.g. thermal runaway).
INTERVENTION_IMMEDIATE_SHUTDOWN_THRESHOLD = 0.95


# ============================================================
# MACHINE STATE HISTORY
# ============================================================

# Maximum number of historical observation records kept
# in memory.  Bounded to prevent unbounded memory growth.
MACHINE_HISTORY_MAXLEN = 200


# ============================================================
# TREND ENGINE
# ============================================================

TREND_WINDOW_SIZE = 20
