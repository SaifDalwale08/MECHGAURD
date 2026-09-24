from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROJECT_NAME = "MECHGUARD"
MACHINE_ID = "Machine 1"


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
MQTT_RELAY_TOPIC = "mechguard/machine/01/relay"


# MQTT delivery quality.
# QoS 1 = message delivered at least once.
MQTT_QOS = 1