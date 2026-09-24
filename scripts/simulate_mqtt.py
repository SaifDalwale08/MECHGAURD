import json
import random
import time

import paho.mqtt.client as mqtt


BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883

SENSOR_TOPIC = "mechguard/machine/01/sensors"


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="mechguard-simulator",
)

print(f"[SIM] Connecting to MQTT broker {BROKER_HOST}:{BROKER_PORT}")

client.connect(BROKER_HOST, BROKER_PORT, 60)

print("[SIM] Connected.")
print("[SIM] Publishing sensor data...")
print()


for i in range(60):

    # Normal machine behaviour
    temperature = 36.0 + random.uniform(-0.8, 0.8)
    current = 2.5 + random.uniform(-0.25, 0.25)

    vibration = random.random() < 0.10

    # After baseline learning, introduce abnormal behaviour
    if i >= 40:
        temperature = 60.0 + random.uniform(-2.0, 2.0)
        current = 9.0 + random.uniform(-0.5, 0.5)
        vibration = True

    payload = {
        "temperature_c": round(temperature, 2),
        "current_a": round(current, 2),
        "vibration_detected": vibration,
    }

    client.publish(
        SENSOR_TOPIC,
        json.dumps(payload),
        qos=1,
    )

    print(
        f"[{i + 1:02d}/60] "
        f"T={payload['temperature_c']}°C | "
        f"I={payload['current_a']}A | "
        f"VIB={payload['vibration_detected']}"
    )

    time.sleep(1)


print()
print("[SIM] Finished publishing 60 samples.")

client.disconnect()
print("[SIM] Disconnected.")