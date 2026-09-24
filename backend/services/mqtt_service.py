"""
MECHGUARD V2 — MQTT Service

Data flow:

    ESP8266
        │
        │  MQTT sensor JSON
        ▼
    MQTT Broker
        │
        ▼
    MECHGUARD Backend
        │
        ├── Feature Engine       (SW-420 event features)
        ├── Online Learning      (adaptive anomaly detection)
        ├── Trend Engine         (rolling slope analysis)
        ├── Health Engine V2     (risk + persistence)
        ├── Intervention Engine  (8-state decision machine)
        ├── Maintenance Engine   (intelligence accumulation)
        └── Machine State Store  (live state + history)
        │
        ▼
    MQTT Relay Command  (only when relay_action == "OFF")
        │
        ▼
    ESP8266 Relay

Expected sensor payload:

    {
        "temperature_c":      36.4,
        "current_a":          2.7,
        "vibration_detected": true
    }

Sensors:
    DS18B20  → temperature
    ACS712   → current
    SW-420   → digital vibration / shock detection
"""

import json
import math
import threading
from datetime import datetime

import paho.mqtt.client as mqtt


class MQTTService:

    def __init__(
        self,
        broker_host:             str,
        broker_port:             int,
        sensor_topic:            str,
        relay_topic:             str,
        feature_engine,
        health_engine,
        machine_store,
        online_learning_engine,
        trend_engine,
        intervention_engine,
        maintenance_engine,
        qos:                     int = 1,
    ):
        self.broker_host   = broker_host
        self.broker_port   = broker_port
        self.sensor_topic  = sensor_topic
        self.relay_topic   = relay_topic

        self.feature_engine         = feature_engine
        self.health_engine          = health_engine
        self.machine_store          = machine_store
        self.online_learning_engine = online_learning_engine
        self.trend_engine           = trend_engine
        self.intervention_engine    = intervention_engine
        self.maintenance_engine     = maintenance_engine

        self.qos = qos

        self.client  = None
        self.thread  = None
        self.running = False

    # =========================================================
    # START
    # =========================================================

    def start(self):
        if self.running:
            print("[MQTT] Service already running.")
            return

        self.running = True

        try:
            self.client = mqtt.Client(
                callback_api_version=(
                    mqtt.CallbackAPIVersion.VERSION2
                ),
                client_id="mechguard-backend",
            )

            self.client.on_connect    = self._on_connect
            self.client.on_message    = self._on_message
            self.client.on_disconnect = self._on_disconnect

            print(
                f"[MQTT] Connecting to "
                f"{self.broker_host}:{self.broker_port}"
            )

            self.client.connect(
                self.broker_host,
                self.broker_port,
                keepalive=60,
            )

            self.thread = threading.Thread(
                target=self._loop,
                daemon=True,
            )
            self.thread.start()
            print("[MQTT] Service started.")

        except Exception as exc:
            print(f"[MQTT] Connection failed: {exc}")
            print(
                "[MQTT] Backend will continue running "
                "without live MQTT data."
            )

    # =========================================================
    # LOOP
    # =========================================================

    def _loop(self):
        try:
            self.client.loop_forever()
        except Exception as exc:
            print(f"[MQTT] Network loop stopped: {exc}")

    # =========================================================
    # CONNECT CALLBACK
    # =========================================================

    def _on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        if reason_code == 0:
            print("[MQTT] Connected to broker.")
            result, _ = client.subscribe(
                self.sensor_topic, qos=self.qos
            )
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] Subscribed to: {self.sensor_topic}")
            else:
                print(f"[MQTT] Subscription failed: {result}")
        else:
            print(
                f"[MQTT] Broker connection failed: {reason_code}"
            )

    # =========================================================
    # DISCONNECT CALLBACK
    # =========================================================

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):
        print(
            f"[MQTT] Disconnected from broker: {reason_code}"
        )

    # =========================================================
    # MESSAGE CALLBACK
    # =========================================================

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            print(
                f"[MQTT] Message received from "
                f"{msg.topic}: {payload}"
            )
            data = json.loads(payload)
            self._process_sensor_data(data)

        except json.JSONDecodeError as exc:
            print(f"[MQTT] Invalid JSON payload: {exc}")
        except Exception as exc:
            print(f"[MQTT] Error processing message: {exc}")

    # =========================================================
    # SENSOR DATA PROCESSING  (V2 pipeline)
    # =========================================================

    def _process_sensor_data(self, data: dict):
        """
        Full V2 processing pipeline for one sensor packet.

        Expected payload:
            {
                "temperature_c":      36.4,
                "current_a":          2.7,
                "vibration_detected": true
            }
        """

        # --------------------------------------------------
        # 1. READ SENSOR VALUES
        # --------------------------------------------------

        if not isinstance(data, dict):
            raise ValueError("Sensor payload must be a JSON object")
        required = {"temperature_c", "current_a", "vibration_detected"}
        if not required.issubset(data):
            raise ValueError("Sensor payload is missing required fields")
        temperature_c = float(data["temperature_c"])
        current_a     = float(data["current_a"])
        if not math.isfinite(temperature_c) or not math.isfinite(current_a):
            raise ValueError("Sensor values must be finite")
        if not -50.0 <= temperature_c <= 150.0:
            raise ValueError("temperature_c is outside the supported range")
        if not 0.0 <= current_a <= 100.0:
            raise ValueError("current_a is outside the supported range")
        if not isinstance(data["vibration_detected"], bool):
            raise ValueError("vibration_detected must be boolean")
        vibration_detected = self._parse_bool(
            data["vibration_detected"]
        )

        # --------------------------------------------------
        # 2. FEATURE ENGINE  (SW-420 event features)
        # --------------------------------------------------

        features = self.feature_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
        )
        vib_hz = features["vibration_frequency_hz"]

        # --------------------------------------------------
        # 3. ONLINE MACHINE LEARNING
        # --------------------------------------------------

        learning = self.online_learning_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=vib_hz,
        )
        anomaly_score = learning["anomaly_score"]
        model_active  = learning["model_initialized"]
        model_samples = learning["samples_seen"]

        # --------------------------------------------------
        # 4. TREND ENGINE  (placeholder health for first pass)
        # --------------------------------------------------

        trend = self.trend_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=vib_hz,
            health_score=50.0,
            anomaly_score=anomaly_score,
        )

        # --------------------------------------------------
        # 5. HEALTH ENGINE V2
        # --------------------------------------------------

        health = self.health_engine.evaluate(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=vib_hz,
            online_anomaly_score=anomaly_score,
            online_learning_active=model_active,
            model_samples=model_samples,
            trend=trend,
        )

        # --------------------------------------------------
        # 6. UPDATE TREND with real health score
        # --------------------------------------------------

        trend = self.trend_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=vib_hz,
            health_score=health["health_score"],
            anomaly_score=health["anomaly_score"],
        )

        # --------------------------------------------------
        # 7. INTERVENTION ENGINE
        # --------------------------------------------------

        intervention = self.intervention_engine.evaluate(
            health_dict=health,
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=vib_hz,
            trend=trend,
        )

        relay_active = (
            intervention["intervention_state"] == "PROTECTIVE_SHUTDOWN"
            and intervention["relay_action"] == "OFF"
        )

        # --------------------------------------------------
        # 8. MAINTENANCE INTELLIGENCE
        # --------------------------------------------------

        self.maintenance_engine.record(
            health_dict=health,
            intervention_dict=intervention,
        )

        # --------------------------------------------------
        # 9. MACHINE STATE
        # --------------------------------------------------

        self.machine_store.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
            vibration_event_count=features["vibration_event_count"],
            vibration_frequency_hz=vib_hz,
            health=health,
            relay_active=relay_active,
            intervention=intervention,
            trend=trend,
            learning=learning,
        )

        # --------------------------------------------------
        # 10. LOG
        # --------------------------------------------------

        print(
            "[MQTT] "
            f"Temp={temperature_c:.1f}°C | "
            f"Current={current_a:.2f}A | "
            f"Vib={vib_hz:.2f}Hz | "
            f"Anomaly={anomaly_score:.3f} | "
            f"Health={health['health_score']:.1f} | "
            f"State={health['state']} | "
            f"Persist={health['persistence_state']} | "
            f"IV={intervention['intervention_state']} | "
            f"Action={intervention['action']} | "
            f"Pattern={learning['operating_pattern']} | "
            f"Relay={intervention['relay_action']}"
        )

        # --------------------------------------------------
        # 11. RELAY COMMAND
        # --------------------------------------------------

        self._publish_relay_command(
            relay_active=relay_active,
            health_state=health["state"],
            anomaly_score=anomaly_score,
            intervention_state=intervention["intervention_state"],
            reason=intervention["reason"],
        )

    # =========================================================
    # RELAY COMMAND PUBLISHER
    # =========================================================

    def _publish_relay_command(
        self,
        relay_active:       bool,
        health_state:       str,
        anomaly_score:      float,
        intervention_state: str,
        reason:             str,
    ):
        """
        Publish relay command to ESP8266.

        relay: "OFF" only when PROTECTIVE_SHUTDOWN is reached.
        In all other states relay remains "ON" (machine running).
        """

        if self.client is None:
            return

        if intervention_state != "PROTECTIVE_SHUTDOWN":
            relay_active = False

        command = {
            "machine_id":         "MG-M1",
            "relay":              "OFF" if relay_active else "ON",
            "reason":             reason or (
                "CRITICAL machine condition — protective shutdown"
                if relay_active
                else "Machine condition normal or under monitoring"
            ),
            "health_state":       health_state,
            "anomaly_score":      round(anomaly_score, 4),
            "intervention_state": intervention_state,
            "timestamp":          datetime.now().isoformat(),
        }

        try:
            result = self.client.publish(
                self.relay_topic,
                json.dumps(command),
                qos=self.qos,
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(
                    f"[MQTT] Relay command published: "
                    f"{command['relay']} "
                    f"(IV={intervention_state})"
                )
            else:
                print(f"[MQTT] Relay publish failed: {result.rc}")

        except Exception as exc:
            print(f"[MQTT] Relay publish error: {exc}")

    # =========================================================
    # BOOL PARSER
    # =========================================================

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {
                "true", "1", "yes", "on",
                "high", "detected", "vibration",
            }
        return False

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):
        self.running = False
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
        print("[MQTT] Service stopped.")
