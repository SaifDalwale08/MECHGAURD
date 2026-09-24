import json
import threading
from datetime import datetime

import paho.mqtt.client as mqtt


class MQTTService:
    """
    MECHGUARD MQTT service.

    Data flow:

        ESP8266
            |
            | MQTT sensor JSON
            v
        MQTT Broker
            |
            v
        MECHGUARD Backend
            |
            +--> Feature Engine
            +--> Online Learning Engine
            +--> Health Engine
            +--> Machine State
            +--> Relay Decision
            |
            v
        MQTT Relay Command
            |
            v
        ESP8266 Relay


    Expected sensor payload:

    {
        "temperature_c": 36.4,
        "current_a": 2.7,
        "vibration_detected": true
    }

    Sensors:
        - DS18B20  -> temperature
        - ACS712   -> current
        - SW-420   -> digital vibration/shock detection
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        sensor_topic: str,
        relay_topic: str,
        feature_engine,
        health_engine,
        machine_store,
        online_learning_engine,
        qos: int = 1,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port

        self.sensor_topic = sensor_topic
        self.relay_topic = relay_topic

        self.feature_engine = feature_engine
        self.health_engine = health_engine
        self.machine_store = machine_store
        self.online_learning_engine = online_learning_engine

        self.qos = qos

        self.client = None
        self.thread = None
        self.running = False

    # =========================================================
    # START MQTT SERVICE
    # =========================================================

    def start(self):
        """
        Start MQTT connection in a background thread.

        The backend is allowed to continue running even if
        the MQTT broker is currently unavailable.
        """

        if self.running:
            print("[MQTT] Service already running.")
            return

        self.running = True

        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="mechguard-backend",
            )

            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
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
            print(
                f"[MQTT] Connection failed: {exc}"
            )

            print(
                "[MQTT] Backend will continue running "
                "without live MQTT data."
            )

    # =========================================================
    # MQTT NETWORK LOOP
    # =========================================================

    def _loop(self):
        try:
            self.client.loop_forever()

        except Exception as exc:
            print(
                f"[MQTT] Network loop stopped: {exc}"
            )

    # =========================================================
    # MQTT CONNECT CALLBACK
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

            print(
                "[MQTT] Connected to broker."
            )

            result, mid = client.subscribe(
                self.sensor_topic,
                qos=self.qos,
            )

            if result == mqtt.MQTT_ERR_SUCCESS:

                print(
                    f"[MQTT] Subscribed to: "
                    f"{self.sensor_topic}"
                )

            else:

                print(
                    f"[MQTT] Subscription failed: "
                    f"{result}"
                )

        else:

            print(
                "[MQTT] Broker connection failed: "
                f"{reason_code}"
            )

    # =========================================================
    # MQTT DISCONNECT CALLBACK
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
            "[MQTT] Disconnected from broker: "
            f"{reason_code}"
        )

    # =========================================================
    # MQTT MESSAGE CALLBACK
    # =========================================================

    def _on_message(
        self,
        client,
        userdata,
        msg,
    ):
        try:

            payload = msg.payload.decode(
                "utf-8"
            )

            print(
                f"[MQTT] Message received "
                f"from {msg.topic}: {payload}"
            )

            data = json.loads(payload)

            self._process_sensor_data(data)

        except json.JSONDecodeError as exc:

            print(
                f"[MQTT] Invalid JSON payload: {exc}"
            )

        except Exception as exc:

            print(
                f"[MQTT] Error processing message: "
                f"{exc}"
            )

    # =========================================================
    # SENSOR DATA PROCESSING
    # =========================================================

    def _process_sensor_data(
        self,
        data: dict,
    ):
        """
        Process one sensor packet.

        Expected:

        {
            "temperature_c": 36.4,
            "current_a": 2.7,
            "vibration_detected": true
        }
        """

        # -----------------------------------------------------
        # READ SENSOR VALUES
        # -----------------------------------------------------

        temperature_c = float(
            data.get(
                "temperature_c",
                0.0,
            )
        )

        current_a = float(
            data.get(
                "current_a",
                0.0,
            )
        )

        vibration_detected = self._parse_bool(
            data.get(
                "vibration_detected",
                False,
            )
        )

        # -----------------------------------------------------
        # FEATURE ENGINEERING
        # -----------------------------------------------------

        features = self.feature_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
        )

        # -----------------------------------------------------
        # ONLINE MACHINE LEARNING
        # -----------------------------------------------------

        learning = self.online_learning_engine.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features[
                "vibration_frequency_hz"
            ],
        )

        # -----------------------------------------------------
        # HEALTH / RISK ENGINE
        # -----------------------------------------------------

        health = self.health_engine.evaluate(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_frequency_hz=features[
                "vibration_frequency_hz"
            ],
            online_anomaly_score=learning[
                "anomaly_score"
            ],
            online_learning_active=True,
            model_samples=learning[
                "samples_seen"
            ],
        )

        # -----------------------------------------------------
        # RELAY DECISION
        # -----------------------------------------------------

        relay_active = (
            health["state"] == "CRITICAL"
        )

        # -----------------------------------------------------
        # UPDATE MACHINE STATE
        # -----------------------------------------------------

        self.machine_store.update(
            temperature_c=temperature_c,
            current_a=current_a,
            vibration_detected=vibration_detected,
            vibration_event_count=features[
                "vibration_event_count"
            ],
            vibration_frequency_hz=features[
                "vibration_frequency_hz"
            ],
            health=health,
            relay_active=relay_active,
        )

        # -----------------------------------------------------
        # LOG LIVE MACHINE STATE
        # -----------------------------------------------------

        print(
            "[MQTT] Machine state updated | "
            f"Temp={temperature_c:.2f}°C | "
            f"Current={current_a:.2f}A | "
            f"Vibration="
            f"{features['vibration_frequency_hz']:.2f}Hz | "
            f"Anomaly="
            f"{learning['anomaly_score']:.3f} | "
            f"Health="
            f"{health['health_score']:.1f} | "
            f"State="
            f"{health['state']} | "
            f"Learning="
            f"{learning['learning_status']}"
        )

        # -----------------------------------------------------
        # RELAY COMMAND
        # -----------------------------------------------------

        self._publish_relay_command(
            relay_active=relay_active,
            health_state=health["state"],
            anomaly_score=learning[
                "anomaly_score"
            ],
        )

    # =========================================================
    # RELAY COMMAND
    # =========================================================

    def _publish_relay_command(
        self,
        relay_active: bool,
        health_state: str,
        anomaly_score: float,
    ):
        """
        Publish relay command back to ESP8266.
        """

        if self.client is None:
            return

        command = {
            "machine_id": "Machine 1",

            "relay": (
                "ON"
                if relay_active
                else "OFF"
            ),

            "reason": (
                "CRITICAL machine condition"
                if relay_active
                else "Machine condition normal"
            ),

            "health_state": health_state,

            "anomaly_score": round(
                anomaly_score,
                4,
            ),

            "timestamp": datetime.now().isoformat(),
        }

        payload = json.dumps(
            command
        )

        try:

            result = self.client.publish(
                self.relay_topic,
                payload,
                qos=self.qos,
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:

                print(
                    "[MQTT] Relay command published: "
                    f"{payload}"
                )

            else:

                print(
                    "[MQTT] Relay publish failed: "
                    f"{result.rc}"
                )

        except Exception as exc:

            print(
                "[MQTT] Relay publish error: "
                f"{exc}"
            )

    # =========================================================
    # BOOLEAN PARSER
    # =========================================================

    @staticmethod
    def _parse_bool(value) -> bool:
        """
        Convert common ESP8266 representations into bool.

        Supports:

            true / false
            1 / 0
            "true" / "false"
            "1" / "0"
            "HIGH" / "LOW"
            "ON" / "OFF"
        """

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return value != 0

        if isinstance(value, str):

            normalized = value.strip().lower()

            return normalized in {
                "true",
                "1",
                "yes",
                "on",
                "high",
                "detected",
                "vibration",
            }

        return False

    # =========================================================
    # STOP MQTT SERVICE
    # =========================================================

    def stop(self):
        """
        Stop MQTT service cleanly.
        """

        self.running = False

        if self.client is not None:

            try:
                self.client.disconnect()

            except Exception:
                pass

        print(
            "[MQTT] Service stopped."
        )