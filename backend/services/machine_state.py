from datetime import datetime

from backend.schemas.machine import (
    MachineHealth,
    MachineState,
)


class MachineStateStore:

    def __init__(self):

        self.state = MachineState(
            machine_id="Machine 1",
            connected=False,
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        temperature_c,
        current_a,
        vibration_detected,
        vibration_event_count,
        vibration_frequency_hz,
        health,
        relay_active=False,
    ):

        self.state.connected = True

        self.state.last_update = (
            datetime.now()
        )

        self.state.temperature_c = (
            temperature_c
        )

        self.state.current_a = (
            current_a
        )

        self.state.vibration_detected = (
            vibration_detected
        )

        self.state.vibration_event_count = (
            vibration_event_count
        )

        self.state.vibration_frequency_hz = (
            vibration_frequency_hz
        )

        self.state.health = (
            MachineHealth(
                health_score=health[
                    "health_score"
                ],

                anomaly_score=health[
                    "anomaly_score"
                ],

                state=health[
                    "state"
                ],

                severity=health[
                    "severity"
                ],

                temperature_risk=health[
                    "temperature_risk"
                ],

                vibration_risk=health[
                    "vibration_risk"
                ],

                current_risk=health[
                    "current_risk"
                ],

                online_learning_active=health[
                    "online_learning_active"
                ],

                model_samples=health[
                    "model_samples"
                ],

                learned_baseline=health[
                    "learned_baseline"
                ],

                recommended_action=health[
                    "recommended_action"
                ],
            )
        )

        self.state.relay_active = (
            relay_active
        )

        return self.state

    # =========================================================
    # GET
    # =========================================================

    def get(self):

        return self.state

    # =========================================================
    # DISCONNECT
    # =========================================================

    def set_disconnected(self):

        self.state.connected = False