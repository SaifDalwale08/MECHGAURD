class HealthEngine:

    TEMP_NORMAL = 40.0
    TEMP_WARNING = 55.0
    TEMP_CRITICAL = 70.0

    CURRENT_NORMAL = 5.0
    CURRENT_WARNING = 8.0
    CURRENT_CRITICAL = 12.0

    VIBRATION_NORMAL = 0.20
    VIBRATION_WARNING = 0.50
    VIBRATION_CRITICAL = 0.80

    def evaluate(
        self,
        temperature_c: float,
        current_a: float,
        vibration_frequency_hz: float,
        online_anomaly_score: float = 0.0,
        online_learning_active: bool = False,
        model_samples: int = 0,
    ):

        temperature_risk = self._risk(
            temperature_c,
            self.TEMP_NORMAL,
            self.TEMP_WARNING,
            self.TEMP_CRITICAL,
        )

        current_risk = self._risk(
            current_a,
            self.CURRENT_NORMAL,
            self.CURRENT_WARNING,
            self.CURRENT_CRITICAL,
        )

        vibration_risk = self._risk(
            vibration_frequency_hz,
            self.VIBRATION_NORMAL,
            self.VIBRATION_WARNING,
            self.VIBRATION_CRITICAL,
        )

        # -------------------------------------------------
        # SENSOR RISK
        # -------------------------------------------------

        sensor_risk = (
            temperature_risk * 0.25
            + vibration_risk * 0.35
            + current_risk * 0.20
        )

        # -------------------------------------------------
        # ONLINE ML
        # -------------------------------------------------

        combined_risk = (
            sensor_risk * 0.80
            + online_anomaly_score * 0.20
        )

        combined_risk = max(
            0.0,
            min(
                1.0,
                combined_risk
            )
        )

        health_score = (
            100.0 * (1.0 - combined_risk)
        )

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        if combined_risk < 0.25:
            state = "NORMAL"
            severity = "LOW"

        elif combined_risk < 0.55:
            state = "WATCH"
            severity = "MEDIUM"

        elif combined_risk < 0.80:
            state = "WARNING"
            severity = "HIGH"

        else:
            state = "CRITICAL"
            severity = "CRITICAL"

        # -------------------------------------------------
        # RECOMMENDED ACTION
        # -------------------------------------------------

        if state == "NORMAL":
            action = "Continue normal operation."

        elif state == "WATCH":
            action = (
                "Continue monitoring machine behaviour."
            )

        elif state == "WARNING":
            action = (
                "Inspect machine and review abnormal "
                "sensor behaviour."
            )

        else:
            action = (
                "Stop or isolate machine and inspect "
                "critical condition."
            )

        return {
            "health_score": health_score,
            "anomaly_score": combined_risk,
            "state": state,
            "severity": severity,

            "temperature_risk": temperature_risk,
            "vibration_risk": vibration_risk,
            "current_risk": current_risk,

            "online_learning_active":
                online_learning_active,

            "model_samples": model_samples,

            "learned_baseline":
                model_samples >= 30,

            "recommended_action": action,
        }

    @staticmethod
    def _risk(
        value: float,
        normal: float,
        warning: float,
        critical: float,
    ) -> float:

        if value <= normal:
            return 0.0

        if value >= critical:
            return 1.0

        if value <= warning:
            return (
                0.35
                * (value - normal)
                / (warning - normal)
            )

        return (
            0.35
            + 0.65
            * (value - warning)
            / (critical - warning)
        )