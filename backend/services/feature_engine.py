from collections import deque
from statistics import mean, pstdev


class FeatureEngine:
    """
    Feature extraction for MECHGUARD.

    Sensors:
        - DS18B20 -> temperature
        - ACS712  -> current
        - SW-420  -> digital vibration/shock detection

    SW-420 is a digital threshold sensor. Therefore, vibration
    is represented as:
        - vibration detected/not detected
        - vibration event count
        - vibration event frequency

    window_size is retained as a compatibility parameter because
    older MECHGUARD routes may still initialize FeatureEngine
    using window_size=50.
    """

    def __init__(
        self,
        window_seconds: int = 10,
        window_size: int | None = None,
        sample_interval_seconds: float = 1.0,
    ):
        # Backward compatibility with previous backend code.
        if window_size is not None:
            self.window_size = window_size
        else:
            self.window_size = window_seconds
        self.sample_interval_seconds = max(float(sample_interval_seconds), 1e-6)

        self.temperature_window = deque(
            maxlen=self.window_size
        )

        self.current_window = deque(
            maxlen=self.window_size
        )

        self.vibration_window = deque(
            maxlen=self.window_size
        )

        self.event_count_window = deque(
            maxlen=self.window_size
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        temperature_c: float,
        current_a: float,
        vibration_detected: bool,
    ) -> dict:
        """
        Add one sensor reading and calculate current features.
        """

        temperature_c = float(temperature_c)
        current_a = float(current_a)
        vibration_detected = bool(vibration_detected)

        self.temperature_window.append(
            temperature_c
        )

        self.current_window.append(
            current_a
        )

        self.vibration_window.append(
            vibration_detected
        )

        event = (
            1
            if vibration_detected
            else 0
        )

        self.event_count_window.append(
            event
        )

        # =====================================================
        # TEMPERATURE FEATURES
        # =====================================================

        temperatures = list(
            self.temperature_window
        )

        temperature_mean = (
            mean(temperatures)
            if temperatures
            else 0.0
        )

        temperature_std = (
            pstdev(temperatures)
            if len(temperatures) > 1
            else 0.0
        )

        # =====================================================
        # CURRENT FEATURES
        # =====================================================

        currents = list(
            self.current_window
        )

        current_mean = (
            mean(currents)
            if currents
            else 0.0
        )

        current_std = (
            pstdev(currents)
            if len(currents) > 1
            else 0.0
        )

        # =====================================================
        # SW-420 VIBRATION FEATURES
        # =====================================================

        vibration_events = sum(
            self.event_count_window
        )

        samples = len(
            self.event_count_window
        )

        vibration_frequency_hz = (
            vibration_events / (samples * self.sample_interval_seconds)
            if samples > 0
            else 0.0
        )

        return {
            "temperature_mean": temperature_mean,
            "temperature_std": temperature_std,

            "current_mean": current_mean,
            "current_std": current_std,

            "vibration_detected": (
                vibration_detected
            ),

            "vibration_event_count": (
                vibration_events
            ),

            "vibration_frequency_hz": (
                vibration_frequency_hz
            ),

            "samples": samples,
        }