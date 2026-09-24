"""
MECHGUARD V2 — Trend Engine

Maintains a bounded rolling history of sensor and health metrics
and calculates simple, explainable linear trends from ordered samples.

The trend engine does NOT use deep learning.
It uses a least-squares slope calculation over a fixed window.
"""

from collections import deque
from typing import Optional


# ============================================================
# DIRECTION THRESHOLDS
# ============================================================

# Minimum slope magnitude to be considered RISING or FALLING.
# Below this the signal is STABLE.

_TEMP_SLOPE_THRESHOLD     = 0.10   # °C per sample
_CURRENT_SLOPE_THRESHOLD  = 0.05   # A per sample
_VIBRATION_SLOPE_THRESHOLD = 0.01  # Hz per sample
_HEALTH_SLOPE_THRESHOLD   = 0.20   # score points per sample
_ANOMALY_SLOPE_THRESHOLD  = 0.005  # normalised score per sample


# ============================================================
# TREND ENGINE
# ============================================================

class TrendEngine:
    """
    Calculates rolling trend slopes for key machine metrics.

    For each metric a deque of the last `window` values is kept.
    The slope is estimated by fitting y = mx + b to the indices
    [0, 1, 2, …, n-1] using the closed-form least-squares formula,
    which is O(n) and produces a single explainable number.

    Positive slope  → metric is rising.
    Negative slope  → metric is falling.
    Near-zero slope → metric is stable.
    """

    def __init__(self, window: int = 20):
        self.window = window

        self._temperature:  deque[float] = deque(maxlen=window)
        self._current:      deque[float] = deque(maxlen=window)
        self._vibration:    deque[float] = deque(maxlen=window)
        self._health_score: deque[float] = deque(maxlen=window)
        self._anomaly:      deque[float] = deque(maxlen=window)

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        temperature_c: float,
        current_a: float,
        vibration_frequency_hz: float,
        health_score: float,
        anomaly_score: float,
    ) -> dict:
        """
        Add one observation and return the current trend dict.
        """

        self._temperature.append(temperature_c)
        self._current.append(current_a)
        self._vibration.append(vibration_frequency_hz)
        self._health_score.append(health_score)
        self._anomaly.append(anomaly_score)

        t_slope  = self._slope(self._temperature)
        c_slope  = self._slope(self._current)
        v_slope  = self._slope(self._vibration)
        h_slope  = self._slope(self._health_score)
        a_slope  = self._slope(self._anomaly)

        return {
            "temperature_trend":   round(t_slope, 4),
            "current_trend":       round(c_slope, 4),
            "vibration_trend":     round(v_slope, 4),
            "health_trend":        round(h_slope, 4),
            "anomaly_trend":       round(a_slope, 4),

            "temperature_direction":  _direction(
                t_slope, _TEMP_SLOPE_THRESHOLD
            ),
            "current_direction":      _direction(
                c_slope, _CURRENT_SLOPE_THRESHOLD
            ),
            "vibration_direction":    _direction(
                v_slope, _VIBRATION_SLOPE_THRESHOLD
            ),
            "health_direction":       _direction(
                h_slope, _HEALTH_SLOPE_THRESHOLD
            ),
            "anomaly_direction":      _direction(
                a_slope, _ANOMALY_SLOPE_THRESHOLD
            ),

            "samples": len(self._temperature),
        }

    # =========================================================
    # SLOPE CALCULATION
    # =========================================================

    @staticmethod
    def _slope(data: deque) -> float:
        """
        Least-squares slope of the sequence.

        Given n points (0, y0), (1, y1), …, (n-1, y_{n-1}):

            m = ( n * Σ(i*yi) − Σi * Σyi ) / ( n * Σ(i²) − (Σi)² )

        Returns 0.0 if there are fewer than 2 points.
        """
        n = len(data)

        if n < 2:
            return 0.0

        sum_x   = 0.0
        sum_y   = 0.0
        sum_xy  = 0.0
        sum_x2  = 0.0

        for i, y in enumerate(data):
            sum_x  += i
            sum_y  += y
            sum_xy += i * y
            sum_x2 += i * i

        denom = n * sum_x2 - sum_x * sum_x

        if denom == 0.0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denom

    # =========================================================
    # SUMMARY DESCRIPTION
    # =========================================================

    def summary(self, trend: Optional[dict] = None) -> str:
        """
        Return a one-sentence human-readable summary of the
        dominant trend direction.

        Pass the dict returned by update(), or call with no
        argument to recompute from current buffers.
        """

        if trend is None:
            trend = self.update(
                temperature_c=self._temperature[-1] if self._temperature else 0.0,
                current_a=self._current[-1]         if self._current     else 0.0,
                vibration_frequency_hz=self._vibration[-1] if self._vibration else 0.0,
                health_score=self._health_score[-1] if self._health_score else 0.0,
                anomaly_score=self._anomaly[-1]     if self._anomaly     else 0.0,
            )

        parts: list[str] = []

        if trend["temperature_direction"] == "RISING":
            parts.append("temperature rising")
        elif trend["temperature_direction"] == "FALLING":
            parts.append("temperature falling")

        if trend["current_direction"] == "RISING":
            parts.append("current rising")
        elif trend["current_direction"] == "FALLING":
            parts.append("current falling")

        if trend["vibration_direction"] == "RISING":
            parts.append("vibration activity increasing")
        elif trend["vibration_direction"] == "FALLING":
            parts.append("vibration activity decreasing")

        if trend["health_direction"] == "FALLING":
            parts.append("health deteriorating")
        elif trend["health_direction"] == "RISING":
            parts.append("health improving")

        if trend["anomaly_direction"] == "RISING":
            parts.append("anomaly score increasing")

        if not parts:
            return "All monitored metrics are stable."

        return "Trends detected: " + ", ".join(parts) + "."

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:
        """Clear all history buffers."""
        self._temperature.clear()
        self._current.clear()
        self._vibration.clear()
        self._health_score.clear()
        self._anomaly.clear()


# ============================================================
# HELPERS
# ============================================================

def _direction(slope: float, threshold: float) -> str:
    if slope >  threshold:
        return "RISING"
    if slope < -threshold:
        return "FALLING"
    return "STABLE"
