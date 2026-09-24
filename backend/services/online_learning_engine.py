"""
MECHGUARD V2 — Online Learning Engine

Adaptive unsupervised anomaly detection using:
    StandardScaler  (incremental normalisation)
    MiniBatchKMeans (incremental clustering)
    partial_fit     (online updates)

The model learns the machine's operating behaviour from incoming
sensor features.  It does NOT learn named failure classes and does
NOT claim exact failure-time prediction.

Features used:
    1. temperature_c
    2. current_a
    3. vibration_frequency_hz  (SW-420 event rate)

Operating patterns returned:
    LEARNING_BASELINE   — warmup period, model not yet initialised
    NORMAL_OPERATION    — anomaly score below watch threshold
    ELEVATED_OPERATION  — anomaly score in watch band
    ABNORMAL_OPERATION  — anomaly score above warning threshold
"""

from collections import deque

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# OPERATING PATTERN THRESHOLDS
# ============================================================

_PATTERN_WATCH_THRESHOLD   = 0.25
_PATTERN_ABNORMAL_THRESHOLD = 0.50


# ============================================================
# ONLINE LEARNING ENGINE
# ============================================================

class OnlineLearningEngine:
    """
    Incremental unsupervised anomaly detection engine.

    Call update() once per sensor observation.
    The engine handles its own warmup buffer; callers do not
    need to manage the initialisation lifecycle.
    """

    def __init__(
        self,
        warmup_samples: int = 30,
        n_clusters: int = 2,
        learning_rate: float = 0.05,
    ):
        self.warmup_samples  = warmup_samples
        self.n_clusters      = n_clusters
        self.learning_rate   = learning_rate

        self.scaler = StandardScaler()
        self.model  = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=32,
            n_init=3,
        )

        # Warmup buffer (filled before first model init)
        self.buffer: deque[np.ndarray] = deque(
            maxlen=warmup_samples
        )

        self.samples_seen       = 0
        self.model_initialized  = False

        # Keep the warmup distribution as the stable reference for scoring.
        self._baseline_distance_limit = 1.0

        # Rolling distance history for context
        self.distance_history: deque[float] = deque(maxlen=100)

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        temperature_c: float,
        current_a: float,
        vibration_frequency_hz: float,
    ) -> dict:
        """
        Process one sensor observation.

        Returns a dict with:
            anomaly_score       float  [0, 1]
            model_initialized   bool
            samples_seen        int
            learning_status     str
            baseline_state      str
            operating_pattern   str
            distance            float  (0.0 during warmup)
        """

        vector = np.array(
            [[temperature_c, current_a, vibration_frequency_hz]],
            dtype=float,
        )
        vector = np.nan_to_num(
            vector, nan=0.0, posinf=1e6, neginf=-1e6
        )

        self.samples_seen += 1

        # -------------------------------------------------
        # WARMUP — accumulate baseline samples
        # -------------------------------------------------

        if not self.model_initialized:

            self.buffer.append(vector[0])

            if len(self.buffer) < self.warmup_samples:
                return {
                    "anomaly_score":     0.0,
                    "model_initialized": False,
                    "samples_seen":      self.samples_seen,
                    "learning_status":   "LEARNING_BASELINE",
                    "baseline_state":    "COLLECTING",
                    "operating_pattern": "LEARNING_BASELINE",
                    "distance":          0.0,
                }

            # Enough warmup samples — fit the initial model
            initial_data = np.array(list(self.buffer))
            self.scaler.fit(initial_data)
            scaled_data = self.scaler.transform(initial_data)
            self.model.partial_fit(scaled_data)
            baseline_distances = np.min(
                self.model.transform(scaled_data), axis=1
            )
            self._baseline_distance_limit = max(
                float(np.percentile(baseline_distances, 95.0)), 1.0
            )
            self.model_initialized = True

            return {
                "anomaly_score":     0.0,
                "model_initialized": True,
                "samples_seen":      self.samples_seen,
                "learning_status":   "ADAPTIVE",
                "baseline_state":    "ESTABLISHED",
                "operating_pattern": "NORMAL_OPERATION",
                "distance":          0.0,
            }

        # -------------------------------------------------
        # ADAPTIVE — score then update incrementally
        # -------------------------------------------------

        # Score in the fixed warmup reference frame before adapting KMeans.
        scaled_vector = self.scaler.transform(vector)

        # Score BEFORE this sample is incorporated into the model
        distances     = self.model.transform(scaled_vector)
        min_distance  = float(np.min(distances))

        excess = max(0.0, min_distance - self._baseline_distance_limit)
        anomaly_score = excess / (
            self._baseline_distance_limit * 2.0 + 1.0 + excess
        )
        anomaly_score = max(0.0, min(1.0, float(anomaly_score)))

        self.distance_history.append(min_distance)

        # Incorporate new observation into the cluster model
        if anomaly_score < _PATTERN_ABNORMAL_THRESHOLD:
            self.model.partial_fit(scaled_vector)

        operating_pattern = _classify_pattern(anomaly_score)

        return {
            "anomaly_score":     round(anomaly_score, 4),
            "model_initialized": True,
            "samples_seen":      self.samples_seen,
            "learning_status":   "ADAPTIVE",
            "baseline_state":    "ESTABLISHED",
            "operating_pattern": operating_pattern,
            "distance":          round(min_distance, 4),
        }

    # =========================================================
    # STATUS
    # =========================================================

    def status(self) -> dict:
        return {
            "active":            True,
            "model_initialized": self.model_initialized,
            "samples_seen":      self.samples_seen,
            "learning_status": (
                "ADAPTIVE"
                if self.model_initialized
                else "LEARNING_BASELINE"
            ),
            "baseline_state": (
                "ESTABLISHED"
                if self.model_initialized
                else "COLLECTING"
            ),
        }


# ============================================================
# HELPERS
# ============================================================

def _classify_pattern(anomaly_score: float) -> str:
    if anomaly_score >= _PATTERN_ABNORMAL_THRESHOLD:
        return "ABNORMAL_OPERATION"
    if anomaly_score >= _PATTERN_WATCH_THRESHOLD:
        return "ELEVATED_OPERATION"
    return "NORMAL_OPERATION"
