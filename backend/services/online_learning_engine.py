import math
from collections import deque

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class OnlineLearningEngine:
    """
    Online unsupervised machine-learning engine.

    The model continuously learns the machine's operating
    behaviour from incoming sensor features.

    Features:
        1. Temperature
        2. Current
        3. SW-420 vibration frequency

    The model uses MiniBatchKMeans with incremental
    partial_fit updates.

    It does NOT claim to predict exact failure time.
    It learns operating behaviour and produces an
    adaptive anomaly score.
    """

    def __init__(
        self,
        warmup_samples: int = 30,
        n_clusters: int = 2,
        learning_rate: float = 0.05,
    ):
        self.warmup_samples = warmup_samples
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate

        self.scaler = StandardScaler()

        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=32,
            n_init=3,
        )

        self.buffer = deque(
            maxlen=warmup_samples
        )

        self.samples_seen = 0
        self.model_initialized = False

        self.distance_history = deque(
            maxlen=100
        )

    # =====================================================
    # UPDATE MODEL
    # =====================================================

    def update(
        self,
        temperature_c: float,
        current_a: float,
        vibration_frequency_hz: float,
    ) -> dict:

        vector = np.array(
            [[
                temperature_c,
                current_a,
                vibration_frequency_hz,
            ]],
            dtype=float,
        )

        self.samples_seen += 1

        # -------------------------------------------------
        # WARMUP
        # -------------------------------------------------

        if not self.model_initialized:

            self.buffer.append(vector[0])

            if len(self.buffer) < self.warmup_samples:
                return {
                    "anomaly_score": 0.0,
                    "model_initialized": False,
                    "samples_seen": self.samples_seen,
                    "learning_status": "LEARNING_BASELINE",
                }

            initial_data = np.array(
                list(self.buffer)
            )

            # Learn scaling parameters
            self.scaler.partial_fit(
                initial_data
            )

            scaled_data = self.scaler.transform(
                initial_data
            )

            self.model.partial_fit(
                scaled_data
            )

            self.model_initialized = True

            return {
                "anomaly_score": 0.0,
                "model_initialized": True,
                "samples_seen": self.samples_seen,
                "learning_status": "ADAPTIVE",
            }

        # -------------------------------------------------
        # SCALE NEW SAMPLE
        # -------------------------------------------------

        self.scaler.partial_fit(vector)

        scaled_vector = self.scaler.transform(
            vector
        )

        # -------------------------------------------------
        # CALCULATE ANOMALY BEFORE MODEL UPDATE
        # -------------------------------------------------

        distances = self.model.transform(
            scaled_vector
        )

        min_distance = float(
            np.min(distances)
        )

        # Convert distance to 0–1 anomaly score.
        anomaly_score = 1.0 - math.exp(
            -min_distance
        )

        anomaly_score = max(
            0.0,
            min(
                1.0,
                anomaly_score
            )
        )

        self.distance_history.append(
            min_distance
        )

        # -------------------------------------------------
        # CONTINUOUS ONLINE LEARNING
        # -------------------------------------------------

        self.model.partial_fit(
            scaled_vector
        )

        return {
            "anomaly_score": anomaly_score,
            "model_initialized": True,
            "samples_seen": self.samples_seen,
            "learning_status": "ADAPTIVE",
            "distance": min_distance,
        }

    # =====================================================
    # STATUS
    # =====================================================

    def status(self) -> dict:

        return {
            "active": True,
            "model_initialized": self.model_initialized,
            "samples_seen": self.samples_seen,
            "learning_status": (
                "ADAPTIVE"
                if self.model_initialized
                else "LEARNING_BASELINE"
            ),
        }