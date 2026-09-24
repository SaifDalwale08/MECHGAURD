"""
MECHGUARD V2 — Pydantic schemas

All schema changes are additive and backward-compatible.
Existing fields used by the frontend live endpoint are preserved.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# SENSOR READING
# ============================================================

class SensorReading(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)

    temperature_c: float
    current_a: float

    # SW-420 digital vibration sensor
    vibration_detected:    bool  = False
    vibration_event_count: int   = 0
    vibration_frequency_hz: float = 0.0


# ============================================================
# MACHINE HEALTH  (V2 — extended)
# ============================================================

class MachineHealth(BaseModel):
    health_score:  float = Field(ge=0.0, le=100.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)

    state:   str
    severity: str

    # V2 additions
    health_status:   str = "GOOD"
    condition:       str = "NORMAL"
    persistence_state: str = "NORMAL"
    raw_state:       str = "NORMAL"

    temperature_risk: float = Field(ge=0.0, le=1.0)
    vibration_risk:   float = Field(ge=0.0, le=1.0)
    current_risk:     float = Field(ge=0.0, le=1.0)

    # Persistence counters
    consecutive_warning:  int = 0
    consecutive_critical: int = 0
    consecutive_normal:   int = 0

    # Online ML context
    online_learning_active: bool = True
    model_samples:          int  = 0
    learned_baseline:       bool = False

    # Intelligence
    primary_contributor:      str = ""
    trend_summary:            str = ""
    recommended_action:       str = ""
    recommended_maintenance:  str = ""

    # Decision flags
    intervention_required: bool = False
    shutdown_allowed:      bool = False


# ============================================================
# HISTORY RECORD  (one row in the bounded observation log)
# ============================================================

class HistoryRecord(BaseModel):
    timestamp:          datetime
    temperature_c:      float
    current_a:          float
    vibration_detected: bool
    vibration_frequency_hz: float
    health_score:       Optional[float]
    anomaly_score:      Optional[float]
    health_status:      str
    intervention_state: str
    action:             str
    reason:             str


# ============================================================
# MACHINE STATE  (V2 — extended)
# ============================================================

class MachineState(BaseModel):
    machine_id: str = "MG-M1"

    connected:   bool              = False
    last_update: Optional[datetime] = None

    temperature_c:         float = 0.0
    current_a:             float = 0.0

    # SW-420
    vibration_detected:    bool  = False
    vibration_event_count: int   = 0
    vibration_frequency_hz: float = 0.0

    health: Optional[MachineHealth] = None

    relay_active: bool = False

    # V2 — intervention and trend state (stored as plain dicts
    # to avoid circular import; typed as Any for flexibility)
    intervention: Optional[Any] = None
    trend:        Optional[Any] = None
    learning:     Optional[Any] = None
