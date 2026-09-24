from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)

    temperature_c: float
    current_a: float

    # SW-420 digital vibration sensor
    vibration_detected: bool = False
    vibration_event_count: int = 0
    vibration_frequency_hz: float = 0.0


class MachineHealth(BaseModel):
    health_score: float = Field(ge=0.0, le=100.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)

    state: str
    severity: str

    temperature_risk: float = Field(ge=0.0, le=1.0)
    vibration_risk: float = Field(ge=0.0, le=1.0)
    current_risk: float = Field(ge=0.0, le=1.0)

    # Online ML information
    online_learning_active: bool = True
    model_samples: int = 0
    learned_baseline: bool = False

    recommended_action: str


class MachineState(BaseModel):
    machine_id: str = "Machine 1"

    connected: bool = False
    last_update: Optional[datetime] = None

    temperature_c: float = 0.0
    current_a: float = 0.0

    # SW-420
    vibration_detected: bool = False
    vibration_event_count: int = 0
    vibration_frequency_hz: float = 0.0

    health: Optional[MachineHealth] = None

    relay_active: bool = False