# MECHGUARD V2

**Adaptive Machine Monitoring System — AI-First Architecture**

---

## Overview

MECHGUARD V2 is an embedded + cloud machine monitoring prototype that
combines low-cost IoT sensors, an incremental online machine-learning
pipeline, and a structured intervention engine to protect industrial
machines from damage caused by sustained abnormal operating conditions.

> **Important capability statement:**
> MECHGUARD's current live AI performs **adaptive unsupervised anomaly
> detection** and **machine-health assessment**. It learns the machine's
> operating behaviour from incoming sensor data. It does **not** claim
> universal exact failure-time prediction.

---

## V2 Architecture: SENSE → LEARN → ASSESS → INTERVENE → VERIFY → PROTECT → REPORT

```
Sensors (ESP8266)
      │
      ▼
Feature Engine          SW-420 vibration event rate, DS18B20 temp, ACS712 current
      │
      ▼
Online Learning         Adaptive unsupervised anomaly detection
(MiniBatchKMeans         Learns machine operating behaviour incrementally
 + StandardScaler)       Produces anomaly score [0, 1]
      │
      ▼
Trend Engine            Rolling least-squares slope per metric
                        Direction: RISING / STABLE / FALLING
      │
      ▼
Health Engine V2        Persistence-aware risk scoring
                        TRANSIENT vs PERSISTENT anomaly classification
                        intervention_required / shutdown_allowed flags
      │
      ▼
Intervention Engine     8-state decision machine
                        NORMAL → WATCH → WARNING → INTERVENTION
                        → RECOVERY_CHECK → RECOVERED / FAILED_RECOVERY
                        → CRITICAL → PROTECTIVE_SHUTDOWN
      │
      ▼
Maintenance Engine      Event accumulation, dominant risk, priority
                        Structured maintenance intelligence report
      │
      ▼
Dashboard (Frontend)    Live status, trends, alerts, history
      │
      ▼
Relay (ESP8266)         Protective shutdown — ONLY when justified
```

---

## Why Immediate Shutdown Was Replaced

**V1 behaviour:** Any sensor reading above a threshold → relay OFF.

**Problem:** This causes unnecessary production loss. A single
temperature spike (e.g. ambient heat, brief overload) should not stop
the machine. The judges correctly identified this as a weakness.

**V2 behaviour:**

1. A single abnormal sample is classified as a TRANSIENT anomaly.
2. The system monitors subsequent readings.
3. Only after `warning_persistence_samples` consecutive abnormal readings
   does the state escalate to PERSISTENT.
4. At PERSISTENT the system recommends controlled intervention
   (load reduction) rather than immediate shutdown.
5. After intervention, recovery is monitored for `recovery_window_samples`.
6. If the machine improves → RECOVERED → return to normal operation.
7. Only if the machine does **not** recover AND the condition persists
   through `critical_persistence_samples` does PROTECTIVE_SHUTDOWN occur.

---

## Persistence Configuration

All thresholds are configurable in `backend/config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `HEALTH_WARNING_PERSISTENCE_SAMPLES` | 3 | Consecutive WARNING samples before PERSISTENT |
| `HEALTH_CRITICAL_PERSISTENCE_SAMPLES` | 5 | Consecutive CRITICAL samples before PROTECTIVE_SHUTDOWN allowed |
| `HEALTH_RECOVERY_SAMPLES` | 3 | Consecutive normal samples to confirm RECOVERED |
| `INTERVENTION_RECOVERY_WINDOW_SAMPLES` | 5 | Samples to observe after intervention |
| `INTERVENTION_RECOVERY_SUCCESS_SAMPLES` | 3 | Consecutive improved samples to confirm recovery |
| `INTERVENTION_IMMEDIATE_SHUTDOWN_THRESHOLD` | 0.95 | Combined risk threshold for immediate shutdown |

---

## Hardware

| Component | Role |
|---|---|
| ESP8266 | Wi-Fi sensor node, MQTT client, relay driver |
| DS18B20 | Temperature sensor |
| ACS712 | Current sensor |
| SW-420 | Digital vibration / shock threshold sensor |
| Relay module | Prototype protective shutdown pathway |

**SW-420 note:** SW-420 is a digital threshold sensor. It produces
vibration-detected / not-detected events. MECHGUARD uses event count
and event frequency as vibration features. It does **not** produce
acceleration RMS, vibration spectrum, or industrial vibration amplitude.

---

## AI Pipeline Detail

### Online Learning Engine

- Algorithm: `MiniBatchKMeans` + `StandardScaler` (scikit-learn)
- Update method: `partial_fit` (online, incremental)
- Warmup: 30 baseline samples before the model activates
- Features: `[temperature_c, current_a, vibration_frequency_hz]`
- Output: `anomaly_score` [0, 1], `operating_pattern`, `baseline_state`
- Operating patterns: `LEARNING_BASELINE`, `NORMAL_OPERATION`,
  `ELEVATED_OPERATION`, `ABNORMAL_OPERATION`

### Trend Engine

- Method: Closed-form least-squares slope over rolling window
- Window size: 20 samples (configurable)
- Metrics: temperature, current, vibration frequency, health score,
  anomaly score
- Output: slope value + direction (`RISING` / `STABLE` / `FALLING`)
  per metric

### Health Engine V2

- Sensor risk: piecewise linear [0, 1] per sensor
- Combined risk: weighted sum (temperature 35%, current 30%,
  vibration 35%) blended with anomaly score
- Persistence: stateful consecutive-sample counters
- Output: `health_score`, `health_status`, `condition`,
  `persistence_state`, `intervention_required`, `shutdown_allowed`,
  `primary_contributor`, `trend_summary`, `recommended_action`,
  `recommended_maintenance`

### Intervention Engine

8-state machine with the following valid transitions:

```
NORMAL → WATCH → WARNING → INTERVENTION → RECOVERY_CHECK
                                               │
                                    ┌──────────┴──────────┐
                                RECOVERED            FAILED_RECOVERY
                                    │                     │
                                 NORMAL/WATCH          CRITICAL
                                                          │
                                                  PROTECTIVE_SHUTDOWN
```

The relay OFF command is issued **only** in `PROTECTIVE_SHUTDOWN` state.

### Maintenance Intelligence Engine

Accumulates event counts across the monitoring session and generates:
- Dominant risk sensor
- Maintenance priority: `LOW / MEDIUM / HIGH / CRITICAL`
- Structured natural-language recommendation
- Full maintenance summary

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root health and version info |
| GET | `/api/health` | API health check |
| GET | `/api/machine` | Raw machine state |
| GET | `/api/machine/health` | Current health assessment |
| GET | `/api/machine/sensors` | Current sensor readings |
| GET | `/api/machine/history?limit=50` | Bounded observation history |
| GET | `/api/machine/maintenance-report` | Maintenance intelligence report |
| GET | `/api/machines/MG-M1/live` | Full live dashboard snapshot |
| POST | `/api/sensor-data` | Manual sensor data ingestion |

---

## Live API Response Structure (`/api/machines/MG-M1/live`)

```json
{
  "machine":   { "id", "name", "status", "connection" },
  "sensors":   { "temperature", "current", "vibration" },
  "health":    { "score", "status", "trend", "reason" },
  "analytics": {
    "condition", "anomaly_score", "anomaly_status",
    "operating_pattern", "trend", "primary_contributor"
  },
  "protection": { "relay", "mode", "reason", "triggeredBy", ... },
  "intervention": {
    "state", "action", "reason", "recovery_status",
    "shutdown_allowed", "relay_action", "requires_operator"
  },
  "maintenance": { "priority", "recommendation", "summary" },
  "history":   [ ... last 10 observations ... ],
  "alerts":    [ ... active alerts ... ],
  "system":    { "esp8266", "backend", "api", "relay", "sensors" },
  "prediction": { "available": false, "model": null, "rul_normalized": null }
}
```

---

## Running the Simulation

```bash
# Both scenarios
python scripts/simulate_mechguard_v2.py

# Only recovery scenario
python scripts/simulate_mechguard_v2.py --scenario A

# Only failed recovery scenario
python scripts/simulate_mechguard_v2.py --scenario B
```

---

## Starting the Backend

```bash
cd D:\MECHGUARD
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

## Frontend Integration Contract

The backend is the single source of truth for machine state, health,
anomaly, trends, intervention, and relay protection. The examples below are
the response shapes returned by the running API. Empty-start values are shown
where the backend has not received sensor data yet.

### `GET /api/health`

Purpose: service liveness.

```json
{
  "status": "online",
  "service": "MECHGUARD API",
  "timestamp": "2026-09-24T19:56:00.865937"
}
```

Types: `status`, `service`, and `timestamp` are strings. `timestamp` is an
ISO-8601 local datetime.

### `GET /api/machines/MG-M1/live`

Purpose: complete current dashboard snapshot. The top-level response is an
object containing `machine`, `sensors`, `health`, `analytics`, `protection`,
`relayEvents`, `alerts`, `system`, `prediction`, `intervention`,
`maintenance`, `history`, and `timestamp`.

Important fields and types:

- `machine.id`: string, always `MG-M1`; `name`, `status`, and `connection` are strings.
- `sensors.temperature.value`: number or `null`, degrees Celsius.
- `sensors.current.value`: number or `null`, amperes.
- `sensors.vibration.state`: string, `VIBRATION` or `NO_VIBRATION`.
- `sensors.vibration.event_count`: integer, rolling SW-420 event count.
- `sensors.vibration.frequency_hz`: number, rolling event rate in Hz using the configured one-second sample interval and bounded feature window. This is not a physical vibration spectrum.
- `health.score`: number or `null`, 0-100; `status`, `trend`, and `reason` are strings.
- `analytics.anomaly_score`: number or `null`, 0-1; `operating_pattern` is a backend ML state string; `trend` is an object; `primary_contributor` is a string.
- `protection.relay`: string, `OFFLINE`, `ARMED`, or `TRIGGERED`; `triggeredBy` is an object of booleans; `lastChangedAt` is an ISO-8601 string or `null`.
- `intervention`: object containing string `state`, `action`, `reason`, and `recovery_status`, booleans `shutdown_allowed` and `requires_operator`, and string `relay_action`.
- `prediction.available`: boolean and currently `false`; `model` and `rul_normalized` are `null`.
- `history` and `alerts` are arrays. `timestamp` is an ISO-8601 string.

Nullable fields are the initial/no-data sensor values, initial health and
anomaly values, `protection.lastChangedAt`, and unavailable prediction
fields. The backend does not expose RMS, peak, crest-factor, acceleration, or
MPU6050 fields.

### `GET /api/machine/history`

Purpose: recent observations. The response is:

```json
{
  "machine_id": "MG-M1",
  "count": 0,
  "limit": 50,
  "records": []
}
```

`machine_id` is a string, `count` and `limit` are integers, and `records` is
an array ordered newest first. At most 200 records are retained. Each record
contains an ISO-8601 `timestamp`, numeric `temperature_c`, `current_a`,
`vibration_frequency_hz`, `health_score`, and `anomaly_score`, boolean
`vibration_detected`, and string `health_status`, `intervention_state`,
`action`, and `reason`. History relay semantics are represented by the
intervention state and action: only `PROTECTIVE_SHUTDOWN` may use a shutdown
action.

### `GET /api/machine/maintenance-report`

Purpose: accumulated maintenance intelligence. The response is an object with
string `machine_id`, `monitoring_period`, `current_health_status`,
`health_trend`, `dominant_risk`, `maintenance_priority`,
`recommended_action`, `maintenance_summary`, and `generated_at`; nullable
numeric `current_health`; and integer counters `total_samples`,
`anomaly_events`, `warning_events`, `intervention_events`,
`recovery_attempts`, `successful_recoveries`, `failed_recoveries`,
`recovery_events`, `critical_events`, and `shutdown_events`.

The recovery invariants are `successful_recoveries <= recovery_attempts`,
`failed_recoveries <= recovery_attempts`, and the sum of successful and
failed recoveries cannot exceed attempts. Shutdowns count only actual
protective-shutdown transitions.

### Intervention states and sensor rules

Implemented intervention states are `NORMAL`, `WATCH`, `WARNING`,
`INTERVENTION`, `RECOVERY_CHECK`, `RECOVERING`, `RECOVERED`,
`FAILED_RECOVERY`, `CRITICAL`, `CRITICAL_PERSISTENT`, and
`PROTECTIVE_SHUTDOWN`. Only the last state produces relay `OFF`.

Sensor mapping is fixed: DS18B20 → `temperature_c` in degrees Celsius;
ACS712 → `current_a` in amperes; SW-420 → `vibration_detected` as a digital
vibration/shock event. SW-420 is not an accelerometer. Its exposed event rate
uses `SENSOR_SAMPLE_INTERVAL_SECONDS` (one second by default) over the
bounded feature window.

Frontend code must not calculate health or anomaly independently, invent RUL,
infer vibration RMS or acceleration, override intervention state, directly
control the relay, or create separate machine-health logic. Use backend values
as the authoritative contract.

## Validation and Prototype Limitations

The backend is the authoritative V2 pipeline for machine `MG-M1`:
feature extraction, stable baseline learning, anomaly assessment, ordered
trends, persistence-aware health, intervention/recovery decisions, and
maintenance accounting. The HTTP and MQTT ingestion paths share the live
machine state; history is bounded to 200 records.

### Baseline and anomaly model

The first 30 valid observations establish a `StandardScaler` reference and
initialize `MiniBatchKMeans`. The scaler remains fixed for anomaly scoring so
normal updates cannot move the reference distribution immediately. KMeans
adapts only for non-abnormal observations. Scores are normalized against the
warmup cluster-distance envelope and remain in `[0, 1]`.

### Sensor semantics and trends

DS18B20 supplies temperature, ACS712 supplies current, and SW-420 supplies a
digital vibration event. SW-420 does not provide acceleration, RMS, peak, or a
frequency spectrum. Its reported rate uses the configured
`SENSOR_SAMPLE_INTERVAL_SECONDS` (one second by default). Trend slopes are
ordered rolling sample slopes, not physical-time rates.

### Health, persistence, and protection

Health is bounded to 0-100 and risk/anomaly values to 0-1. Warning requires
three consecutive warning samples; critical persistence requires five
consecutive critical samples. Counters reset on recovery. Intervention begins
before shutdown. Recovery captures temperature, current, vibration, health,
and anomaly baselines and requires multiple improving observations. Failed
recovery is latched into critical handling; `PROTECTIVE_SHUTDOWN` is latched
until an explicit operator reset. Relay `OFF` is generated only in that final
state.

### MQTT and connection behavior

The development broker remains `127.0.0.1:1883`. MQTT failure is reported and
does not prevent the API from starting. Payloads must contain finite
`temperature_c`, `current_a`, and boolean `vibration_detected` values within
reasonable ranges. Invalid messages are rejected without stopping the service.
The machine becomes disconnected after `SENSOR_TIMEOUT_SECONDS` without a
valid update. Live RUL is intentionally unavailable because no valid feature
bridge exists for `MG-M1`.

### API and simulation checks

Run `python -m compileall backend`, then
`python scripts\\simulate_mechguard_v2.py`. The simulator contains a
successful recovery and a failed recovery ending in protective shutdown.
The primary API checks are `/api/health`,
`/api/machines/MG-M1/live`, `/api/machine/history`, and
`/api/machine/maintenance-report`.

This remains a prototype: the broker and ESP8266 may be unavailable, SW-420
is only a threshold event sensor, online clustering is unsupervised, and
protective shutdown requires an operator-controlled hardware deployment.

---

## Datasets

Raw dataset files are not stored in this repository.
See `datasets/README.md` for download instructions.

Datasets used for offline RUL model training:
- CWRU Bearing Dataset
- Paderborn Bearing Dataset
- NASA Milling Dataset

---

## Prototype Limitations

MECHGUARD V2 is a research prototype. The following limitations apply:

- **Anomaly detection is unsupervised.** The model learns operating
  behaviour but cannot identify specific fault types or guarantee
  detection of all failure modes.
- **Thresholds are not industrially validated.** The sensor risk
  thresholds in `health_engine.py` are prototype values and should
  be calibrated for specific machines.
- **SW-420 vibration is limited.** It provides digital event detection
  only — not acceleration RMS, spectrum, or industrial vibration
  amplitude.
- **Load reduction is a recommendation only.** MECHGUARD does not
  have verified variable-speed motor hardware. "REDUCE_LOAD" is an
  operator recommendation, not an automated hardware command.
- **MQTT broker is external.** The live MQTT broker is operated by a
  team member. The backend runs without it and degrades gracefully.
- **No zero false positives guarantee.** All anomaly detection systems
  can produce false positives or false negatives.
- **Not a certified safety system.** MECHGUARD is not a replacement
  for industrial safety-rated protection equipment.

---

## File Structure

```
MECHGUARD/
├── backend/
│   ├── api/
│   │   └── routes.py              API endpoints (V2)
│   ├── services/
│   │   ├── feature_engine.py      SW-420 feature extraction
│   │   ├── online_learning_engine.py  Adaptive anomaly detection
│   │   ├── trend_engine.py        Rolling trend analysis  [NEW]
│   │   ├── health_engine.py       Persistence-aware health scoring  [V2]
│   │   ├── intervention_engine.py 8-state decision machine  [NEW]
│   │   ├── maintenance_engine.py  Maintenance intelligence  [NEW]
│   │   ├── machine_state.py       State store + history  [V2]
│   │   └── mqtt_service.py        MQTT ingestion + pipeline  [V2]
│   ├── schemas/
│   │   └── machine.py             Pydantic schemas  [V2]
│   ├── config.py                  All configuration values  [V2]
│   └── main.py                    Application entry point  [V2]
├── frontend/                      Static HTML/CSS/JS dashboard
├── hardware/                      ESP8266 firmware
├── scripts/
│   ├── simulate_mechguard_v2.py   V2 pipeline simulator  [NEW]
│   └── simulate_mqtt.py           V1 MQTT simulator
├── models/piecuch_rul/            Trained RUL .joblib models
├── datasets/README.md             Dataset download instructions
├── ai/                            AI pipeline modules
└── README.md                      This file
```
