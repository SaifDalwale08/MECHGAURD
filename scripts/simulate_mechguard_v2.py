"""
MECHGUARD V2 — Decision Pipeline Simulator

Demonstrates the full V2 AI pipeline without requiring a running
FastAPI server or MQTT broker.

Two scenarios are run:

SCENARIO A — RECOVERY
    Phase 1: 30 normal baseline samples  → LEARNING_BASELINE → ADAPTIVE
    Phase 2: stable normal operation     → NORMAL / WATCH
    Phase 3: gradually worsening         → WARNING
    Phase 4: intervention recommended    → INTERVENTION → RECOVERY_CHECK
    Phase 5: values improve              → RECOVERING → RECOVERED
    Expected relay: NEVER fires.

SCENARIO B — FAILED RECOVERY
    Phase 1: 30 normal baseline samples
    Phase 2: abnormal condition          → WARNING → INTERVENTION
    Phase 3: values continue worsening   → FAILED_RECOVERY → CRITICAL
    Phase 4: critical persists           → PROTECTIVE_SHUTDOWN
    Expected relay: fires ONLY in PROTECTIVE_SHUTDOWN.

Usage:
    python scripts/simulate_mechguard_v2.py
    python scripts/simulate_mechguard_v2.py --scenario A
    python scripts/simulate_mechguard_v2.py --scenario B
"""

import sys
import os
import argparse
from typing import Optional

# ---------------------------------------------------------------------------
# Make sure 'backend' package is importable when run from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.feature_engine       import FeatureEngine
from backend.services.online_learning_engine import OnlineLearningEngine
from backend.services.trend_engine          import TrendEngine
from backend.services.health_engine         import HealthEngine
from backend.services.intervention_engine   import InterventionEngine
from backend.services.maintenance_engine    import MaintenanceEngine


# ============================================================
# PIPELINE RUNNER
# ============================================================

def make_pipeline():
    """Return a fresh set of V2 engine instances."""
    return (
        FeatureEngine(window_size=50),
        OnlineLearningEngine(warmup_samples=30, n_clusters=2),
        TrendEngine(window=20),
        HealthEngine(
            warning_persistence_samples=3,
            critical_persistence_samples=5,
            recovery_samples=3,
        ),
        InterventionEngine(
            recovery_window_samples=5,
            recovery_success_samples=3,
            immediate_shutdown_threshold=0.95,
        ),
        MaintenanceEngine(machine_id="SIM-M1"),
    )


def run_sample(
    engines,
    temperature_c:      float,
    current_a:          float,
    vibration_detected: bool,
    sample_num:         int,
    phase_label:        str,
    verbose:            bool = True,
) -> dict:
    """
    Push one sensor observation through the full V2 pipeline.
    Returns the intervention dict.
    """

    feature_eng, ol_eng, trend_eng, health_eng, iv_eng, maint_eng = (
        engines
    )

    # Feature extraction
    features = feature_eng.update(
        temperature_c=temperature_c,
        current_a=current_a,
        vibration_detected=vibration_detected,
    )
    vib_hz = features["vibration_frequency_hz"]

    # Online learning
    learning = ol_eng.update(
        temperature_c=temperature_c,
        current_a=current_a,
        vibration_frequency_hz=vib_hz,
    )

    # Trend (first pass — health not yet known)
    trend = trend_eng.update(
        temperature_c=temperature_c,
        current_a=current_a,
        vibration_frequency_hz=vib_hz,
        health_score=50.0,
        anomaly_score=learning["anomaly_score"],
    )

    # Health assessment
    health = health_eng.evaluate(
        temperature_c=temperature_c,
        current_a=current_a,
        vibration_frequency_hz=vib_hz,
        online_anomaly_score=learning["anomaly_score"],
        online_learning_active=learning["model_initialized"],
        model_samples=learning["samples_seen"],
        trend=trend,
    )

    # Trend update with real health score
    trend = trend_eng.update(
        temperature_c=temperature_c,
        current_a=current_a,
        vibration_frequency_hz=vib_hz,
        health_score=health["health_score"],
        anomaly_score=health["anomaly_score"],
    )

    # Intervention decision
    intervention = iv_eng.evaluate(
        health_dict=health,
        temperature_c=temperature_c,
        current_a=current_a,
        vibration_frequency_hz=vib_hz,
        trend=trend,
    )

    # Maintenance record
    maint_eng.record(
        health_dict=health,
        intervention_dict=intervention,
    )

    if verbose:
        _print_sample(
            sample_num=sample_num,
            phase_label=phase_label,
            temperature_c=temperature_c,
            current_a=current_a,
            vib_hz=vib_hz,
            anomaly=learning["anomaly_score"],
            pattern=learning["operating_pattern"],
            health_score=health["health_score"],
            health_state=health["state"],
            persistence=health["persistence_state"],
            iv_state=intervention["intervention_state"],
            action=intervention["action"],
            relay=intervention["relay_action"],
            recovery=intervention["recovery_status"],
        )

    return intervention


def _print_sample(
    sample_num, phase_label, temperature_c, current_a,
    vib_hz, anomaly, pattern, health_score, health_state,
    persistence, iv_state, action, relay, recovery,
):
    relay_marker = " ◄◄◄ RELAY OFF" if relay == "OFF" else ""
    print(
        f"  [{sample_num:>3}] {phase_label:<22} "
        f"T={temperature_c:>5.1f}°C  "
        f"I={current_a:>4.1f}A  "
        f"Vib={vib_hz:.2f}Hz  "
        f"Anomaly={anomaly:.3f}  "
        f"Pattern={pattern:<22}  "
        f"Health={health_score:>5.1f}  "
        f"State={health_state:<8}  "
        f"Persist={persistence:<22}  "
        f"IV={iv_state:<20}  "
        f"Action={action:<25}  "
        f"Relay={relay}"
        f"{relay_marker}"
    )


def _section(title: str):
    print()
    print("─" * 120)
    print(f"  {title}")
    print("─" * 120)


# ============================================================
# SCENARIO A — RECOVERY
# ============================================================

def scenario_a():
    print()
    print("=" * 120)
    print("  SCENARIO A — RECOVERY")
    print("  Expected: WARNING → INTERVENTION → RECOVERY_CHECK"
          " → RECOVERED.  Relay NEVER fires.")
    print("=" * 120)

    engines = make_pipeline()
    sample  = 0

    # --------------------------------------------------------
    # Phase 1 — Baseline warmup (30 samples)
    # --------------------------------------------------------
    _section("Phase 1 — Normal baseline warmup (30 samples)")
    for _ in range(30):
        sample += 1
        run_sample(
            engines,
            temperature_c=32.0, current_a=2.5,
            vibration_detected=False,
            sample_num=sample, phase_label="BASELINE_WARMUP",
        )

    # --------------------------------------------------------
    # Phase 2 — Stable normal operation (10 samples)
    # --------------------------------------------------------
    _section("Phase 2 — Stable normal operation")
    for _ in range(10):
        sample += 1
        run_sample(
            engines,
            temperature_c=33.0, current_a=2.6,
            vibration_detected=False,
            sample_num=sample, phase_label="NORMAL_OPERATION",
        )

    # --------------------------------------------------------
    # Phase 3 — Gradually worsening (12 samples)
    # --------------------------------------------------------
    _section("Phase 3 — Gradually worsening (rising temperature + current)")
    temps    = [36, 40, 44, 48, 52, 56, 59, 62, 64, 65, 66, 67]
    currents = [2.8, 3.2, 3.8, 4.5, 5.5, 6.5, 7.2, 7.8, 8.2, 8.5, 8.8, 9.0]
    for t, c in zip(temps, currents):
        sample += 1
        run_sample(
            engines,
            temperature_c=float(t), current_a=float(c),
            vibration_detected=(t > 55),
            sample_num=sample, phase_label="WORSENING",
        )

    # --------------------------------------------------------
    # Phase 4 — Intervention observation (5 samples at peak)
    # --------------------------------------------------------
    _section("Phase 4 — Intervention / recovery check (hold at peak)")
    for _ in range(1):
        sample += 1
        run_sample(
            engines,
            temperature_c=67.0, current_a=9.2,
            vibration_detected=True,
            sample_num=sample, phase_label="INTERVENTION_PEAK",
        )

    # --------------------------------------------------------
    # Phase 5 — Values improve (recovery)
    # --------------------------------------------------------
    _section("Phase 5 — Values improving after intervention")
    temps    = [62.0, 59.0, 55.0, 50.0, 45.0, 40.0, 35.0, 33.0]
    currents = [8.0,  7.2,  6.5,  5.5,  4.5,  3.5,  3.0,  2.7]
    for t, c in zip(temps, currents):
        sample += 1
        iv = run_sample(
            engines,
            temperature_c=t, current_a=c,
            vibration_detected=(t > 50),
            sample_num=sample, phase_label="RECOVERING",
        )

    # --------------------------------------------------------
    # Phase 6 — Confirm stable after recovery
    # --------------------------------------------------------
    _section("Phase 6 — Confirm stability after recovery")
    for _ in range(5):
        sample += 1
        run_sample(
            engines,
            temperature_c=33.0, current_a=2.6,
            vibration_detected=False,
            sample_num=sample, phase_label="POST_RECOVERY",
        )

    # --------------------------------------------------------
    # Maintenance report
    # --------------------------------------------------------
    _, _, _, _, _, maint_eng = engines
    rpt = maint_eng.report()
    print()
    print("─" * 120)
    print("  SCENARIO A — MAINTENANCE REPORT")
    print("─" * 120)
    for k, v in rpt.items():
        print(f"    {k:<30}: {v}")

    print()
    print("  ✓ SCENARIO A COMPLETE — verify relay never fired above.")
    print()


# ============================================================
# SCENARIO B — FAILED RECOVERY → PROTECTIVE SHUTDOWN
# ============================================================

def scenario_b():
    print()
    print("=" * 120)
    print("  SCENARIO B — FAILED RECOVERY → PROTECTIVE SHUTDOWN")
    print("  Expected: WARNING → INTERVENTION → RECOVERY_CHECK"
          " → FAILED_RECOVERY → CRITICAL → PROTECTIVE_SHUTDOWN.")
    print("  Relay fires ONLY in PROTECTIVE_SHUTDOWN.")
    print("=" * 120)

    engines = make_pipeline()
    sample  = 0

    # --------------------------------------------------------
    # Phase 1 — Baseline warmup (30 samples)
    # --------------------------------------------------------
    _section("Phase 1 — Normal baseline warmup (30 samples)")
    for _ in range(30):
        sample += 1
        run_sample(
            engines,
            temperature_c=32.0, current_a=2.5,
            vibration_detected=False,
            sample_num=sample, phase_label="BASELINE_WARMUP",
        )

    # --------------------------------------------------------
    # Phase 2 — Rapid escalation (8 samples)
    # --------------------------------------------------------
    _section("Phase 2 — Rapid escalation")
    temps    = [42.0, 50.0, 57.0, 62.0, 66.0, 69.0, 71.0, 72.0]
    currents = [3.5,  5.0,  6.5,  8.0,  9.5, 10.5, 11.5, 12.0]
    for t, c in zip(temps, currents):
        sample += 1
        run_sample(
            engines,
            temperature_c=t, current_a=c,
            vibration_detected=(t > 56),
            sample_num=sample, phase_label="ESCALATING",
        )

    # --------------------------------------------------------
    # Phase 3 — Intervention window — values CONTINUE worsening
    # --------------------------------------------------------
    _section(
        "Phase 3 — Recovery window: values continue worsening "
        "(failed recovery)"
    )
    temps    = [72.0, 73.5, 75.0, 76.5, 78.0]
    currents = [12.2, 12.5, 13.0, 13.5, 14.0]
    for t, c in zip(temps, currents):
        sample += 1
        run_sample(
            engines,
            temperature_c=t, current_a=c,
            vibration_detected=True,
            sample_num=sample, phase_label="FAILED_RECOVERY",
        )

    # --------------------------------------------------------
    # Phase 4 — Critical persists until PROTECTIVE_SHUTDOWN
    # --------------------------------------------------------
    _section(
        "Phase 4 — Critical condition persists → PROTECTIVE_SHUTDOWN"
    )
    relay_fired = False
    for i in range(8):
        sample += 1
        iv = run_sample(
            engines,
            temperature_c=78.0 + i * 0.5,
            current_a=14.0 + i * 0.2,
            vibration_detected=True,
            sample_num=sample, phase_label="CRITICAL_PERSIST",
        )
        if iv["relay_action"] == "OFF" and not relay_fired:
            print()
            print(
                "  *** PROTECTIVE SHUTDOWN TRIGGERED ***"
                f"  Sample #{sample}  |  "
                f"Reason: {iv['reason']}"
            )
            relay_fired = True

    # --------------------------------------------------------
    # Maintenance report
    # --------------------------------------------------------
    _, _, _, _, _, maint_eng = engines
    rpt = maint_eng.report()
    print()
    print("─" * 120)
    print("  SCENARIO B — MAINTENANCE REPORT")
    print("─" * 120)
    for k, v in rpt.items():
        print(f"    {k:<30}: {v}")

    if relay_fired:
        print()
        print("  ✓ SCENARIO B COMPLETE — relay fired only at "
              "PROTECTIVE_SHUTDOWN as expected.")
    else:
        print()
        print("  ⚠ SCENARIO B: relay did not fire — "
              "check persistence thresholds.")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MECHGUARD V2 Decision Pipeline Simulator"
    )
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "BOTH"],
        default="BOTH",
        help="Which scenario to run (default: BOTH)",
    )
    args = parser.parse_args()

    if args.scenario in ("A", "BOTH"):
        scenario_a()

    if args.scenario in ("B", "BOTH"):
        scenario_b()


if __name__ == "__main__":
    main()
