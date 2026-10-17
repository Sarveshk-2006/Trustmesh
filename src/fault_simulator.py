"""Deterministic fault simulator for TrustMesh sensor streams.

Provides reproducible fault injection into synthetic sensor telemetry while
preserving the original DataFrame unmodified.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.data_pipeline import REQUIRED_SENSORS, validate_sensor_data

# Supported scenario definitions with descriptive names and keys
SCENARIO_MAP: Dict[str, str] = {
    "normal operation": "normal",
    "normal": "normal",
    "gradual drift in one sensor": "drift",
    "gradual drift": "drift",
    "drift": "drift",
    "a short sudden spike": "spike",
    "short sudden spike": "spike",
    "spike": "spike",
    "a stuck sensor value": "stuck",
    "stuck sensor value": "stuck",
    "stuck": "stuck",
    "a block of missing measurements": "missing",
    "missing measurements": "missing",
    "missing data": "missing",
    "missing": "missing",
    "a coordinated process change affecting multiple related sensors": "coordinated_process_change",
    "coordinated process change": "coordinated_process_change",
    "coordinated_process_change": "coordinated_process_change",
    "a conflicting-evidence scenario for later uncertainty testing": "conflicting_evidence",
    "conflicting-evidence scenario": "conflicting_evidence",
    "conflicting evidence": "conflicting_evidence",
    "conflicting_evidence": "conflicting_evidence",
}

AVAILABLE_SCENARIOS: List[str] = [
    "normal operation",
    "gradual drift in one sensor",
    "a short sudden spike",
    "a stuck sensor value",
    "a block of missing measurements",
    "a coordinated process change affecting multiple related sensors",
    "a conflicting-evidence scenario for later uncertainty testing",
]


def normalize_scenario_name(scenario: str) -> Tuple[str, str]:
    """Normalize scenario input string to canonical key and descriptive title.

    Args:
        scenario: Scenario name or shorthand key.

    Returns:
        Tuple of (canonical_key, full_display_name).

    Raises:
        ValueError: If the scenario name is unrecognized.
    """
    clean = scenario.strip().lower()
    if clean not in SCENARIO_MAP:
        valid_options = ", ".join(f"'{s}'" for s in AVAILABLE_SCENARIOS)
        raise ValueError(
            f"Unrecognized scenario '{scenario}'. Supported scenarios are: {valid_options}"
        )

    key = SCENARIO_MAP[clean]
    # Find matching display name
    for display_name in AVAILABLE_SCENARIOS:
        if SCENARIO_MAP.get(display_name.lower()) == key:
            return key, display_name

    return key, scenario


def inject_fault(
    df: pd.DataFrame,
    scenario: str,
    target_sensor: str = "FIT101",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inject a deterministic fault or operational scenario into sensor telemetry.

    The original DataFrame is never modified; a modified copy is returned along
    with metadata describing the affected time window and sensors.

    Args:
        df: Input DataFrame containing Timestamp and required sensors.
        scenario: Scenario name to simulate (e.g. 'gradual drift in one sensor').
        target_sensor: Primary sensor targeted for single-sensor faults.

    Returns:
        Tuple of (modified_df, metadata_dict).

    Raises:
        ValueError: If target_sensor is invalid or scenario name is unrecognized.
    """
    # 1. Validate inputs
    validate_sensor_data(df)

    if target_sensor not in REQUIRED_SENSORS:
        valid_sensors = ", ".join(REQUIRED_SENSORS)
        raise ValueError(
            f"Invalid target sensor '{target_sensor}'. Must be one of: {valid_sensors}"
        )

    scenario_key, display_name = normalize_scenario_name(scenario)

    # 2. Work on an isolated copy to preserve original immutability
    out_df = df.copy(deep=True)
    n = len(out_df)

    if n < 10:
        raise ValueError("DataFrame must contain at least 10 samples for fault simulation.")

    metadata: Dict[str, Any] = {
        "scenario": display_name,
        "scenario_key": scenario_key,
        "target_sensor": target_sensor,
        "affected_sensors": [],
        "start_idx": None,
        "end_idx": None,
        "start_time": None,
        "end_time": None,
        "description": "",
    }

    # Helper for timestamp lookup
    def get_time(idx: int):
        return out_df["Timestamp"].iloc[idx]

    # 3. Deterministic injection logic
    if scenario_key == "normal":
        metadata["affected_sensors"] = []
        metadata["description"] = "Normal nominal operation. No faults injected."
        return out_df, metadata

    elif scenario_key == "drift":
        # Gradual linear drift from 40% to 70% of stream
        start_idx = int(n * 0.40)
        end_idx = int(n * 0.70)
        window_len = end_idx - start_idx

        # Calculate a realistic drift ramp (e.g., +25% of sensor dynamic mean)
        sensor_mean = float(out_df[target_sensor].mean())
        drift_magnitude = sensor_mean * 0.30 if sensor_mean != 0 else 1.0
        ramp = np.linspace(0.0, drift_magnitude, window_len)

        out_df.loc[out_df.index[start_idx:end_idx], target_sensor] += ramp

        metadata.update(
            {
                "affected_sensors": [target_sensor],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": get_time(start_idx),
                "end_time": get_time(end_idx - 1),
                "description": (
                    f"Gradual linear drift injected into {target_sensor} "
                    f"over samples [{start_idx}:{end_idx}] (+{drift_magnitude:.2f} max offset)."
                ),
            }
        )

    elif scenario_key == "spike":
        # Short sudden spike between 50% and 52% of stream
        start_idx = int(n * 0.50)
        end_idx = max(start_idx + 3, int(n * 0.52))

        sensor_mean = float(out_df[target_sensor].mean())
        spike_magnitude = abs(sensor_mean) * 0.85 if sensor_mean != 0 else 5.0

        out_df.loc[out_df.index[start_idx:end_idx], target_sensor] += spike_magnitude

        metadata.update(
            {
                "affected_sensors": [target_sensor],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": get_time(start_idx),
                "end_time": get_time(end_idx - 1),
                "description": (
                    f"Short sudden spike (+{spike_magnitude:.2f}) injected into {target_sensor} "
                    f"over samples [{start_idx}:{end_idx}]."
                ),
            }
        )

    elif scenario_key == "stuck":
        # Sensor output is frozen at a fixed constant reading from 35% to 65% of stream
        start_idx = int(n * 0.35)
        end_idx = int(n * 0.65)
        stuck_value = float(out_df[target_sensor].iloc[start_idx])

        out_df.loc[out_df.index[start_idx:end_idx], target_sensor] = stuck_value

        metadata.update(
            {
                "affected_sensors": [target_sensor],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": get_time(start_idx),
                "end_time": get_time(end_idx - 1),
                "description": (
                    f"Sensor {target_sensor} stuck at constant reading {stuck_value:.3f} "
                    f"over samples [{start_idx}:{end_idx}]."
                ),
            }
        )

    elif scenario_key == "missing":
        # Block of missing measurements (NaN) from 45% to 55% of stream
        start_idx = int(n * 0.45)
        end_idx = int(n * 0.55)

        out_df.loc[out_df.index[start_idx:end_idx], target_sensor] = np.nan

        metadata.update(
            {
                "affected_sensors": [target_sensor],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": get_time(start_idx),
                "end_time": get_time(end_idx - 1),
                "description": (
                    f"Block of {end_idx - start_idx} missing telemetry measurements (NaN) "
                    f"in {target_sensor} over samples [{start_idx}:{end_idx}]."
                ),
            }
        )

    elif scenario_key == "coordinated_process_change":
        # Coordinated shift across multiple coupled sensors: FIT101, LIT101, FIT301
        start_idx = int(n * 0.30)
        end_idx = int(n * 0.70)
        affected = ["FIT101", "LIT101", "FIT301"]

        # Coupled physical changes: higher pump duty cycle
        out_df.loc[out_df.index[start_idx:end_idx], "FIT101"] += 0.65  # +0.65 m3/h
        out_df.loc[out_df.index[start_idx:end_idx], "LIT101"] += 55.0  # +55 mm
        out_df.loc[out_df.index[start_idx:end_idx], "FIT301"] += 0.50  # +0.50 m3/h

        metadata.update(
            {
                "affected_sensors": affected,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": get_time(start_idx),
                "end_time": get_time(end_idx - 1),
                "description": (
                    "Coordinated process load change across coupled sensors "
                    f"({', '.join(affected)}) over samples [{start_idx}:{end_idx}]."
                ),
            }
        )

    elif scenario_key == "conflicting_evidence":
        # Target sensor reports severe anomaly/surge while coupled partner shows contradictory normal/drop
        start_idx = int(n * 0.35)
        end_idx = int(n * 0.65)

        # Primary contradiction pair: FIT101 and LIT101 (or target_sensor and counterpart)
        partner_sensor = "LIT101" if target_sensor != "LIT101" else "FIT101"
        affected = sorted(list({target_sensor, partner_sensor}))

        # Target spikes significantly
        sensor_mean = float(out_df[target_sensor].mean())
        surge = sensor_mean * 0.6 if sensor_mean != 0 else 2.0
        out_df.loc[out_df.index[start_idx:end_idx], target_sensor] += surge

        # Partner counter-intuitively drops or remains suppressed, creating conflicting evidence
        partner_mean = float(out_df[partner_sensor].mean())
        partner_drop = partner_mean * 0.25 if partner_mean != 0 else 1.0
        out_df.loc[out_df.index[start_idx:end_idx], partner_sensor] -= partner_drop

        metadata.update(
            {
                "affected_sensors": affected,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": get_time(start_idx),
                "end_time": get_time(end_idx - 1),
                "description": (
                    f"Conflicting-evidence scenario: {target_sensor} surges (+{surge:.2f}) "
                    f"while coupled {partner_sensor} drops (-{partner_drop:.2f}) over samples [{start_idx}:{end_idx}]."
                ),
            }
        )

    return out_df, metadata
