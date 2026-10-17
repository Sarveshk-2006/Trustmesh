"""TrustMesh Dynamic Sensor Trust Scoring & Explainable Recommendations Engine.

Provides transparent, deterministic, heuristic trust evaluation based strictly
on measured telemetry evidence, data-quality metrics, and cross-sensor classification.
Notice: All scores are heuristic prototype indicators (0-100), not calibrated probabilities.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.data_pipeline import REQUIRED_SENSORS
from src.detection_engine import (
    VERDICT_NORMAL,
    VERDICT_PROCESS_EVENT,
    VERDICT_SENSOR_FAULT,
    VERDICT_UNCERTAIN,
)

# Trust level classifications
TRUST_HIGH = "HIGH"
TRUST_MODERATE = "MODERATE"
TRUST_LOW = "LOW"

# Score thresholds
THRESHOLD_HIGH = 80.0
THRESHOLD_MODERATE = 50.0


def classify_trust_level(score: float) -> str:
    """Classify numeric trust score into HIGH, MODERATE, or LOW.

    Args:
        score: Numeric trust score between 0.0 and 100.0.

    Returns:
        One of 'HIGH', 'MODERATE', or 'LOW'.
    """
    if score >= THRESHOLD_HIGH:
        return TRUST_HIGH
    elif score >= THRESHOLD_MODERATE:
        return TRUST_MODERATE
    else:
        return TRUST_LOW


def generate_sensor_recommendation(
    sensor: str,
    trust_score: float,
    trust_level: str,
    is_stuck: bool,
    nan_count: int,
    is_spike: bool,
    is_process: bool,
    has_uncertainty: bool,
) -> str:
    """Generate an explainable, actionable recommendation for a sensor.

    Safety rule: Never recommend suppressing a genuine safety alarm solely
    because a trust score is low.

    Args:
        sensor: Sensor identifier.
        trust_score: Numeric trust score.
        trust_level: HIGH, MODERATE, or LOW.
        is_stuck: Whether sensor output is frozen.
        nan_count: Number of missing values.
        is_spike: Whether deviation was an isolated transient impulse.
        is_process: Whether sensor participated in a verified process event.
        has_uncertainty: Whether cross-sensor evidence is ambiguous.

    Returns:
        Actionable recommendation string.
    """
    if is_process:
        return (
            f"Continue nominal monitoring. Telemetry changes in `{sensor}` reliably reflect "
            "a verified multi-sensor process state transition."
        )

    if has_uncertainty:
        return (
            f"Investigate contradictory evidence for `{sensor}`; conduct physical field verification "
            "before adjusting control loops or setpoints. Do not suppress safety alarms."
        )

    if trust_level == TRUST_HIGH:
        if is_spike:
            return (
                f"Review electrical shielding and transient filtering for `{sensor}`; "
                "continue standard monitoring for recurrence."
            )
        return f"Continue nominal telemetry monitoring. `{sensor}` exhibits high operational fidelity."

    elif trust_level == TRUST_MODERATE:
        if nan_count > 0:
            return (
                f"Check telemetry connection and communication buffers for `{sensor}` to prevent "
                "further packet drops."
            )
        return (
            f"Schedule routine calibration and zero-check for `{sensor}`; compare readings against "
            "redundant instruments or physical mass balance. Do not suppress safety alarms."
        )

    else:  # TRUST_LOW
        if is_stuck:
            return (
                f"Immediate instrument maintenance required for `{sensor}`: inspect physical probe, "
                "impeller, or transmitter for mechanical blockage or frozen output latching. "
                "Do not suppress safety alarms."
            )
        if nan_count > 0:
            return (
                f"Critical signal loss on `{sensor}`: inspect field wiring, communication gateway, "
                "and acquisition hardware immediately. Do not suppress safety alarms."
            )
        return (
            f"Perform instrument recalibration or sensor replacement for `{sensor}`. Severe uncorroborated "
            "deviations detected from baseline process envelope. Do not suppress safety alarms."
        )


def calculate_sensor_trust(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dynamic, explainable trust scores for all monitored sensors.

    Computes bounded penalties from empirical evidence (anomalies, missing data,
    frozen readings, transient spikes, and cross-sensor correlation).
    Does not use scenario names, ground-truth labels, or hardcoded verdicts.
    The original analysis result is never mutated.

    Args:
        analysis_result: Output dictionary from analyze_sensor_data().

    Returns:
        Dict containing:
            - 'overall_trust_score': Mean trust score across monitored sensors.
            - 'has_uncertainty': Boolean indicating unresolved cross-sensor ambiguity.
            - 'uncertainty_note': Explanatory string when uncertainty exists.
            - 'system_status': Concise high-level operational status.
            - 'sensors': Dict mapping sensor names to individual trust results.

    Raises:
        ValueError: If analysis_result is invalid, empty, or missing required keys.
    """
    # 1. Validate input structure
    if not isinstance(analysis_result, dict):
        raise ValueError(
            f"Input must be a dictionary from analyze_sensor_data(), got {type(analysis_result).__name__}."
        )

    required_keys = ["verdict", "summary", "evidence", "data"]
    missing = [k for k in required_keys if k not in analysis_result]
    if missing:
        raise ValueError(f"Incomplete analysis result; missing required keys: {missing}")

    verdict = analysis_result["verdict"]
    summary = analysis_result["summary"]
    evidence = analysis_result["evidence"]
    annotated_df = analysis_result["data"]

    if not isinstance(annotated_df, pd.DataFrame):
        raise ValueError("analysis_result['data'] must be a pandas DataFrame.")

    total_samples = summary.get("total_samples", len(annotated_df))
    if total_samples <= 0:
        raise ValueError("total_samples must be greater than zero.")

    stuck_sensors = set(evidence.get("stuck_sensors", []))
    significant_sensors = set(evidence.get("significant_sensors", []))

    # 2. Uncertainty assessment
    has_uncertainty = verdict == VERDICT_UNCERTAIN
    uncertainty_note = (
        analysis_result.get("explanation", "Cross-sensor evidence exhibits unresolved ambiguity.")
        if has_uncertainty
        else ""
    )

    per_sensor_results: Dict[str, Dict[str, Any]] = {}
    trust_scores_list: List[float] = []

    # 3. Compute heuristic trust score per sensor
    for sensor in REQUIRED_SENSORS:
        score = 100.0
        reasons: List[str] = []

        # Empirical measurements
        nan_count = int(annotated_df[sensor].isna().sum()) if sensor in annotated_df else 0
        is_stuck = sensor in stuck_sensors
        anom_col = f"{sensor}_anomaly"
        score_col = f"{sensor}_score"

        anom_count = int(annotated_df[anom_col].sum()) if anom_col in annotated_df else 0
        r_anom = anom_count / total_samples
        nan_ratio = nan_count / total_samples

        s_scores = annotated_df[score_col] if score_col in annotated_df else pd.Series(0.0, index=annotated_df.index)
        max_z = float(s_scores.max()) if not s_scores.empty else 0.0
        mean_z = float(s_scores.mean()) if not s_scores.empty else 0.0
        crest_factor = max_z / (mean_z + 1e-4)

        # Transient spike: high crest factor, large sudden peak, but un-stuck and clean
        is_spike = (crest_factor >= 10.0 and max_z >= 8.0 and not is_stuck and nan_count == 0)
        is_process_participant = (verdict == VERDICT_PROCESS_EVENT and sensor in significant_sensors)

        # A. Missing Data Penalty
        if nan_count > 0:
            nan_penalty = min(60.0, 30.0 + nan_ratio * 150.0)
            score -= nan_penalty
            reasons.append(
                f"Telemetry loss: {nan_count} missing (NaN) readings detected ({nan_ratio * 100:.1f}% of stream)."
            )

        # B. Stuck / Frozen Reading Penalty
        if is_stuck:
            score -= 55.0
            reasons.append("Sensor output is frozen at a static constant reading (stuck sensor).")

        # C. Anomaly & Operational Context Deductions
        elif is_process_participant:
            # Genuine coordinated process event: readings changed because plant state shifted,
            # not because the sensor failed. Minimal bounded deduction.
            process_penalty = min(4.0, r_anom * 8.0)
            score -= process_penalty
            reasons.append(
                "Telemetry shifted in consistent physical coordination with multi-sensor process event."
            )

        elif has_uncertainty and sensor in significant_sensors:
            # Contradictory evidence: apply moderate penalty, but do not force LOW without proof
            uncertain_penalty = min(28.0, 16.0 + r_anom * 50.0)
            score -= uncertain_penalty
            reasons.append(
                "Telemetry displays conflicting cross-sensor dynamics; physical root cause is ambiguous."
            )

        elif is_spike:
            # Short isolated spike: smaller bounded penalty than sustained drift or data loss
            spike_penalty = 15.0
            score -= spike_penalty
            reasons.append(
                f"Isolated transient spike detected (max deviation {max_z:.1f}z, crest factor {crest_factor:.1f}); "
                "minimal persistent degradation."
            )

        elif anom_count > 0:
            # Sustained isolated deviation / drift
            drift_penalty = min(50.0, 24.0 + r_anom * 100.0)
            score -= drift_penalty
            reasons.append(
                f"Sustained uncorroborated deviation/drift ({anom_count} anomalous samples, {r_anom * 100:.1f}% of stream)."
            )

        else:
            reasons.append("Operating stably within nominal baseline tolerance.")

        # Ensure bounded range [0.0, 100.0]
        final_score = round(max(0.0, min(100.0, score)), 1)
        trust_level = classify_trust_level(final_score)
        trust_scores_list.append(final_score)

        # Generate contextual recommendation
        recommendation = generate_sensor_recommendation(
            sensor=sensor,
            trust_score=final_score,
            trust_level=trust_level,
            is_stuck=is_stuck,
            nan_count=nan_count,
            is_spike=is_spike,
            is_process=is_process_participant,
            has_uncertainty=has_uncertainty and (sensor in significant_sensors),
        )

        per_sensor_results[sensor] = {
            "sensor": sensor,
            "trust_score": final_score,
            "trust_level": trust_level,
            "reasons": reasons,
            "recommended_action": recommendation,
            "evidence": {
                "anomaly_count": anom_count,
                "anomaly_rate_pct": round(r_anom * 100, 2),
                "missing_count": nan_count,
                "is_stuck": is_stuck,
                "max_z_score": round(max_z, 2),
                "is_spike": is_spike,
            },
        }

    # System-level trust score
    overall_trust = round(sum(trust_scores_list) / len(trust_scores_list), 1) if trust_scores_list else 100.0

    # System status message
    low_trust_sensors = [s for s, r in per_sensor_results.items() if r["trust_level"] == TRUST_LOW]
    mod_trust_sensors = [s for s, r in per_sensor_results.items() if r["trust_level"] == TRUST_MODERATE]

    if low_trust_sensors:
        system_status = f"Critical sensor degradation suspected on: {', '.join(low_trust_sensors)}."
    elif mod_trust_sensors:
        system_status = f"Moderate trust degradation observed on: {', '.join(mod_trust_sensors)}."
    elif has_uncertainty:
        system_status = "System under investigation: contradictory cross-sensor evidence observed."
    else:
        system_status = "All sensors operating with high telemetry trust."

    return {
        "overall_trust_score": overall_trust,
        "has_uncertainty": has_uncertainty,
        "uncertainty_note": uncertainty_note,
        "system_status": system_status,
        "sensors": per_sensor_results,
    }
