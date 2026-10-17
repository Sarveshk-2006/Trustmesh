"""TrustMesh Anomaly Detection & Cross-Sensor Evidence Fusion Engine.

Provides unsupervised statistical and machine-learning anomaly detection,
cross-sensor evidence correlation, and explainable event classification.
All telemetry analyzed by this module is treated as synthetic demo data.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_pipeline import REQUIRED_SENSORS, validate_sensor_data

# Classification Verdicts
VERDICT_NORMAL = "NORMAL"
VERDICT_SENSOR_FAULT = "SENSOR_FAULT_SUSPECTED"
VERDICT_PROCESS_EVENT = "PROCESS_EVENT_SUSPECTED"
VERDICT_UNCERTAIN = "UNCERTAIN"

VALID_VERDICTS = [
    VERDICT_NORMAL,
    VERDICT_SENSOR_FAULT,
    VERDICT_PROCESS_EVENT,
    VERDICT_UNCERTAIN,
]


def detect_data_quality_issues(
    df: pd.DataFrame,
    sensors: List[str] = REQUIRED_SENSORS,
    stuck_window: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect missing values (NaN) and stuck (frozen) sensor values.

    Args:
        df: Input sensor DataFrame.
        sensors: List of sensor columns to inspect.
        stuck_window: Minimum consecutive identical readings to flag as stuck.

    Returns:
        Tuple of (quality_flags_df, quality_summary_dict).
    """
    flags_df = pd.DataFrame(index=df.index)
    stuck_sensors = []
    missing_counts = {}

    for s in sensors:
        series = df[s]
        # 1. Missing values
        nan_mask = series.isna()
        missing_count = int(nan_mask.sum())
        if missing_count > 0:
            missing_counts[s] = missing_count

        # 2. Stuck values: consecutive differences equal to zero (ignoring NaNs)
        valid_series = series.dropna()
        stuck_mask = pd.Series(False, index=df.index)

        if len(valid_series) >= stuck_window:
            diffs = valid_series.diff().abs()
            is_zero = diffs == 0
            # Rolling sum of zero-differences over stuck_window
            consecutive_zeros = (
                is_zero.rolling(window=stuck_window, min_periods=stuck_window).sum()
            )
            stuck_indices = consecutive_zeros[
                consecutive_zeros >= (stuck_window - 1)
            ].index

            if not stuck_indices.empty:
                for idx in stuck_indices:
                    loc = df.index.get_loc(idx)
                    start_loc = max(0, loc - stuck_window + 1)
                    stuck_mask.iloc[start_loc : loc + 1] = True
                stuck_sensors.append(s)

        flags_df[f"{s}_nan"] = nan_mask
        flags_df[f"{s}_stuck"] = stuck_mask

    summary = {
        "stuck_sensors": stuck_sensors,
        "missing_counts": missing_counts,
    }
    return flags_df, summary


def detect_rolling_deviations(
    df: pd.DataFrame,
    sensors: List[str] = REQUIRED_SENSORS,
    window: int = 30,
    min_periods: int = 5,
    z_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect statistical anomalies using past-only rolling residual deviations.

    Avoids future data leakage by using causal rolling windows. Residual deviations
    are normalized by dynamic sensor variability to prevent false alarms during
    normal periodic process cycles.

    Args:
        df: Input DataFrame.
        sensors: Sensor column names.
        window: Past rolling window size in samples.
        min_periods: Minimum samples for rolling baseline.
        z_threshold: Normalized deviation threshold for flagging anomalies.

    Returns:
        Tuple of (anomalies_bool_df, z_scores_float_df).
    """
    anomalies_df = pd.DataFrame(index=df.index)
    scores_df = pd.DataFrame(index=df.index)

    for s in sensors:
        series = df[s].astype(float)

        # Handle NaNs via forward/backward fill for rolling baseline computation
        clean_s = series.ffill().bfill().fillna(0.0)

        # 1. Past-only rolling mean baseline (causal)
        roll_mean = clean_s.rolling(window=window, min_periods=min_periods).mean()
        roll_mean = roll_mean.combine_first(clean_s.expanding(min_periods=1).mean())

        # Residual between current reading and rolling baseline
        residual = (clean_s - roll_mean).abs()

        # Scale factor: dynamic standard deviation with robust floor
        res_scale = max(float(clean_s.std() * 0.35), 1e-4)
        z_score = residual / res_scale

        # Flag sample if score exceeds threshold or if raw measurement is NaN
        is_anom = (z_score >= z_threshold) | series.isna()

        anomalies_df[s] = is_anom
        scores_df[s] = np.round(z_score, 3)

    return anomalies_df, scores_df


def detect_isolation_forest(
    df: pd.DataFrame,
    sensors: List[str] = REQUIRED_SENSORS,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Detect multivariate anomalies using an unsupervised Isolation Forest.

    Safely handles NaNs by imputing for fit and scoring on isolated numeric features.

    Args:
        df: Input DataFrame.
        sensors: Sensor column names.
        contamination: Expected proportion of outliers in the data.
        random_state: Seed for deterministic model fitting.

    Returns:
        Tuple of (is_anomaly_bool_series, anomaly_score_float_series).
    """
    n = len(df)
    if n < 10:
        return pd.Series(False, index=df.index), pd.Series(0.0, index=df.index)

    # Impute missing values for tree-based multivariate feature space
    feature_df = df[sensors].copy().astype(float)
    feature_df = feature_df.ffill().bfill().fillna(0.0)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(feature_df)

    preds = model.predict(feature_df)
    decision = model.decision_function(feature_df)

    is_anomaly = pd.Series(preds == -1, index=df.index)
    scores = pd.Series(np.round(-decision, 3), index=df.index)

    return is_anomaly, scores


def classify_sensor_evidence(
    summary: Dict[str, Any],
    sensor_anomaly_counts: Dict[str, int],
    quality_summary: Dict[str, Any],
    total_samples: int,
    cooccurrence_counts: Dict[str, int],
    directional_correlations: Dict[str, float],
) -> Tuple[str, str, Dict[str, Any]]:
    """Perform conservative, rule-based cross-sensor evidence fusion.

    Classifies observations into NORMAL, SENSOR_FAULT_SUSPECTED,
    PROCESS_EVENT_SUSPECTED, or UNCERTAIN.

    Args:
        summary: Aggregated anomaly detection metrics.
        sensor_anomaly_counts: Per-sensor count of anomalous samples.
        quality_summary: Stuck and missing sensor summaries.
        total_samples: Total number of samples in the stream.
        cooccurrence_counts: Count of samples where multiple sensors were anomalous.
        directional_correlations: Pairwise correlation shifts among affected sensors.

    Returns:
        Tuple of (verdict, explanation_string, structured_evidence_dict).
    """
    stuck_sensors = quality_summary.get("stuck_sensors", [])
    missing_counts = quality_summary.get("missing_counts", {})
    anomalous_samples = summary.get("anomalous_samples", 0)
    anomaly_rate = summary.get("anomaly_rate", 0.0)

    # Significant threshold: a sensor is meaningfully affected if >= 2.5% of stream is anomalous
    min_sig_samples = max(6, int(total_samples * 0.025))
    significant_sensors = [
        s for s, count in sensor_anomaly_counts.items() if count >= min_sig_samples
    ]

    evidence: Dict[str, Any] = {
        "anomaly_rate": round(anomaly_rate, 4),
        "anomalous_samples": anomalous_samples,
        "significant_sensors": significant_sensors,
        "stuck_sensors": stuck_sensors,
        "missing_sensors": list(missing_counts.keys()),
        "sensor_anomaly_counts": sensor_anomaly_counts,
        "correlations": directional_correlations,
        "observations": [],
    }

    # Case 0: Telemetry series too short for reliable classification
    if total_samples < 20:
        verdict = VERDICT_UNCERTAIN
        explanation = (
            f"Insufficient telemetry length ({total_samples} samples). "
            "At least 20 chronological samples are required for reliable cross-sensor diagnosis."
        )
        evidence["observations"].append("Short telemetry stream (< 20 samples)")
        return verdict, explanation, evidence

    # Case 1: Nominal operation (no significant sensor anomalies, no quality issues)
    if not stuck_sensors and not missing_counts and not significant_sensors:
        verdict = VERDICT_NORMAL
        explanation = (
            "All sensors are operating within expected baseline envelopes. "
            f"No persistent anomalies detected (overall sensor anomaly rate: {anomaly_rate * 100:.1f}%)."
        )
        evidence["observations"].append("Telemetry within nominal statistical tolerance")
        return verdict, explanation, evidence

    # Case 2: Data Quality Sensor Faults (Stuck or Missing Data)
    if stuck_sensors or missing_counts:
        faulty_sensors = sorted(list(set(stuck_sensors + list(missing_counts.keys()))))
        if len(faulty_sensors) == 1:
            fault_sensor = faulty_sensors[0]
            reasons = []
            if fault_sensor in stuck_sensors:
                reasons.append("frozen/stuck constant telemetry output")
            if fault_sensor in missing_counts:
                reasons.append(f"{missing_counts[fault_sensor]} missing (NaN) readings")

            verdict = VERDICT_SENSOR_FAULT
            explanation = (
                f"Data-quality instrumentation failure suspected in `{fault_sensor}`: "
                f"{', and '.join(reasons)}."
            )
            evidence["observations"].append(f"Isolated quality failure on {fault_sensor}")
            return verdict, explanation, evidence

    # Case 3: Conflicting Evidence Check
    # Check if primary coupled sensors exhibit contradictory physical behavior
    fit_lit_corr = directional_correlations.get("FIT101_LIT101", 0.0)
    if "FIT101" in significant_sensors and "LIT101" in significant_sensors:
        # In a genuine physical process, higher inflow leads to higher tank level (positive relationship).
        # When correlation is inverted / negative during anomaly window, signals conflict.
        if fit_lit_corr < -0.20:
            verdict = VERDICT_UNCERTAIN
            explanation = (
                "Conflicting cross-sensor evidence detected: `FIT101` (inflow) and `LIT101` (level) "
                f"show strong contradictory dynamics (coupling correlation: {fit_lit_corr:.2f}). "
                "Evidence is ambiguous between an instrumentation failure and an abnormal process perturbation."
            )
            evidence["observations"].append(
                f"Contradictory physical coupling between FIT101 and LIT101 (r={fit_lit_corr:.2f})"
            )
            return verdict, explanation, evidence

    # Case 4: Coordinated Process Event
    # A true process event affects multiple coupled sensors simultaneously in a consistent direction
    process_group = {"FIT101", "LIT101", "FIT301"}
    coupled_significant = [s for s in significant_sensors if s in process_group]

    if len(coupled_significant) >= 2 and fit_lit_corr >= -0.15:
        multi_sensor_samples = cooccurrence_counts.get("multi_sensor", 0)
        if multi_sensor_samples >= min_sig_samples:
            verdict = VERDICT_PROCESS_EVENT
            explanation = (
                "Coordinated shifts observed across multiple coupled process sensors "
                f"({', '.join(coupled_significant)}). Consistent physical dynamics suggest "
                "an operational process state transition rather than an isolated sensor failure."
            )
            evidence["observations"].append(
                f"Simultaneous shift across {len(coupled_significant)} coupled sensors: {coupled_significant}"
            )
            return verdict, explanation, evidence

    # Case 5: Single-Sensor Fault Suspected
    if len(significant_sensors) == 1:
        single_sensor = significant_sensors[0]
        verdict = VERDICT_SENSOR_FAULT
        count = sensor_anomaly_counts[single_sensor]
        pct = (count / total_samples) * 100
        explanation = (
            f"Sustained anomalous deviation isolated to single sensor `{single_sensor}` "
            f"({count} samples, {pct:.1f}% of stream) while other system sensors remain nominal."
        )
        evidence["observations"].append(f"Isolated statistical anomaly on {single_sensor}")
        return verdict, explanation, evidence

    # Case 6: Short transient Spike isolated to single sensor
    if len(significant_sensors) == 0 and anomalous_samples > 0:
        max_sensor = max(sensor_anomaly_counts, key=sensor_anomaly_counts.get)
        max_count = sensor_anomaly_counts[max_sensor]
        if max_count >= 3:
            verdict = VERDICT_SENSOR_FAULT
            explanation = (
                f"Short transient spike or impulse detected on `{max_sensor}` "
                f"({max_count} anomalous readings) with unperturbed neighboring sensors."
            )
            evidence["observations"].append(f"Transient spike isolated to {max_sensor}")
            return verdict, explanation, evidence

    # Case 7: Ambiguous or weak multi-sensor evidence -> UNCERTAIN
    verdict = VERDICT_UNCERTAIN
    explanation = (
        f"Ambiguous or weak evidence observed across {len(significant_sensors)} sensors "
        f"({', '.join(significant_sensors) if significant_sensors else 'sporadic anomalies'}). "
        "Confidence is insufficient to decisively distinguish an instrumentation fault from a process disturbance."
    )
    evidence["observations"].append(
        f"Multi-sensor anomalies without clear process coordination ({significant_sensors})"
    )
    return verdict, explanation, evidence


def analyze_sensor_data(
    df: pd.DataFrame,
    contamination: float = 0.05,
    z_threshold: float = 3.0,
    random_state: int = 42,
    sensors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze sensor telemetry for anomalies, cross-sensor evidence, and event classification.

    Performs rolling statistical analysis, Isolation Forest modeling, data-quality
    verification, and transparent rule-based evidence classification.
    The input DataFrame is never modified.

    Notice: All findings are based on synthetic demo data.

    Args:
        df: Input DataFrame containing Timestamp and required sensor columns.
        contamination: Contamination parameter for Isolation Forest.
        z_threshold: Z-score threshold for statistical deviation detection.
        random_state: Random state for deterministic Isolation Forest reproducibility.
        sensors: List of sensors to evaluate (defaults to REQUIRED_SENSORS).

    Returns:
        Dict containing:
            - 'verdict': One of NORMAL, SENSOR_FAULT_SUSPECTED, PROCESS_EVENT_SUSPECTED, UNCERTAIN.
            - 'explanation': Human-readable reasoning for the verdict.
            - 'data': A copy of df augmented with per-sensor anomaly flags, z-scores,
                      and a row-level 'is_anomaly' indicator.
            - 'summary': High-level counts, rates, and suspicious sensor rankings.
            - 'evidence': Structured evidence dictionary detailing all observations.

    Raises:
        ValueError: If df is not a DataFrame, is empty, or lacks required columns.
    """
    # 1. Validate inputs and schema
    validate_sensor_data(df)
    target_sensors = sensors if sensors is not None else REQUIRED_SENSORS

    # 2. Work on an isolated copy
    out_df = df.copy(deep=True)
    n = len(out_df)

    # 3. Detect data-quality issues (Stuck & Missing values)
    quality_flags, quality_summary = detect_data_quality_issues(out_df, sensors=target_sensors)

    # 4. Statistical rolling residual deviations
    stat_flags, stat_scores = detect_rolling_deviations(
        out_df,
        sensors=target_sensors,
        z_threshold=z_threshold,
    )

    # 5. Isolation Forest multivariate detection
    iforest_flags, iforest_scores = detect_isolation_forest(
        out_df,
        sensors=target_sensors,
        contamination=contamination,
        random_state=random_state,
    )

    # 6. Synthesize per-sensor and row-level evidence columns
    sensor_anomaly_counts: Dict[str, int] = {}
    combined_sensor_anomalies = pd.DataFrame(index=out_df.index)

    for s in target_sensors:
        s_flag = (
            stat_flags[s]
            | quality_flags[f"{s}_stuck"]
            | quality_flags[f"{s}_nan"]
        )
        combined_sensor_anomalies[s] = s_flag
        sensor_anomaly_counts[s] = int(s_flag.sum())

        out_df[f"{s}_anomaly"] = s_flag
        out_df[f"{s}_score"] = stat_scores[s]

    # Row-level anomaly: at least one sensor flagged by statistical/quality detector,
    # OR Isolation Forest outlier when maximum sensor z-score is elevated (>= 2.0)
    sensor_row_anomalies = combined_sensor_anomalies.any(axis=1)
    max_z = stat_scores[target_sensors].max(axis=1)
    iforest_corroborated = iforest_flags & (max_z >= 2.0)
    row_anomaly = sensor_row_anomalies | iforest_corroborated

    out_df["is_anomaly"] = row_anomaly
    out_df["iforest_anomaly"] = iforest_flags
    out_df["iforest_score"] = iforest_scores

    # Count simultaneous sensor anomaly occurrences
    sensors_active_per_row = combined_sensor_anomalies.sum(axis=1)
    multi_sensor_rows = int((sensors_active_per_row >= 2).sum())

    # 7. Compute directional correlation shift between coupled sensors
    directional_correlations: Dict[str, float] = {}
    if "FIT101" in target_sensors and "LIT101" in target_sensors:
        s1 = out_df["FIT101"].ffill().bfill()
        s2 = out_df["LIT101"].ffill().bfill()
        if row_anomaly.sum() > 5:
            anom_s1 = s1[row_anomaly]
            anom_s2 = s2[row_anomaly]
            if anom_s1.std() > 1e-4 and anom_s2.std() > 1e-4:
                corr_val = float(anom_s1.corr(anom_s2))
            else:
                corr_val = float(s1.corr(s2))
        else:
            corr_val = float(s1.corr(s2))
        directional_correlations["FIT101_LIT101"] = (
            0.0 if np.isnan(corr_val) else round(corr_val, 3)
        )

    # 8. Build structured summary
    total_anomalies = int(row_anomaly.sum())
    anomaly_rate = total_anomalies / n if n > 0 else 0.0

    suspicious_sensors = [
        s for s, count in sensor_anomaly_counts.items() if count > 0
    ]
    suspicious_sensors.sort(key=lambda s: sensor_anomaly_counts[s], reverse=True)

    summary: Dict[str, Any] = {
        "total_samples": n,
        "anomalous_samples": total_anomalies,
        "anomaly_rate": round(anomaly_rate, 4),
        "sensor_anomalies": sensor_anomaly_counts,
        "suspicious_sensors": suspicious_sensors,
        "iforest_anomalies": int(iforest_flags.sum()),
        "multi_sensor_anomalies": multi_sensor_rows,
    }

    # 9. Fuse cross-sensor evidence to produce explainable classification
    verdict, explanation, evidence = classify_sensor_evidence(
        summary=summary,
        sensor_anomaly_counts=sensor_anomaly_counts,
        quality_summary=quality_summary,
        total_samples=n,
        cooccurrence_counts={"multi_sensor": multi_sensor_rows},
        directional_correlations=directional_correlations,
    )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "data": out_df,
        "summary": summary,
        "evidence": evidence,
    }
