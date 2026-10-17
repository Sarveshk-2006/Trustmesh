"""Unit tests for TrustMesh Anomaly Detection & Evidence Fusion Engine."""

import unittest
import numpy as np
import pandas as pd

from src.data_pipeline import (
    REQUIRED_COLUMNS,
    REQUIRED_SENSORS,
    generate_synthetic_sensor_data,
)
from src.fault_simulator import inject_fault
from src.detection_engine import (
    VERDICT_NORMAL,
    VERDICT_PROCESS_EVENT,
    VERDICT_SENSOR_FAULT,
    VERDICT_UNCERTAIN,
    analyze_sensor_data,
    detect_data_quality_issues,
    detect_isolation_forest,
    detect_rolling_deviations,
)


class TestDetectionEngine(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_sensor_data(n_samples=500, seed=42)

    def test_normal_synthetic_data_behavior(self):
        """1. Normal synthetic data does not produce persistent high-severity anomalies."""
        result = analyze_sensor_data(self.df)
        self.assertEqual(result["verdict"], VERDICT_NORMAL)
        self.assertLess(result["summary"]["anomaly_rate"], 0.03)
        self.assertEqual(len(result["evidence"]["stuck_sensors"]), 0)
        self.assertEqual(len(result["evidence"]["missing_sensors"]), 0)

    def test_sudden_spike_detected(self):
        """2. A sudden spike is detected and isolated to the target sensor."""
        spike_df, meta = inject_fault(
            self.df, scenario="a short sudden spike", target_sensor="FIT101"
        )
        result = analyze_sensor_data(spike_df)

        self.assertIn(result["verdict"], [VERDICT_SENSOR_FAULT, VERDICT_UNCERTAIN])
        # Anomaly flags should be present on FIT101 in the injected window
        start, end = meta["start_idx"], meta["end_idx"]
        flagged_in_window = result["data"].loc[start : end - 1, "FIT101_anomaly"]
        self.assertTrue(flagged_in_window.any())
        self.assertIn("FIT101", result["summary"]["suspicious_sensors"])

    def test_gradual_drift_detected(self):
        """3. A gradual drift is detected."""
        drift_df, meta = inject_fault(
            self.df, scenario="gradual drift in one sensor", target_sensor="LIT101"
        )
        result = analyze_sensor_data(drift_df)

        self.assertIn(result["verdict"], [VERDICT_SENSOR_FAULT, VERDICT_UNCERTAIN])
        self.assertIn("LIT101", result["summary"]["suspicious_sensors"])
        # Check that anomalies were flagged on LIT101
        self.assertGreater(result["summary"]["sensor_anomalies"]["LIT101"], 10)

    def test_stuck_and_missing_surfaced(self):
        """4. Stuck sensor and missing data are surfaced as anomalous/quality issues."""
        # 4a. Stuck sensor
        stuck_df, _ = inject_fault(
            self.df, scenario="a stuck sensor value", target_sensor="AIT202"
        )
        stuck_result = analyze_sensor_data(stuck_df)
        self.assertEqual(stuck_result["verdict"], VERDICT_SENSOR_FAULT)
        self.assertIn("AIT202", stuck_result["evidence"]["stuck_sensors"])

        # 4b. Missing data
        missing_df, _ = inject_fault(
            self.df, scenario="a block of missing measurements", target_sensor="FIT301"
        )
        missing_result = analyze_sensor_data(missing_df)
        self.assertEqual(missing_result["verdict"], VERDICT_SENSOR_FAULT)
        self.assertIn("FIT301", missing_result["evidence"]["missing_sensors"])

    def test_coordinated_process_change_not_single_sensor_fault(self):
        """5. Coordinated process change is not automatically labelled a single-sensor fault."""
        coord_df, meta = inject_fault(
            self.df, scenario="a coordinated process change affecting multiple related sensors"
        )
        result = analyze_sensor_data(coord_df)

        self.assertNotEqual(result["verdict"], VERDICT_SENSOR_FAULT)
        self.assertEqual(result["verdict"], VERDICT_PROCESS_EVENT)
        self.assertGreaterEqual(len(result["evidence"]["significant_sensors"]), 2)

    def test_conflicting_evidence_yields_uncertain(self):
        """6. Conflicting evidence yields UNCERTAIN verdict."""
        conflict_df, _ = inject_fault(
            self.df,
            scenario="a conflicting-evidence scenario for later uncertainty testing",
            target_sensor="FIT101",
        )
        result = analyze_sensor_data(conflict_df)
        self.assertEqual(result["verdict"], VERDICT_UNCERTAIN)

    def test_reproducibility(self):
        """7. The same input and configuration yield reproducible outputs."""
        res1 = analyze_sensor_data(self.df, random_state=42)
        res2 = analyze_sensor_data(self.df, random_state=42)

        self.assertEqual(res1["verdict"], res2["verdict"])
        self.assertEqual(res1["summary"], res2["summary"])
        pd.testing.assert_frame_equal(res1["data"], res2["data"])

    def test_input_dataframe_not_mutated(self):
        """8. Input DataFrame remains unchanged after analysis."""
        original_copy = self.df.copy(deep=True)
        result = analyze_sensor_data(self.df)

        self.assertIsNot(result["data"], self.df)
        pd.testing.assert_frame_equal(self.df, original_copy)

    def test_short_dataframes_and_missing_values(self):
        """9. Short dataframes and missing values do not crash unexpectedly."""
        # Short dataframe (e.g., 15 samples)
        short_df = self.df.iloc[:15].copy()
        result_short = analyze_sensor_data(short_df)
        self.assertIn(result_short["verdict"], [VERDICT_NORMAL, VERDICT_UNCERTAIN])

        # Dataframe with scattered NaNs
        nan_df = self.df.copy()
        nan_df.loc[5:8, "FIT101"] = np.nan
        nan_df.loc[20:22, "LIT101"] = np.nan
        result_nan = analyze_sensor_data(nan_df)
        self.assertIsNotNone(result_nan["verdict"])
        self.assertIn("data", result_nan)

    def test_invalid_inputs_produce_clear_errors(self):
        """10. Invalid inputs produce clear ValueErrors."""
        # Not a DataFrame
        with self.assertRaises(ValueError):
            analyze_sensor_data("not_a_df")  # type: ignore

        # Empty DataFrame
        with self.assertRaises(ValueError):
            analyze_sensor_data(pd.DataFrame())

        # Missing required sensor column
        missing_col_df = self.df.drop(columns=["FIT101"])
        with self.assertRaises(ValueError):
            analyze_sensor_data(missing_col_df)


if __name__ == "__main__":
    unittest.main()
