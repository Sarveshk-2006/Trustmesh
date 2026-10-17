"""Comprehensive test suite for TrustMesh sensor data pipeline and fault simulator."""

import io
import unittest
import numpy as np
import pandas as pd

from src.data_pipeline import (
    REQUIRED_COLUMNS,
    REQUIRED_SENSORS,
    generate_synthetic_sensor_data,
    load_sensor_csv,
    validate_sensor_data,
)
from src.fault_simulator import (
    AVAILABLE_SCENARIOS,
    inject_fault,
    normalize_scenario_name,
)


class TestSensorDataPipeline(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_sensor_data(n_samples=500, seed=42)

    def test_expected_columns_and_sample_count(self):
        """Test default sample count of 500 and presence of all required columns."""
        self.assertEqual(len(self.df), 500)
        self.assertListEqual(list(self.df.columns), REQUIRED_COLUMNS)
        for sensor in REQUIRED_SENSORS:
            self.assertIn(sensor, self.df.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(self.df[sensor]))

    def test_valid_and_monotonically_increasing_timestamps(self):
        """Test that timestamps are valid datetimes and strictly increasing."""
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df["Timestamp"]))
        self.assertTrue(self.df["Timestamp"].is_monotonic_increasing)
        self.assertEqual(self.df["Timestamp"].nunique(), 500)

    def test_reproducibility_with_same_seed(self):
        """Test that generation is strictly reproducible with the same seed."""
        df1 = generate_synthetic_sensor_data(n_samples=500, seed=42)
        df2 = generate_synthetic_sensor_data(n_samples=500, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

        # Different seed produces different readings
        df3 = generate_synthetic_sensor_data(n_samples=500, seed=99)
        self.assertFalse(df1["FIT101"].equals(df3["FIT101"]))

    def test_csv_validation_success_and_failure(self):
        """Test CSV loading success and clear failures for missing/invalid schema."""
        # 1. Successful CSV load
        csv_buffer = io.StringIO(self.df.to_csv(index=False))
        loaded_df = load_sensor_csv(csv_buffer)
        self.assertEqual(len(loaded_df), 500)
        self.assertListEqual(list(loaded_df.columns), REQUIRED_COLUMNS)

        # 2. Missing required columns
        bad_df = self.df.drop(columns=["FIT101"])
        bad_buffer = io.StringIO(bad_df.to_csv(index=False))
        with self.assertRaises(ValueError) as ctx:
            load_sensor_csv(bad_buffer)
        self.assertIn("Missing required sensor columns", str(ctx.exception))
        self.assertIn("FIT101", str(ctx.exception))

        # 3. Missing Timestamp column
        no_time_df = self.df.drop(columns=["Timestamp"])
        with self.assertRaises(ValueError) as ctx:
            validate_sensor_data(no_time_df)
        self.assertIn("Missing required sensor columns", str(ctx.exception))

        # 4. Non-monotonic timestamps
        scrambled_df = self.df.copy()
        scrambled_df.loc[10, "Timestamp"] = scrambled_df.loc[0, "Timestamp"]
        with self.assertRaises(ValueError) as ctx:
            validate_sensor_data(scrambled_df)
        self.assertIn("monotonically increasing", str(ctx.exception))

        # 5. Non-numeric sensor column
        non_numeric_df = self.df.copy()
        non_numeric_df["LIT101"] = "invalid_sensor_text"
        with self.assertRaises(ValueError) as ctx:
            validate_sensor_data(non_numeric_df)
        self.assertIn("must contain numeric data", str(ctx.exception))

        # 6. Empty DataFrame
        with self.assertRaises(ValueError) as ctx:
            validate_sensor_data(pd.DataFrame())
        self.assertIn("empty", str(ctx.exception))


class TestFaultSimulator(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_sensor_data(n_samples=500, seed=42)

    def test_original_dataframe_not_mutated(self):
        """Ensure inject_fault returns a copy and leaves the original DataFrame untouched."""
        original_copy = self.df.copy(deep=True)

        for scenario in AVAILABLE_SCENARIOS:
            mod_df, meta = inject_fault(self.df, scenario=scenario, target_sensor="FIT101")
            self.assertIsNot(mod_df, self.df)
            pd.testing.assert_frame_equal(self.df, original_copy)

    def test_missing_data_introduces_nans(self):
        """Verify that missing measurements injection specifically introduces NaNs."""
        mod_df, meta = inject_fault(
            self.df,
            scenario="a block of missing measurements",
            target_sensor="LIT101",
        )
        start, end = meta["start_idx"], meta["end_idx"]
        self.assertGreater(end, start)
        self.assertTrue(mod_df.loc[start : end - 1, "LIT101"].isna().all())

        # No NaNs outside the target window
        self.assertFalse(mod_df.loc[: start - 1, "LIT101"].isna().any())
        self.assertFalse(mod_df.loc[end:, "LIT101"].isna().any())

        # Other sensors have zero NaNs
        for s in REQUIRED_SENSORS:
            if s != "LIT101":
                self.assertEqual(int(mod_df[s].isna().sum()), 0)

    def test_stuck_value_creates_constant_segment(self):
        """Verify that stuck sensor injection produces a strictly constant segment."""
        mod_df, meta = inject_fault(
            self.df,
            scenario="a stuck sensor value",
            target_sensor="AIT202",
        )
        start, end = meta["start_idx"], meta["end_idx"]
        segment = mod_df.loc[start : end - 1, "AIT202"]

        # Exactly 1 unique value across the injected segment
        self.assertEqual(segment.nunique(), 1)
        self.assertEqual(segment.iloc[0], self.df["AIT202"].iloc[start])

    def test_coordinated_process_changes_affect_multiple_sensors(self):
        """Verify that coordinated process change alters multiple coupled sensors."""
        mod_df, meta = inject_fault(
            self.df,
            scenario="a coordinated process change affecting multiple related sensors",
        )
        affected = meta["affected_sensors"]
        self.assertIn("FIT101", affected)
        self.assertIn("LIT101", affected)
        self.assertIn("FIT301", affected)
        self.assertGreaterEqual(len(affected), 3)

        start, end = meta["start_idx"], meta["end_idx"]

        # Check all affected sensors changed in window
        for sensor in affected:
            self.assertFalse(
                mod_df.loc[start : end - 1, sensor].equals(
                    self.df.loc[start : end - 1, sensor]
                )
            )

        # Check unaffected sensors remain completely untouched everywhere
        unaffected = [s for s in REQUIRED_SENSORS if s not in affected]
        for sensor in unaffected:
            pd.testing.assert_series_equal(mod_df[sensor], self.df[sensor])

    def test_each_scenario_changes_only_intended_signals_and_window(self):
        """Verify that each scenario only modifies affected sensors within the declared window."""
        for scenario in AVAILABLE_SCENARIOS:
            mod_df, meta = inject_fault(
                self.df,
                scenario=scenario,
                target_sensor="FIT101",
            )
            affected = meta["affected_sensors"]

            if scenario == "normal operation":
                self.assertEqual(len(affected), 0)
                pd.testing.assert_frame_equal(mod_df, self.df)
                continue

            start, end = meta["start_idx"], meta["end_idx"]
            self.assertIsNotNone(start)
            self.assertIsNotNone(end)

            # 1. Unaffected sensors must be completely identical across the entire stream
            unaffected = [s for s in REQUIRED_SENSORS if s not in affected]
            for sensor in unaffected:
                pd.testing.assert_series_equal(
                    mod_df[sensor],
                    self.df[sensor],
                    obj=f"{scenario}:{sensor}",
                )

            # 2. Affected sensors must be identical outside the window
            for sensor in affected:
                if start > 0:
                    pd.testing.assert_series_equal(
                        mod_df.loc[: start - 1, sensor],
                        self.df.loc[: start - 1, sensor],
                        obj=f"{scenario}:{sensor}:pre-window",
                    )
                if end < len(self.df):
                    pd.testing.assert_series_equal(
                        mod_df.loc[end:, sensor],
                        self.df.loc[end:, sensor],
                        obj=f"{scenario}:{sensor}:post-window",
                    )

    def test_invalid_scenario_and_sensor_raise_value_errors(self):
        """Ensure invalid scenario or target sensor arguments raise descriptive ValueErrors."""
        with self.assertRaises(ValueError) as ctx1:
            inject_fault(self.df, scenario="non_existent_fault")
        self.assertIn("Unrecognized scenario", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            inject_fault(self.df, scenario="drift", target_sensor="UNKNOWN_SENSOR")
        self.assertIn("Invalid target sensor", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
