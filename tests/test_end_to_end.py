"""End-to-end validation and robustness test suite for TrustMesh.

Validates the complete pipeline:
1. Synthetic Generation / CSV Ingestion
2. Fault Simulation
3. Detection Engine & Evidence Fusion
4. Trust Scoring & Actionable Recommendations
Across all seven operational scenarios and edge cases.
"""

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
from src.fault_simulator import AVAILABLE_SCENARIOS, inject_fault
from src.detection_engine import (
    VERDICT_NORMAL,
    VERDICT_PROCESS_EVENT,
    VERDICT_SENSOR_FAULT,
    VERDICT_UNCERTAIN,
    analyze_sensor_data,
)
from src.trust_engine import (
    TRUST_HIGH,
    TRUST_LOW,
    TRUST_MODERATE,
    calculate_sensor_trust,
)


class TestEndToEndScenarios(unittest.TestCase):
    """Scenario-by-scenario end-to-end integration tests."""

    def setUp(self):
        self.base_df = generate_synthetic_sensor_data(n_samples=500, seed=42)

    def test_scenario_normal_operation(self):
        """1. End-to-end Normal Operation: no widespread anomalies, high trust."""
        sim_df, meta = inject_fault(self.base_df, "normal operation", "FIT101")
        analysis = analyze_sensor_data(sim_df)
        trust = calculate_sensor_trust(analysis)

        self.assertEqual(analysis["verdict"], VERDICT_NORMAL)
        self.assertEqual(len(meta["affected_sensors"]), 0)
        self.assertFalse(trust["has_uncertainty"])
        self.assertGreaterEqual(trust["overall_trust_score"], 90.0)

        for s in REQUIRED_SENSORS:
            self.assertEqual(trust["sensors"][s]["trust_level"], TRUST_HIGH)
            self.assertGreaterEqual(trust["sensors"][s]["trust_score"], 90.0)

    def test_scenario_gradual_drift(self):
        """2. End-to-end Gradual Drift: target sensor degradation evidence & lowered trust."""
        target = "FIT101"
        sim_df, meta = inject_fault(self.base_df, "gradual drift in one sensor", target)
        analysis = analyze_sensor_data(sim_df)
        trust = calculate_sensor_trust(analysis)

        self.assertIn(target, analysis["summary"]["suspicious_sensors"])
        self.assertGreater(analysis["summary"]["sensor_anomalies"][target], 0)

        target_trust = trust["sensors"][target]
        normal_trust = calculate_sensor_trust(analyze_sensor_data(self.base_df))["sensors"][target]

        self.assertLess(target_trust["trust_score"], normal_trust["trust_score"])
        self.assertLess(target_trust["trust_score"], 80.0)
        self.assertIn(target_trust["trust_level"], [TRUST_MODERATE, TRUST_LOW])

        # Other sensors maintain high trust
        for s in REQUIRED_SENSORS:
            if s != target:
                self.assertEqual(trust["sensors"][s]["trust_level"], TRUST_HIGH)

    def test_scenario_short_sudden_spike(self):
        """3. End-to-end Spike: detected, with penalty less severe than sustained drift."""
        target = "FIT101"
        spike_df, _ = inject_fault(self.base_df, "a short sudden spike", target)
        drift_df, _ = inject_fault(self.base_df, "gradual drift in one sensor", target)

        spike_analysis = analyze_sensor_data(spike_df)
        drift_analysis = analyze_sensor_data(drift_df)

        spike_trust = calculate_sensor_trust(spike_analysis)["sensors"][target]
        drift_trust = calculate_sensor_trust(drift_analysis)["sensors"][target]

        self.assertIn(target, spike_analysis["summary"]["suspicious_sensors"])
        self.assertTrue(spike_trust["evidence"]["is_spike"])
        self.assertGreater(spike_trust["trust_score"], drift_trust["trust_score"])

    def test_scenario_stuck_sensor_value(self):
        """4. End-to-end Stuck Sensor: identified as stuck data-quality issue with low trust."""
        target = "AIT202"
        sim_df, meta = inject_fault(self.base_df, "a stuck sensor value", target)
        analysis = analyze_sensor_data(sim_df)
        trust = calculate_sensor_trust(analysis)

        self.assertIn(target, analysis["evidence"]["stuck_sensors"])
        target_trust = trust["sensors"][target]
        self.assertTrue(target_trust["evidence"]["is_stuck"])
        self.assertLess(target_trust["trust_score"], 60.0)
        self.assertEqual(target_trust["trust_level"], TRUST_LOW)

    def test_scenario_block_of_missing_measurements(self):
        """5. End-to-end Missing Measurements: identified as NaN loss with reduced trust."""
        target = "LIT301"
        sim_df, meta = inject_fault(self.base_df, "a block of missing measurements", target)
        analysis = analyze_sensor_data(sim_df)
        trust = calculate_sensor_trust(analysis)

        self.assertIn(target, analysis["evidence"]["missing_sensors"])
        target_trust = trust["sensors"][target]
        self.assertGreater(target_trust["evidence"]["missing_count"], 0)
        self.assertLess(target_trust["trust_score"], 60.0)
        self.assertEqual(target_trust["trust_level"], TRUST_LOW)

    def test_scenario_coordinated_process_change(self):
        """6. End-to-end Coordinated Process Change: multi-sensor coupling, trust remains intact."""
        sim_df, meta = inject_fault(
            self.base_df, "a coordinated process change affecting multiple related sensors"
        )
        analysis = analyze_sensor_data(sim_df)
        trust = calculate_sensor_trust(analysis)

        self.assertEqual(analysis["verdict"], VERDICT_PROCESS_EVENT)
        self.assertGreaterEqual(len(analysis["evidence"]["significant_sensors"]), 2)

        # In a true process event, sensors are not penalized as failing hardware
        for s in ["FIT101", "LIT101", "FIT301"]:
            s_trust = trust["sensors"][s]
            self.assertNotEqual(s_trust["trust_level"], TRUST_LOW)
            self.assertGreaterEqual(s_trust["trust_score"], 80.0)

    def test_scenario_conflicting_evidence(self):
        """7. End-to-end Conflicting Evidence: preserves uncertainty indication."""
        sim_df, meta = inject_fault(
            self.base_df,
            "a conflicting-evidence scenario for later uncertainty testing",
            "FIT101",
        )
        analysis = analyze_sensor_data(sim_df)
        trust = calculate_sensor_trust(analysis)

        self.assertEqual(analysis["verdict"], VERDICT_UNCERTAIN)
        self.assertTrue(trust["has_uncertainty"])
        self.assertNotEqual(trust["uncertainty_note"], "")
        self.assertIn("contradictory", trust["system_status"].lower())


class TestRobustnessAndEdgeCases(unittest.TestCase):
    """Robustness against malformed, short, or edge-case inputs."""

    def setUp(self):
        self.base_df = generate_synthetic_sensor_data(n_samples=500, seed=42)

    def test_csv_with_missing_values(self):
        """CSV with interspersed missing values is loaded and handled cleanly."""
        csv_df = self.base_df.copy()
        csv_df.loc[10:15, "FIT101"] = np.nan
        csv_df.loc[100:105, "LIT101"] = np.nan

        buffer = io.StringIO(csv_df.to_csv(index=False))
        loaded_df = load_sensor_csv(buffer)

        self.assertEqual(len(loaded_df), 500)
        self.assertEqual(int(loaded_df["FIT101"].isna().sum()), 6)

        # Detection and trust execute without unhandled exceptions
        analysis = analyze_sensor_data(loaded_df)
        trust = calculate_sensor_trust(analysis)

        self.assertIn("data", analysis)
        self.assertIn("sensors", trust)
        self.assertGreater(trust["sensors"]["FIT101"]["evidence"]["missing_count"], 0)

    def test_short_time_series_execution(self):
        """Short time-series (< 20 samples) executes safely without crashing."""
        short_df = self.base_df.iloc[:15].copy()
        analysis = analyze_sensor_data(short_df)
        trust = calculate_sensor_trust(analysis)

        self.assertIn(analysis["verdict"], [VERDICT_NORMAL, VERDICT_UNCERTAIN])
        self.assertIn("sensors", trust)
        for s in REQUIRED_SENSORS:
            self.assertGreaterEqual(trust["sensors"][s]["trust_score"], 0.0)
            self.assertLessEqual(trust["sensors"][s]["trust_score"], 100.0)

    def test_invalid_or_nonnumeric_sensor_data(self):
        """Non-numeric or malformed sensor data raises clear ValueError."""
        bad_df = self.base_df.copy()
        bad_df["AIT202"] = "invalid_string_reading"

        buffer = io.StringIO(bad_df.to_csv(index=False))
        with self.assertRaises(ValueError) as ctx:
            load_sensor_csv(buffer)
        self.assertIn("must contain numeric data", str(ctx.exception))

    def test_repeated_execution_determinism(self):
        """Repeated execution with identical inputs produces identical outputs."""
        sim_df, _ = inject_fault(self.base_df, "gradual drift in one sensor", "FIT101")

        run1 = calculate_sensor_trust(analyze_sensor_data(sim_df, random_state=42))
        run2 = calculate_sensor_trust(analyze_sensor_data(sim_df, random_state=42))

        self.assertEqual(run1["overall_trust_score"], run2["overall_trust_score"])
        self.assertEqual(run1["has_uncertainty"], run2["has_uncertainty"])
        for s in REQUIRED_SENSORS:
            self.assertEqual(run1["sensors"][s]["trust_score"], run2["sensors"][s]["trust_score"])
            self.assertEqual(run1["sensors"][s]["trust_level"], run2["sensors"][s]["trust_level"])
            self.assertEqual(run1["sensors"][s]["reasons"], run2["sensors"][s]["reasons"])

    def test_input_dataframe_immutability(self):
        """Input DataFrames remain unmodified through complete pipeline execution."""
        original_copy = self.base_df.copy(deep=True)

        sim_df, _ = inject_fault(self.base_df, "a short sudden spike", "FIT101")
        sim_copy = sim_df.copy(deep=True)

        analysis = analyze_sensor_data(sim_df)
        _ = calculate_sensor_trust(analysis)

        pd.testing.assert_frame_equal(self.base_df, original_copy)
        pd.testing.assert_frame_equal(sim_df, sim_copy)

    def test_all_trust_scores_bounded_between_zero_and_hundred(self):
        """Trust scores never exceed [0.0, 100.0] even under extreme synthetic faults."""
        for scenario in AVAILABLE_SCENARIOS:
            for target in ["FIT101", "LIT101", "AIT401"]:
                sim_df, _ = inject_fault(self.base_df, scenario, target)
                analysis = analyze_sensor_data(sim_df)
                trust = calculate_sensor_trust(analysis)

                self.assertGreaterEqual(trust["overall_trust_score"], 0.0)
                self.assertLessEqual(trust["overall_trust_score"], 100.0)
                for s in REQUIRED_SENSORS:
                    sc = trust["sensors"][s]["trust_score"]
                    self.assertGreaterEqual(sc, 0.0)
                    self.assertLessEqual(sc, 100.0)


if __name__ == "__main__":
    unittest.main()
