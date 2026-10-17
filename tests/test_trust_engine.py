"""Unit tests for TrustMesh Dynamic Trust Scoring Engine."""

import unittest
import copy
import pandas as pd

from src.data_pipeline import REQUIRED_SENSORS, generate_synthetic_sensor_data
from src.fault_simulator import inject_fault
from src.detection_engine import analyze_sensor_data
from src.trust_engine import (
    TRUST_HIGH,
    TRUST_LOW,
    TRUST_MODERATE,
    calculate_sensor_trust,
)


class TestTrustEngine(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_sensor_data(n_samples=500, seed=42)
        self.normal_analysis = analyze_sensor_data(self.df)

    def test_scores_remain_within_bounded_range(self):
        """1. Every trust score strictly remains within [0.0, 100.0]."""
        for scenario in [
            "normal operation",
            "gradual drift in one sensor",
            "a short sudden spike",
            "a stuck sensor value",
            "a block of missing measurements",
            "a coordinated process change affecting multiple related sensors",
            "a conflicting-evidence scenario for later uncertainty testing",
        ]:
            mod_df, _ = inject_fault(self.df, scenario=scenario, target_sensor="FIT101")
            analysis = analyze_sensor_data(mod_df)
            trust_res = calculate_sensor_trust(analysis)

            self.assertGreaterEqual(trust_res["overall_trust_score"], 0.0)
            self.assertLessEqual(trust_res["overall_trust_score"], 100.0)

            for s, s_data in trust_res["sensors"].items():
                self.assertGreaterEqual(s_data["trust_score"], 0.0)
                self.assertLessEqual(s_data["trust_score"], 100.0)
                self.assertIn(s_data["trust_level"], [TRUST_HIGH, TRUST_MODERATE, TRUST_LOW])

    def test_normal_synthetic_telemetry_produces_high_trust(self):
        """2. Normal synthetic telemetry produces HIGH trust across all sensors."""
        trust_res = calculate_sensor_trust(self.normal_analysis)
        self.assertFalse(trust_res["has_uncertainty"])
        self.assertGreaterEqual(trust_res["overall_trust_score"], 90.0)

        for s in REQUIRED_SENSORS:
            s_trust = trust_res["sensors"][s]
            self.assertEqual(s_trust["trust_level"], TRUST_HIGH)
            self.assertGreaterEqual(s_trust["trust_score"], 90.0)

    def test_sustained_isolated_drift_lowers_trust(self):
        """3. Sustained isolated drift lowers the affected sensor's trust below baseline."""
        drift_df, _ = inject_fault(self.df, "gradual drift in one sensor", "FIT101")
        drift_analysis = analyze_sensor_data(drift_df)
        drift_trust = calculate_sensor_trust(drift_analysis)

        fit_drift = drift_trust["sensors"]["FIT101"]
        normal_fit = calculate_sensor_trust(self.normal_analysis)["sensors"]["FIT101"]

        self.assertLess(fit_drift["trust_score"], normal_fit["trust_score"])
        self.assertLess(fit_drift["trust_score"], 80.0)
        self.assertIn(fit_drift["trust_level"], [TRUST_MODERATE, TRUST_LOW])

        # Other untouched sensors should remain high trust
        self.assertEqual(drift_trust["sensors"]["LIT101"]["trust_level"], TRUST_HIGH)

    def test_missing_readings_reduce_trust(self):
        """4. Missing readings reduce the relevant sensor's trust."""
        missing_df, _ = inject_fault(self.df, "a block of missing measurements", "LIT101")
        missing_analysis = analyze_sensor_data(missing_df)
        missing_trust = calculate_sensor_trust(missing_analysis)

        lit_trust = missing_trust["sensors"]["LIT101"]
        self.assertLess(lit_trust["trust_score"], 60.0)
        self.assertIn(lit_trust["trust_level"], [TRUST_LOW, TRUST_MODERATE])
        self.assertGreater(lit_trust["evidence"]["missing_count"], 0)

    def test_stuck_readings_reduce_trust(self):
        """5. Stuck readings reduce the relevant sensor's trust."""
        stuck_df, _ = inject_fault(self.df, "a stuck sensor value", "AIT202")
        stuck_analysis = analyze_sensor_data(stuck_df)
        stuck_trust = calculate_sensor_trust(stuck_analysis)

        ait_trust = stuck_trust["sensors"]["AIT202"]
        self.assertTrue(ait_trust["evidence"]["is_stuck"])
        self.assertLess(ait_trust["trust_score"], 60.0)
        self.assertEqual(ait_trust["trust_level"], TRUST_LOW)

    def test_short_isolated_spike_causes_smaller_penalty_than_drift(self):
        """6. A short isolated spike causes a smaller penalty than sustained degradation/drift."""
        spike_df, _ = inject_fault(self.df, "a short sudden spike", "FIT101")
        drift_df, _ = inject_fault(self.df, "gradual drift in one sensor", "FIT101")

        spike_trust = calculate_sensor_trust(analyze_sensor_data(spike_df))["sensors"]["FIT101"]
        drift_trust = calculate_sensor_trust(analyze_sensor_data(drift_df))["sensors"]["FIT101"]

        # Spike trust must be strictly greater than sustained drift trust
        self.assertGreater(spike_trust["trust_score"], drift_trust["trust_score"])
        self.assertTrue(spike_trust["evidence"]["is_spike"])

    def test_coordinated_process_changes_do_not_mark_sensors_low(self):
        """7. Coordinated process changes do not automatically mark participating sensors as LOW trust."""
        coord_df, _ = inject_fault(
            self.df, "a coordinated process change affecting multiple related sensors"
        )
        coord_trust = calculate_sensor_trust(analyze_sensor_data(coord_df))

        for s in ["FIT101", "LIT101", "FIT301"]:
            s_trust = coord_trust["sensors"][s]
            # Must remain HIGH or MODERATE; not penalized as sensor failure
            self.assertNotEqual(s_trust["trust_level"], TRUST_LOW)
            self.assertGreaterEqual(s_trust["trust_score"], 80.0)

    def test_conflicting_evidence_produces_uncertainty_indicator(self):
        """8. Conflicting evidence produces an uncertainty indicator."""
        conflict_df, _ = inject_fault(
            self.df, "a conflicting-evidence scenario for later uncertainty testing", "FIT101"
        )
        conflict_trust = calculate_sensor_trust(analyze_sensor_data(conflict_df))

        self.assertTrue(conflict_trust["has_uncertainty"])
        self.assertNotEqual(conflict_trust["uncertainty_note"], "")
        self.assertIn("contradictory", conflict_trust["system_status"].lower())

    def test_explanations_and_recommendations_grounded_in_evidence(self):
        """9. Each sensor score provides an explanation and recommendation grounded in evidence."""
        drift_df, _ = inject_fault(self.df, "gradual drift in one sensor", "FIT101")
        trust_res = calculate_sensor_trust(analyze_sensor_data(drift_df))

        for s, s_data in trust_res["sensors"].items():
            self.assertIsInstance(s_data["reasons"], list)
            self.assertGreater(len(s_data["reasons"]), 0)
            self.assertIsInstance(s_data["recommended_action"], str)
            self.assertGreater(len(s_data["recommended_action"]), 10)
            # Ensure safety guideline is strictly preserved
            if "safety alarm" in s_data["recommended_action"].lower():
                self.assertIn("do not suppress", s_data["recommended_action"].lower())

    def test_reproducibility(self):
        """10. Identical inputs produce identical trust outputs."""
        res1 = calculate_sensor_trust(self.normal_analysis)
        res2 = calculate_sensor_trust(self.normal_analysis)

        self.assertEqual(res1["overall_trust_score"], res2["overall_trust_score"])
        self.assertEqual(res1["has_uncertainty"], res2["has_uncertainty"])
        self.assertEqual(res1["system_status"], res2["system_status"])

        for s in REQUIRED_SENSORS:
            self.assertEqual(res1["sensors"][s]["trust_score"], res2["sensors"][s]["trust_score"])
            self.assertEqual(res1["sensors"][s]["trust_level"], res2["sensors"][s]["trust_level"])

    def test_invalid_or_incomplete_inputs_raise_errors(self):
        """11. Invalid or incomplete analysis results raise clear ValueErrors."""
        with self.assertRaises(ValueError):
            calculate_sensor_trust("not_a_dict")  # type: ignore

        with self.assertRaises(ValueError):
            calculate_sensor_trust({"incomplete": "dict"})

        with self.assertRaises(ValueError):
            calculate_sensor_trust({
                "verdict": "NORMAL",
                "summary": {},
                "evidence": {},
                "data": "not_a_df",
            })

    def test_input_analysis_result_not_mutated(self):
        """12. Original analysis result and telemetry DataFrame are not mutated."""
        copy_analysis = copy.deepcopy(self.normal_analysis)
        _ = calculate_sensor_trust(self.normal_analysis)

        self.assertEqual(self.normal_analysis["verdict"], copy_analysis["verdict"])
        self.assertEqual(self.normal_analysis["summary"], copy_analysis["summary"])
        pd.testing.assert_frame_equal(self.normal_analysis["data"], copy_analysis["data"])


if __name__ == "__main__":
    unittest.main()
