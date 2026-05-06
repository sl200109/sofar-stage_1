import unittest

from sofar.serve.open6dor_json_utils import (
    normalize_open6dor_joint_result,
    normalize_open6dor_reasoning_result,
)


class Open6DORJsonUtilsTest(unittest.TestCase):
    def test_normalize_valid_dict(self):
        payload = normalize_open6dor_reasoning_result(
            {
                "calculation_process": "ok",
                "target_position": [0.53, 0.11, 0.3],
            }
        )
        self.assertEqual(payload["target_position"], [0.53, 0.11, 0.3])
        self.assertFalse(payload["json_repair_failed"])

    def test_normalize_invalid_target_position_raises_when_not_degraded(self):
        with self.assertRaises(ValueError):
            normalize_open6dor_reasoning_result(
                {
                    "calculation_process": "bad schema",
                    "target_position": [0.1, 0.2],
                }
            )

    def test_degraded_with_valid_fallback_position(self):
        payload = normalize_open6dor_reasoning_result(
            None,
            fallback_position=[0.1, 0.2, 0.3],
            raw_output_text="not valid json",
            json_repair_failed=True,
            degraded_reason="No valid Open6DOR reasoning JSON found in model output",
        )
        self.assertTrue(payload["json_repair_failed"])
        self.assertEqual(payload["target_position"], [0.1, 0.2, 0.3])
        self.assertEqual(payload["degraded_position_source"], "fallback_position")
        self.assertIn("degraded fallback", payload["calculation_process"])

    def test_degraded_with_none_fallback_position_returns_zero_fallback(self):
        payload = normalize_open6dor_reasoning_result(
            None,
            fallback_position=None,
            raw_output_text="not valid json",
            json_repair_failed=True,
            degraded_reason="schema mismatch",
        )
        self.assertTrue(payload["json_repair_failed"])
        self.assertEqual(payload["target_position"], [0.0, 0.0, 0.0])
        self.assertEqual(payload["degraded_position_source"], "zero_fallback")
        self.assertEqual(payload["degraded_reason"], "schema mismatch")

    def test_joint_degraded_payload_schema(self):
        payload = normalize_open6dor_joint_result(
            None,
            fallback_position=None,
            raw_output_text="broken joint output",
            json_repair_failed=True,
            degraded_reason="joint parse failed",
            fallback_picked_object="mug",
        )
        self.assertEqual(payload["picked_object"], "mug")
        self.assertEqual(payload["related_objects"], [])
        self.assertEqual(payload["direction_attributes"], [])
        self.assertEqual(payload["target_orientation"], {})
        self.assertEqual(payload["target_position"], [0.0, 0.0, 0.0])
        self.assertTrue(payload["json_repair_failed"])
        self.assertEqual(payload["degraded_position_source"], "zero_fallback")


if __name__ == "__main__":
    unittest.main()
