import unittest

from sofar.serve.semantic_orientation_agent import verify_open6dor_agent_outcome


def _decision(mode):
    return {
        "decision": "use_stage5_conditional_verify" if mode == "lying_flat" else "use_stage5_direct",
        "decision_reason": "test",
        "selected_execution_mode": "stage5_direct" if mode != "lying_flat" else "stage5_conditional_verify",
    }


def _prediction(vector, target_axis):
    return {
        "direction_vector": vector,
        "target_orientation": {"semantic_axis": target_axis},
    }


class Open6DORSemanticVerifierTest(unittest.TestCase):
    def test_lying_flat_switches_from_old_z_rule_to_semantic_axis_rule(self):
        result = verify_open6dor_agent_outcome(
            _decision("lying_flat"),
            prediction=_prediction([0.02, -0.01, -0.999], [0.0, 0.0, -1.0]),
            orientation_mode="lying_flat",
            stage4_cache_available=True,
            target_orientation={"semantic_axis": [0.0, 0.0, -1.0]},
            direction_attributes=["larger face"],
        )

        self.assertEqual(result["verification_status"], "accepted")
        self.assertFalse(result["old_rule_pass"])
        self.assertTrue(result["new_rule_pass"])
        self.assertEqual(result["verifier_rule_version"], "rule_v3_semantic_axis")
        self.assertEqual(result["target_axis"], [0.0, 0.0, -1.0])
        self.assertGreater(result["cosine_to_target_axis"], 0.99)
        self.assertIn("semantic_axis cosine=", result["verifier_decision_reason"])

    def test_upright_and_plug_keep_legacy_behavior(self):
        upright = verify_open6dor_agent_outcome(
            _decision("upright"),
            prediction=_prediction([0.0, 0.05, 0.998], [0.0, 0.0, 1.0]),
            orientation_mode="upright",
            stage4_cache_available=True,
            target_orientation={"semantic_axis": [0.0, 0.0, 1.0]},
            direction_attributes=["top"],
        )
        plug = verify_open6dor_agent_outcome(
            _decision("plug_right"),
            prediction=_prediction([0.999, 0.01, 0.01], [1.0, 0.0, 0.0]),
            orientation_mode="plug_right",
            stage4_cache_available=True,
            target_orientation={"semantic_axis": [1.0, 0.0, 0.0]},
            direction_attributes=["plug end"],
        )

        self.assertEqual(upright["verification_status"], "accepted")
        self.assertTrue(upright["old_rule_pass"])
        self.assertTrue(upright["new_rule_pass"])
        self.assertEqual(upright["verifier_rule_version"], "rule_v2_legacy")
        self.assertEqual(plug["verification_status"], "accepted")
        self.assertTrue(plug["old_rule_pass"])
        self.assertTrue(plug["new_rule_pass"])
        self.assertEqual(plug["verifier_rule_version"], "rule_v2_legacy")


if __name__ == "__main__":
    unittest.main()
