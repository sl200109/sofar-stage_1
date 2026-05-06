import unittest

from sofar.serve.semantic_orientation_agent import infer_open6dor_execution_band


class Open6DORRuntimeAgentModesTest(unittest.TestCase):
    def test_upright_family_modes_use_direct_allow(self):
        self.assertEqual(infer_open6dor_execution_band("watch_upright"), "direct_allow")
        self.assertEqual(infer_open6dor_execution_band("tape_measure_upright"), "direct_allow")
        self.assertEqual(infer_open6dor_execution_band("upright_textual"), "direct_allow")

    def test_plug_family_modes_use_conditional_verify(self):
        self.assertEqual(infer_open6dor_execution_band("handle_right"), "conditional_verify")
        self.assertEqual(infer_open6dor_execution_band("blade_right"), "conditional_verify")
        self.assertEqual(infer_open6dor_execution_band("ballpoint_right"), "conditional_verify")


if __name__ == "__main__":
    unittest.main()
