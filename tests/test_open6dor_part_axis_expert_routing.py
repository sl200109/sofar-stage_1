import unittest

from sofar.open6dor import open6dor_perception as perception
from sofar.serve.semantic_orientation_agent import decide_open6dor_agent_action


class Open6DORPartAxisExpertRoutingTest(unittest.TestCase):
    def tearDown(self):
        perception.STAGE5_OPTIONS.clear()

    def _configure(self, part_axis_checkpoint=None):
        perception.STAGE5_OPTIONS.clear()
        perception.STAGE5_OPTIONS.update(
            {
                "enabled": True,
                "checkpoint_path": "/tmp/shared_upright.pth",
                "expert_routing": "task_family",
                "expert_checkpoints": {
                    perception.STAGE5_TASK_FAMILY_UPRIGHT: "/tmp/upright.pth",
                    perception.STAGE5_TASK_FAMILY_FLAT: "/tmp/flat.pth",
                    perception.STAGE5_TASK_FAMILY_PLUG: "/tmp/plug.pth",
                    perception.STAGE5_TASK_FAMILY_PART_AXIS: part_axis_checkpoint,
                },
            }
        )

    def test_no_part_axis_checkpoint_cannot_inject_or_reuse_plug(self):
        self._configure(part_axis_checkpoint=None)
        route = perception.resolve_open6dor_stage5_checkpoint_route(
            "handle_right",
            stage4_cache_available=True,
            object_score=1.0,
            part_score=1.0,
        )
        self.assertEqual(route["task_family"], "part_axis_left_right")
        self.assertIsNone(route["checkpoint_path"])
        self.assertEqual(route["checkpoint_source"], "none")
        self.assertFalse(route["family_checkpoint_available"])
        self.assertTrue(route["family_shadow_only"])
        self.assertIn("part_axis_left_right_no_checkpoint_shadow_only", route["route_reason"])

    def test_part_axis_checkpoint_uses_family_specific_source(self):
        self._configure(part_axis_checkpoint="/tmp/part_axis.pth")
        route = perception.resolve_open6dor_stage5_checkpoint_route(
            "handle_right",
            stage4_cache_available=True,
            object_score=1.0,
            part_score=1.0,
        )
        self.assertEqual(route["task_family"], "part_axis_left_right")
        self.assertEqual(route["checkpoint_path"], "/tmp/part_axis.pth")
        self.assertEqual(route["checkpoint_source"], "family_specific")
        self.assertTrue(route["family_checkpoint_available"])
        self.assertFalse(route["family_shadow_only"])

    def test_pscr_verified_part_axis_uses_conditional_verify_band(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="handle_right",
            stage4_cache_available=True,
            object_score=1.0,
            part_score=1.0,
            shadow_enabled=True,
        )
        self.assertEqual(decision["decision"], "use_stage5_conditional_verify")
        self.assertEqual(decision["selected_execution_mode"], "stage5_conditional_verified")


if __name__ == "__main__":
    unittest.main()
