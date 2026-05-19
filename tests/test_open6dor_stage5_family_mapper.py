import unittest
import sys
import types


def _install_test_stubs():
    if "serve.pointso" not in sys.modules:
        sys.modules["serve.pointso"] = types.ModuleType("serve.pointso")
    if "segmentation" not in sys.modules:
        sys.modules["segmentation"] = types.ModuleType("segmentation")
    if "segmentation.sam" not in sys.modules:
        sys.modules["segmentation.sam"] = types.ModuleType("segmentation.sam")
    if "segmentation.florence" not in sys.modules:
        sys.modules["segmentation.florence"] = types.ModuleType("segmentation.florence")
    if "serve.stage5_inference" not in sys.modules:
        stage5_inference = types.ModuleType("serve.stage5_inference")
        stage5_inference.predict_from_stage4_dir = lambda *args, **kwargs: {}
        sys.modules["serve.stage5_inference"] = stage5_inference


_install_test_stubs()

from sofar.open6dor.open6dor_perception import (
    STAGE5_OPTIONS,
    STAGE5_TASK_FAMILY_FLAT,
    STAGE5_TASK_FAMILY_PART_AXIS,
    STAGE5_TASK_FAMILY_PLUG,
    STAGE5_TASK_FAMILY_UNKNOWN,
    STAGE5_TASK_FAMILY_UPRIGHT,
    infer_open6dor_stage5_task_family,
    resolve_open6dor_stage5_checkpoint_route,
    summarize_open6dor_agent_records,
)
from sofar.serve.semantic_orientation_agent import decide_open6dor_agent_action


class Open6DORStage5FamilyMapperTest(unittest.TestCase):
    def setUp(self):
        self._old_stage5_options = dict(STAGE5_OPTIONS)
        STAGE5_OPTIONS.clear()
        STAGE5_OPTIONS.update(
            {
                "expert_routing": "task_family",
                "checkpoint_path": "/tmp/shared.pth",
                "expert_checkpoints": {
                    STAGE5_TASK_FAMILY_UPRIGHT: "/tmp/upright.pth",
                    STAGE5_TASK_FAMILY_FLAT: "/tmp/flat.pth",
                    STAGE5_TASK_FAMILY_PLUG: "/tmp/plug.pth",
                    STAGE5_TASK_FAMILY_PART_AXIS: None,
                },
            }
        )

    def tearDown(self):
        STAGE5_OPTIONS.clear()
        STAGE5_OPTIONS.update(self._old_stage5_options)

    def test_upright_vertical_mapping(self):
        for mode in [
            "upright",
            "watch_upright",
            "tape_measure_upright",
            "upright_lens_forth",
            "upright_textual",
        ]:
            self.assertEqual(infer_open6dor_stage5_task_family(mode), STAGE5_TASK_FAMILY_UPRIGHT)

    def test_flat_mapping(self):
        for mode in ["lying_flat", "upside_down", "lower_rim", "upside_down_textual"]:
            self.assertEqual(infer_open6dor_stage5_task_family(mode), STAGE5_TASK_FAMILY_FLAT)

    def test_plug_mapping(self):
        for mode in [
            "plug_right",
            "prong_right",
            "cap_right",
            "cap_forth",
            "cap_left_bottom_right",
            "cap_right_bottom_left",
            "sideways",
            "sideways_textual",
            "clip_sideways",
            "card_forth_textual",
            "remote_control_forth",
            "earpiece_far",
            "multimeter_forth",
        ]:
            self.assertEqual(infer_open6dor_stage5_task_family(mode), STAGE5_TASK_FAMILY_PLUG)

    def test_part_axis_mapping(self):
        for mode in [
            "handle_left",
            "handle_right",
            "blade_right",
            "blades_right",
            "ballpoint_right",
            "clasp_right",
            "spout_right",
            "bulb_right_handle_left",
        ]:
            self.assertEqual(infer_open6dor_stage5_task_family(mode), STAGE5_TASK_FAMILY_PART_AXIS)

    def test_unknown_mapping(self):
        for mode in ["unknown_mode_xxx", "", None]:
            self.assertEqual(infer_open6dor_stage5_task_family(mode), STAGE5_TASK_FAMILY_UNKNOWN)

    def test_part_axis_without_checkpoint_routes_to_shadow_only(self):
        route = resolve_open6dor_stage5_checkpoint_route(
            "handle_right",
            parser_confidence=None,
            stage4_cache_available=True,
            object_score=None,
            part_score=None,
            fallback_required=False,
        )
        self.assertEqual(route["task_family"], STAGE5_TASK_FAMILY_PART_AXIS)
        self.assertEqual(route["checkpoint_source"], "none")
        self.assertFalse(route["family_checkpoint_available"])
        self.assertTrue(route["family_shadow_only"])
        self.assertIn("part_axis_left_right_no_checkpoint_shadow_only", route["route_reason"])

    def test_pscr_verified_part_axis_without_checkpoint_never_injects(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="handle_right",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family=STAGE5_TASK_FAMILY_PART_AXIS,
            checkpoint_source="none",
            checkpoint_available=False,
        )
        self.assertNotIn(decision["decision"], {"use_stage5_direct", "use_stage5_conditional_verify"})
        self.assertFalse(decision["stage5_allowed"])
        self.assertIn(
            "part_axis_left_right_no_checkpoint_shadow_only",
            decision["agent_signals"]["pscr_block_reason"],
        )

    def test_pscr_verified_plug_family_still_allows_conditional_verify(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family=STAGE5_TASK_FAMILY_PLUG,
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertEqual(decision["decision"], "use_stage5_conditional_verify")
        self.assertTrue(decision["agent_signals"]["pscr_fallback_override"])

    def test_summary_family_statistics(self):
        summary = summarize_open6dor_agent_records(
            [
                {
                    "stage5_enabled": True,
                    "agent_policy": "pscr_verified",
                    "agent_decision": "use_stage5_conditional_verify",
                    "agent_selected_execution_mode": "stage5_conditional_verified",
                    "agent_used_stage5": True,
                    "agent_shadow_used": False,
                    "agent_fallback_to_baseline": False,
                    "agent_verification_status": "accepted",
                    "agent_signals": {"pscr_block_reason": ""},
                    "stage5_mode": "plug_right",
                    "stage5_checkpoint_family": STAGE5_TASK_FAMILY_PLUG,
                    "stage5_checkpoint_source": "family_specific",
                    "stage5_family_shadow_only": False,
                    "stage5_family_checkpoint_available": True,
                    "stage5_orientation_diagnostics": {},
                },
                {
                    "stage5_enabled": True,
                    "agent_policy": "pscr_verified",
                    "agent_decision": "skip_stage5_due_to_missing_checkpoint",
                    "agent_selected_execution_mode": "baseline_only",
                    "agent_used_stage5": False,
                    "agent_shadow_used": False,
                    "agent_fallback_to_baseline": True,
                    "agent_verification_status": "not_run",
                    "agent_signals": {"pscr_block_reason": "part_axis_left_right_no_checkpoint_shadow_only"},
                    "stage5_mode": "handle_right",
                    "stage5_checkpoint_family": STAGE5_TASK_FAMILY_PART_AXIS,
                    "stage5_checkpoint_source": "none",
                    "stage5_family_shadow_only": True,
                    "stage5_family_checkpoint_available": False,
                    "stage5_orientation_diagnostics": {},
                },
                {
                    "stage5_enabled": True,
                    "agent_policy": "pscr_verified",
                    "agent_decision": "skip_stage5_due_to_mode_gating",
                    "agent_selected_execution_mode": "baseline_only",
                    "agent_used_stage5": False,
                    "agent_shadow_used": False,
                    "agent_fallback_to_baseline": True,
                    "agent_verification_status": "not_run",
                    "agent_signals": {"pscr_block_reason": "unknown_family"},
                    "stage5_mode": "unknown_mode_xxx",
                    "stage5_checkpoint_family": STAGE5_TASK_FAMILY_UNKNOWN,
                    "stage5_checkpoint_source": "none",
                    "stage5_family_shadow_only": False,
                    "stage5_family_checkpoint_available": False,
                    "stage5_orientation_diagnostics": {},
                },
            ]
        )
        self.assertGreaterEqual(summary["part_axis_left_right_count"], 1)
        self.assertGreaterEqual(summary["part_axis_left_right_shadow_only_count"], 1)
        self.assertGreaterEqual(summary["unknown_family_count"], 1)
        self.assertGreaterEqual(summary["stage5_used_count_by_family"][STAGE5_TASK_FAMILY_PLUG], 1)
        self.assertIn("pscr_verified", summary["agent_policy_distribution"])
        self.assertIn("pscr_block_reason_distribution", summary)


if __name__ == "__main__":
    unittest.main()
