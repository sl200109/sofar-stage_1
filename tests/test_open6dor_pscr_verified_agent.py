import sys
import types
import unittest
from pathlib import Path


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

from sofar.analysis.run_open6dor_subset_ablation import build_method_command
from sofar.open6dor.open6dor_perception import (
    STAGE5_OPTIONS,
    STAGE5_TASK_FAMILY_FLAT,
    STAGE5_TASK_FAMILY_PART_AXIS,
    STAGE5_TASK_FAMILY_PLUG,
    STAGE5_TASK_FAMILY_UPRIGHT,
    resolve_open6dor_stage5_checkpoint_route,
)
from sofar.serve.semantic_orientation_agent import decide_open6dor_agent_action


class _Args:
    python = "python"
    speed_profile = "conservative"
    stage5_checkpoint = None
    stage5_upright_expert_checkpoint = "/tmp/upright.pth"
    stage5_flat_expert_checkpoint = "/tmp/flat.pth"
    stage5_plug_expert_checkpoint = "/tmp/plug.pth"


class Open6DORPSCRVerifiedAgentTest(unittest.TestCase):
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

    def test_fallback_required_with_checkpoint_allows_conditional_verify(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="plug_cap_sideways",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertEqual(decision["decision"], "use_stage5_conditional_verify")
        self.assertTrue(decision["stage5_allowed"])
        self.assertTrue(decision["agent_signals"]["pscr_fallback_override"])

    def test_stage4_cache_missing_still_blocks(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=False,
            task_family="plug_cap_sideways",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertFalse(decision["stage5_allowed"])
        self.assertIn("missing_stage4_cache", decision["decision"])

    def test_unknown_family_cannot_inject(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="unknown_mode",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="unknown",
            checkpoint_source="none",
            checkpoint_available=False,
        )
        self.assertFalse(decision["stage5_allowed"])
        self.assertNotEqual(decision["decision"], "use_stage5_conditional_verify")

    def test_part_axis_without_checkpoint_cannot_inject(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="handle_right",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="part_axis_left_right",
            checkpoint_source="none",
            checkpoint_available=False,
        )
        self.assertFalse(decision["stage5_allowed"])
        self.assertIn(
            "part_axis_left_right_no_checkpoint_shadow_only",
            decision["decision_reason"],
        )

    def test_part_axis_with_checkpoint_allows_conditional_verify(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="handle_right",
            agent_policy="pscr_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="part_axis_left_right",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertEqual(decision["decision"], "use_stage5_conditional_verify")

    def test_route_fallback_required_no_longer_early_returns(self):
        route = resolve_open6dor_stage5_checkpoint_route(
            "plug_right",
            fallback_required=True,
            stage4_cache_available=True,
        )
        self.assertEqual(route["checkpoint_source"], "family_specific")
        self.assertTrue(route["family_checkpoint_available"])
        self.assertEqual(route["checkpoint_path"], "/tmp/plug.pth")

    def test_ablation_runner_uses_pscr_verified(self):
        pscr = build_method_command("pscr_verified", _Args(), Path("/tmp/tasks.json"))
        self.assertIn("--agent-policy", pscr["command"])
        self.assertIn("pscr_verified", pscr["command"])
        self.assertIn("--use-stage5-head", pscr["command"])
        self.assertIn("--stage5-expert-routing", pscr["command"])
        self.assertIn("task_family", pscr["command"])

        baseline = build_method_command("baseline_only", _Args(), Path("/tmp/tasks.json"))
        self.assertNotIn("--use-stage5-head", baseline["command"])


if __name__ == "__main__":
    unittest.main()
