import unittest

from sofar.analysis.run_open6dor_subset_ablation import build_method_command, detect_rule_v3_support
from sofar.serve.semantic_orientation_agent import decide_open6dor_agent_action


class _Args:
    python = "python"
    speed_profile = "conservative"
    stage5_checkpoint = None
    stage5_upright_expert_checkpoint = None
    stage5_flat_expert_checkpoint = None
    stage5_plug_expert_checkpoint = None


class Open6DORRuleV3AgentTest(unittest.TestCase):
    def test_rule_v2_keeps_fallback_required_skip_behavior(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="rule_v2",
            fallback_required=True,
            parser_confidence=None,
            stage4_cache_available=True,
            task_family="plug_cap_sideways",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertEqual(decision["decision"], "skip_stage5_due_to_fallback_required")

    def test_rule_v3_allows_fallback_required_conditional_verify(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="rule_v3_verified",
            fallback_required=True,
            parser_confidence=None,
            stage4_cache_available=True,
            task_family="plug_cap_sideways",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertEqual(decision["decision"], "use_stage5_conditional_verify")
        self.assertTrue(decision["stage5_allowed"])
        self.assertTrue(decision["agent_signals"]["rule_v3_allowed_by_fallback_override"])

    def test_rule_v3_blocks_missing_stage4_cache(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="rule_v3_verified",
            fallback_required=True,
            stage4_cache_available=False,
            task_family="plug_cap_sideways",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertFalse(decision["stage5_allowed"])
        self.assertIn("missing_stage4_cache", decision["agent_signals"]["rule_v3_block_reason"])

    def test_rule_v3_blocks_unknown_family(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="unknown_mode",
            agent_policy="rule_v3_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="unknown",
            checkpoint_source="none",
            checkpoint_available=False,
        )
        self.assertNotIn(decision["decision"], {"use_stage5_direct", "use_stage5_conditional_verify"})
        self.assertFalse(decision["stage5_allowed"])
        self.assertIn("unknown_family", decision["agent_signals"]["rule_v3_block_reason"])

    def test_rule_v3_part_axis_without_checkpoint_never_injects(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="handle_right",
            agent_policy="rule_v3_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="part_axis_left_right",
            checkpoint_source="none",
            checkpoint_available=False,
        )
        self.assertNotIn(decision["decision"], {"use_stage5_direct", "use_stage5_conditional_verify"})
        self.assertFalse(decision["stage5_allowed"])
        self.assertIn(
            "part_axis_left_right_no_checkpoint_shadow_only",
            decision["agent_signals"]["rule_v3_block_reason"],
        )

    def test_rule_v3_parser_confidence_missing_does_not_block(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="upright",
            agent_policy="rule_v3_verified",
            fallback_required=False,
            parser_confidence=None,
            stage4_cache_available=True,
            task_family="upright_vertical",
            checkpoint_source="family_specific",
            checkpoint_available=True,
        )
        self.assertEqual(decision["decision"], "use_stage5_conditional_verify")
        self.assertTrue(decision["agent_signals"]["parser_confidence_missing"])

    def test_rule_v3_missing_checkpoint_skips(self):
        decision = decide_open6dor_agent_action(
            stage5_enabled=True,
            orientation_mode="plug_right",
            agent_policy="rule_v3_verified",
            fallback_required=True,
            stage4_cache_available=True,
            task_family="plug_cap_sideways",
            checkpoint_source="none",
            checkpoint_available=False,
        )
        self.assertNotEqual(decision["decision"], "use_stage5_conditional_verify")
        self.assertIn("missing_checkpoint", decision["agent_signals"]["rule_v3_block_reason"])

    def test_ablation_runner_recognizes_rule_v3(self):
        self.assertTrue(detect_rule_v3_support())
        method = build_method_command("pscr_rule_v3_verified", _Args(), "/tmp/tasks.json")
        self.assertTrue(method["runnable"])
        self.assertIn("--agent-policy", method["command"])
        self.assertIn("rule_v3_verified", method["command"])


if __name__ == "__main__":
    unittest.main()
