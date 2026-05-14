import unittest
from pathlib import Path

from sofar.analysis.run_open6dor_subset_ablation import build_method_command


class _Args:
    python = "python"
    speed_profile = "conservative"
    stage5_checkpoint = None
    stage5_upright_expert_checkpoint = "/tmp/upright.pth"
    stage5_flat_expert_checkpoint = "/tmp/flat.pth"
    stage5_plug_expert_checkpoint = "/tmp/plug.pth"
    stage5_part_axis_expert_checkpoint = "/tmp/part_axis.pth"


class Open6DORPSCRVerifiedAgentTest(unittest.TestCase):
    def test_pscr_verified_command_passes_part_axis_checkpoint(self):
        command = build_method_command("pscr_verified", _Args(), Path("/tmp/tasks.json"))["command"]
        self.assertIn("--use-stage5-head", command)
        self.assertIn("--stage5-expert-routing", command)
        self.assertIn("--agent-policy", command)
        self.assertIn("pscr_verified", command)
        self.assertIn("--stage5-part-axis-expert-checkpoint", command)
        self.assertIn("/tmp/part_axis.pth", command)


if __name__ == "__main__":
    unittest.main()
