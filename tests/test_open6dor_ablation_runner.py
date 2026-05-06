import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from sofar.analysis.run_open6dor_subset_ablation import (
    build_method_command,
    extract_pipeline_summary_fields,
    make_sliced_task_list,
    parse_eval_output,
    summarize_valid_results,
)


class _Args:
    python = "python"
    speed_profile = "conservative"
    stage5_checkpoint = None
    stage5_upright_expert_checkpoint = "/tmp/upright.pth"
    stage5_flat_expert_checkpoint = "/tmp/flat.pth"
    stage5_plug_expert_checkpoint = "/tmp/plug.pth"


class Open6DORAblationRunnerTest(unittest.TestCase):
    def make_temp_dir(self):
        path = Path("D:/桌面/sofar实验同步/.tmp_test_open6dor_ablation_runner") / str(uuid.uuid4())
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_dry_run_command_generation(self):
        args = _Args()
        baseline = build_method_command("baseline_only", args, Path("/tmp/tasks.json"))
        self.assertNotIn("--use-stage5-head", baseline["command"])
        self.assertNotIn("--stage5-checkpoint", baseline["command"])
        self.assertNotIn("--stage5-upright-expert-checkpoint", baseline["command"])
        self.assertNotIn("--stage5-flat-expert-checkpoint", baseline["command"])
        self.assertNotIn("--stage5-plug-expert-checkpoint", baseline["command"])

        safe = build_method_command("pscr_rule_v2_safe", args, Path("/tmp/tasks.json"))
        self.assertIn("--use-stage5-head", safe["command"])
        self.assertIn("--agent-policy", safe["command"])
        self.assertIn("rule_v2", safe["command"])
        self.assertIn("--agent-shadow-eval", safe["command"])

    def test_baseline_only_does_not_force_agent_mode_when_unsupported(self):
        args = _Args()
        with mock.patch(
            "sofar.analysis.run_open6dor_subset_ablation.detect_open6dor_agent_mode_support",
            return_value=set(),
        ):
            baseline = build_method_command("baseline_only", args, Path("/tmp/tasks.json"))
        self.assertNotIn("--agent-mode", baseline["command"])
        self.assertIn("baseline_only_no_stage5_no_agent_mode", baseline["notes"])

    def test_parse_eval_output(self):
        stdout = "\n".join(
            [
                "Position L0: 96.0",
                "Position L1: 81.5",
                "Rotation L0: 68.6",
                "Rotation L1: 42.2",
                "Rotation L2: 70.1",
                "6-DoF Overall: 48.7",
            ]
        )
        metrics = parse_eval_output(stdout, "", "/tmp/nonexistent")
        self.assertEqual(metrics["position_l0"], 96.0)
        self.assertEqual(metrics["position_l1"], 81.5)
        self.assertEqual(metrics["rotation_l0"], 68.6)
        self.assertEqual(metrics["rotation_l1"], 42.2)
        self.assertEqual(metrics["rotation_l2"], 70.1)
        self.assertEqual(metrics["six_dof_overall"], 48.7)

    def test_summarize_valid_results(self):
        tmpdir = self.make_temp_dir()
        method_dir = tmpdir / "baseline_only"
        mirror_root = method_dir / "eval_dataset_root" / "open6dor_v2"
        valid_task = Path("task_refine_pos/a/b/task_valid")
        invalid_task = Path("task_refine_pos/a/b/task_invalid")

        valid_result = mirror_root / valid_task / "output" / "result.json"
        valid_result.parent.mkdir(parents=True, exist_ok=True)
        valid_result.write_text(json.dumps({"target_position": [0.1, 0.2, 0.3]}), encoding="utf-8")

        invalid_result = mirror_root / invalid_task / "output" / "result.json"
        invalid_result.parent.mkdir(parents=True, exist_ok=True)
        invalid_result.write_text(json.dumps({"target_position": [0.1, 0.2]}), encoding="utf-8")

        summary = summarize_valid_results(method_dir, [str(valid_task), str(invalid_task)])
        self.assertEqual(summary["total_task_count"], 2)
        self.assertEqual(summary["valid_result_count"], 1)

    def test_extract_pipeline_summary_fields(self):
        method_dir = self.make_temp_dir()
        summary_payload = {
            "avg_success_sec": 12.3,
            "median_success_sec": 11.1,
            "error_count": 2,
            "agent_summary": {
                "total_records": 20,
                "used_stage5_count": 7,
                "fallback_count": 13,
                "rejected_count": 3,
                "verification_distribution": {"accepted": 4, "rejected": 3},
            },
            "reasoning_json_summary": {
                "repaired_count": 5,
                "degraded_count": 2,
            },
        }
        (method_dir / "open6dor_perception_summary_test.json").write_text(
            json.dumps(summary_payload),
            encoding="utf-8",
        )
        parsed = extract_pipeline_summary_fields(method_dir)
        self.assertEqual(parsed["stage5_used_count"], 7)
        self.assertEqual(parsed["fallback_count"], 13)
        self.assertEqual(parsed["reasoning_json_repaired_count"], 5)
        self.assertEqual(parsed["reasoning_json_degraded_count"], 2)

    def test_make_sliced_task_list(self):
        tmpdir = self.make_temp_dir()
        task_list_path = tmpdir / "tasks.json"
        original = [f"task_{idx}" for idx in range(5)]
        task_list_path.write_text(json.dumps(original), encoding="utf-8")
        output_run_dir = tmpdir / "run"
        sliced_path, sliced_entries = make_sliced_task_list(task_list_path, output_run_dir, 2)

        self.assertTrue(sliced_path.exists())
        self.assertEqual(sliced_path.name, "task_list_first2.json")
        self.assertEqual(sliced_entries, ["task_0", "task_1"])
        self.assertEqual(json.loads(task_list_path.read_text(encoding="utf-8")), original)

    def test_handoff_50_case_command_contains_required_args(self):
        handoff = Path("D:/桌面/sofar实验同步/交接操作.txt").read_text(encoding="utf-8")
        self.assertIn("--run-id smoke50_run", handoff)
        self.assertIn("--max-tasks 50", handoff)
        self.assertIn("--task-list /data/coding/SoFar/paper_results/open6dor_short_experiments_20260429/open6dor_eval_subset_400_from4389_seed42_task_list.json", handoff)
        self.assertIn("--output-root /data/coding/SoFar/output/open6dor_ablation_smoke", handoff)
        self.assertIn("--speed-profile conservative", handoff)
        self.assertIn("--stage5-upright-expert-checkpoint /data/coding/SoFar/output/stage5_open6dor_upright_expert_round2_semanticfix/stage5_pilot_best.pth", handoff)
        self.assertIn("--stage5-flat-expert-checkpoint /data/coding/SoFar/output/stage5_open6dor_flat_expert_round2_scratch/stage5_pilot_best.pth", handoff)


if __name__ == "__main__":
    unittest.main()
