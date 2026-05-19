import json
import shutil
import unittest
import uuid
from pathlib import Path

from sofar.analysis.analyze_open6dor_oracle_and_axis_errors import run_analysis


class _Args:
    def __init__(self, run_root, task_list, output_dir=None):
        self.run_root = str(run_root)
        self.methods = "baseline_only,pscr_verified"
        self.task_list = str(task_list)
        self.output_dir = str(output_dir) if output_dir else None


class AnalyzeOpen6DOROracleAxisErrorsTest(unittest.TestCase):
    def make_temp_dir(self):
        path = Path("D:/桌面/sofar实验同步/.tmp_test_open6dor_oracle_axis") / str(uuid.uuid4())
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def make_method(self, run_root, method, eval_payload):
        method_dir = run_root / method
        self.write_json(method_dir / "evaluator_output" / "eval_6dof.json", eval_payload)
        for task_id in ["run_a", "run_b"]:
            task_dir = (
                method_dir
                / "eval_dataset_root"
                / "open6dor_v2"
                / "task_refine_6dof"
                / "center"
                / f"Task_{task_id}.__upside_down"
                / task_id
            )
            self.write_json(task_dir / "task_config_new5.json", {"position_tag": "center", "init_obj_pos": [[0, 0, 0]]})
            self.write_json(
                task_dir / "output" / "result.json",
                {
                    "target_position": [0, 0, 0],
                    "init_orientation": {"top": [0, 0, 1]},
                    "target_orientation": {"top": [0, 0, 1]},
                    "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
            )

    def test_oracle_union_breakdown_and_sign_flip_outputs(self):
        tmpdir = self.make_temp_dir()
        run_root = tmpdir / "run"
        task_list = tmpdir / "tasks.json"
        self.write_json(
            task_list,
            [
                "/data/open6dor_v2/task_refine_6dof/center/Task_run_a.__upside_down/run_a",
                "/data/open6dor_v2/task_refine_6dof/center/Task_run_b.__plug_right/run_b",
            ],
        )

        self.make_method(
            run_root,
            "baseline_only",
            {
                "run_a": {"pos_success": 1, "deviation": 90},
                "run_b": {"pos_success": 1, "deviation": 10},
            },
        )
        self.make_method(
            run_root,
            "pscr_verified",
            {
                "run_a": {"pos_success": 1, "deviation": 10},
                "run_b": {"pos_success": 1, "deviation": 90},
            },
        )
        self.write_json(
            run_root / "pscr_verified" / "stage5_open6dor_pipeline_records_test.json",
            {
                "records": [
                    {
                        "task_dir": "/data/open6dor_v2/task_refine_6dof/center/Task_run_a.__upside_down/run_a",
                        "stage5_mode": "upside_down",
                        "stage5_checkpoint_family": "flat_upside_down_lying_flat",
                        "stage5_semantic_status": "top_points_down",
                        "agent_used_stage5": True,
                        "agent_decision": "use_stage5_conditional_verify",
                    },
                    {
                        "task_dir": "/data/open6dor_v2/task_refine_6dof/center/Task_run_b.__plug_right/run_b",
                        "stage5_mode": "plug_right",
                        "stage5_checkpoint_family": "plug_cap_sideways",
                        "stage5_semantic_status": "axis_sideways",
                        "agent_used_stage5": True,
                        "agent_decision": "use_stage5_conditional_verify",
                    },
                ]
            },
        )

        def fake_rotation_evaluator(config_path, result_payload, task_id="", flip_name=""):
            return 0 if flip_name == "x" else 90

        output_dir = tmpdir / "analysis"
        summary = run_analysis(_Args(run_root, task_list, output_dir), rotation_evaluator=fake_rotation_evaluator)

        self.assertEqual(summary["method_metrics"]["baseline_only"]["all_pass_count"], 1)
        self.assertEqual(summary["method_metrics"]["pscr_verified"]["all_pass_count"], 1)
        self.assertEqual(summary["oracle_union"]["all_pass_count"], 2)
        self.assertTrue((output_dir / "oracle_summary.json").exists())
        self.assertTrue((output_dir / "oracle_summary.csv").exists())
        self.assertTrue((output_dir / "family_breakdown.csv").exists())
        self.assertTrue((output_dir / "mode_breakdown.csv").exists())
        self.assertTrue((output_dir / "semantic_status_breakdown.csv").exists())
        self.assertTrue((output_dir / "sign_flip_ablation.csv").exists())

        sign_flip_rows = (output_dir / "sign_flip_ablation.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sign_flip_rows), 6)
        self.assertIn("x,", sign_flip_rows[1])

        semantic_csv = (output_dir / "semantic_status_breakdown.csv").read_text(encoding="utf-8")
        self.assertIn("top_points_down", semantic_csv)
        self.assertIn("axis_sideways", semantic_csv)


if __name__ == "__main__":
    unittest.main()
