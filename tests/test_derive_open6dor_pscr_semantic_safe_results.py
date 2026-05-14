import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sofar.analysis.derive_open6dor_pscr_semantic_safe_results import main


def _write_result(method_dir, task_id, marker, *, target_orientation=None, extra=None):
    result_path = method_dir / "eval_dataset_root" / "open6dor_v2" / task_id / "output" / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_position": [1, 2, 3],
        "target_orientation": target_orientation if target_orientation is not None else {"baseline_axis": [0, 1, 0]},
        "marker": marker,
    }
    if extra:
        payload.update(extra)
    result_path.write_text(json.dumps(payload), encoding="utf-8")


class DeriveOpen6DORPSCRModePriorSafeResultsTest(unittest.TestCase):
    def _run(self, args):
        with mock.patch.object(sys, "argv", ["derive"] + args):
            main()

    def test_part_axis_stage5_must_match_x_sign_and_dominant_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            task_a = "task_refine_6dof/a/taskA.__handle_right"
            task_b = "task_refine_6dof/a/taskB.__handle_right"
            _write_result(run_root / "baseline_only", task_a, "baseline_a")
            _write_result(run_root / "baseline_only", task_b, "baseline_b")
            _write_result(run_root / "pscr_verified", task_a, "pscr_a")
            _write_result(run_root / "pscr_verified", task_b, "pscr_b")
            records = {
                "records": [
                    {
                        "task_dir": task_a,
                        "stage5_checkpoint_family": "part_axis_left_right",
                        "stage5_mode": "handle_right",
                        "stage5_direction_vector": [0.98, 0.1, 0.05],
                        "stage5_target_orientation": {"handle": [0.98, 0.1, 0.05]},
                        "official_pass": False,
                    },
                    {
                        "task_dir": task_b,
                        "stage5_checkpoint_family": "part_axis_left_right",
                        "stage5_mode": "handle_right",
                        "stage5_direction_vector": [0.1, 0.98, 0.05],
                        "stage5_target_orientation": {"handle": [0.1, 0.98, 0.05]},
                        "official_pass": True,
                    },
                ]
            }
            records_path = run_root / "records.json"
            records_path.write_text(json.dumps(records), encoding="utf-8")

            self._run(["--run-root", str(run_root), "--records-file", str(records_path)])

            output_root = run_root / "pscr_mode_prior_safe"
            task_a_result = json.loads((output_root / "eval_dataset_root/open6dor_v2" / task_a / "output/result.json").read_text())
            task_b_result = json.loads((output_root / "eval_dataset_root/open6dor_v2" / task_b / "output/result.json").read_text())
            selection = json.loads((output_root / "selection.json").read_text())
            self.assertEqual(task_a_result["marker"], "baseline_a")
            self.assertEqual(task_a_result["target_position"], [1, 2, 3])
            self.assertEqual(task_a_result["target_orientation"], {"handle": [0.98, 0.1, 0.05]})
            self.assertEqual(task_b_result["target_orientation"], {"baseline_axis": [0, 1, 0]})
            self.assertEqual(selection["selected_count"], 1)
            self.assertEqual(selection["selected_by_source"], {"stage5_part_axis_x_dominant": 1})

    def test_part_axis_missing_baseline_orientation_uses_mode_prior_without_pscr_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            task_id = "task_refine_6dof/a/taskC.__handle_left"
            _write_result(run_root / "baseline_only", task_id, "baseline_c", target_orientation={})

            self._run(["--run-root", str(run_root)])

            output_root = run_root / "pscr_mode_prior_safe"
            result = json.loads((output_root / "eval_dataset_root/open6dor_v2" / task_id / "output/result.json").read_text())
            selection = json.loads((output_root / "selection.json").read_text())
            self.assertEqual(result["marker"], "baseline_c")
            self.assertEqual(result["target_orientation"], {"handle": [-1.0, 0.0, 0.0]})
            self.assertEqual(selection["selected_count"], 1)
            self.assertEqual(selection["skipped_missing_pscr_result"], [task_id])

    def test_flat_modes_only_inject_on_explicit_semantic_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            task_a = "task_refine_6dof/a/taskA.__lying_flat"
            task_b = "task_refine_6dof/a/taskB.__lying_flat"
            for task_id in [task_a, task_b]:
                _write_result(run_root / "baseline_only", task_id, "baseline")
                _write_result(run_root / "pscr_verified", task_id, "pscr")
            records = {
                "records": [
                    {
                        "task_dir": task_a,
                        "stage5_checkpoint_family": "flat_upside_down_lying_flat",
                        "stage5_mode": "lying_flat",
                        "stage5_semantic_status": "top_points_down",
                        "stage5_target_orientation": {"larger face": [0.0, 0.0, -1.0]},
                    },
                    {
                        "task_dir": task_b,
                        "stage5_checkpoint_family": "flat_upside_down_lying_flat",
                        "stage5_mode": "lying_flat",
                        "stage5_semantic_status": "axis_sideways",
                        "stage5_target_orientation": {"larger face": [1.0, 0.0, 0.0]},
                    },
                ]
            }
            records_path = run_root / "records.json"
            records_path.write_text(json.dumps(records), encoding="utf-8")

            self._run(["--run-root", str(run_root), "--records-file", str(records_path)])

            output_root = run_root / "pscr_mode_prior_safe"
            task_a_result = json.loads((output_root / "eval_dataset_root/open6dor_v2" / task_a / "output/result.json").read_text())
            task_b_result = json.loads((output_root / "eval_dataset_root/open6dor_v2" / task_b / "output/result.json").read_text())
            selection = json.loads((output_root / "selection.json").read_text())
            self.assertEqual(task_a_result["target_orientation"], {"larger face": [0.0, 0.0, -1.0]})
            self.assertEqual(task_b_result["target_orientation"], {"baseline_axis": [0, 1, 0]})
            self.assertEqual(selection["selected_count"], 1)
            self.assertEqual(selection["selected_by_semantic_status"], {"top_points_down": 1})

    def test_eval_6dof_and_pass_fail_fields_do_not_affect_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            task_id = "task_refine_6dof/a/taskD.__handle_right"
            _write_result(run_root / "baseline_only", task_id, "baseline_d", extra={"pass": False, "success": False})
            _write_result(run_root / "pscr_verified", task_id, "pscr_d", extra={"pass": True, "success": True})
            evaluator_dir = run_root / "pscr_verified" / "evaluator_output"
            evaluator_dir.mkdir(parents=True)
            (evaluator_dir / "eval_6dof.json").write_text(json.dumps({task_id: {"pass": False}}), encoding="utf-8")
            records = {
                "records": [
                    {
                        "task_dir": task_id,
                        "stage5_checkpoint_family": "part_axis_left_right",
                        "stage5_mode": "handle_right",
                        "stage5_direction_vector": [0.9, 0.1, 0.1],
                        "stage5_target_orientation": {"handle": [0.9, 0.1, 0.1]},
                        "pass": False,
                        "fail": True,
                    }
                ]
            }
            records_path = run_root / "records.json"
            records_path.write_text(json.dumps(records), encoding="utf-8")

            self._run(["--run-root", str(run_root), "--records-file", str(records_path)])

            output_root = run_root / "pscr_mode_prior_safe"
            result = json.loads((output_root / "eval_dataset_root/open6dor_v2" / task_id / "output/result.json").read_text())
            selection = json.loads((output_root / "selection.json").read_text())
            self.assertEqual(result["marker"], "baseline_d")
            self.assertEqual(result["target_orientation"], {"handle": [0.9, 0.1, 0.1]})
            self.assertEqual(selection["forbidden_inputs"], ["eval_6dof.json", "official_pass_fail", "evaluator_gt"])
            self.assertEqual(selection["selected_count"], 1)


if __name__ == "__main__":
    unittest.main()
