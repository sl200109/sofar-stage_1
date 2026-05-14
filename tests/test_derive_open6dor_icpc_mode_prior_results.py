import json
import tempfile
import unittest
from pathlib import Path

from analysis.derive_open6dor_icpc_mode_prior_results import derive, parse_args


def _make_args(*args):
    original = ["derive"] + list(args)
    import sys

    previous = sys.argv
    try:
        sys.argv = original
        return parse_args()
    finally:
        sys.argv = previous


def _write_result(baseline_dir, task_path, *, target_orientation=None, direction_attributes=None):
    result_path = baseline_dir / "eval_dataset_root" / "open6dor_v2" / task_path / "output" / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_position": [0.1, 0.2, 0.3],
        "target_orientation": {} if target_orientation is None else target_orientation,
    }
    if direction_attributes is not None:
        payload["direction_attributes"] = direction_attributes
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _read_output(output_root, task_path):
    path = output_root / "icpc_mode_prior" / "eval_dataset_root" / "open6dor_v2" / task_path / "output" / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


class ICPCModePriorDerivationTest(unittest.TestCase):
    def _run(self, tmp, *extra_args):
        baseline_dir = Path(tmp) / "baseline_only"
        output_root = Path(tmp) / "icpc_out"
        args = _make_args(
            "--baseline-dir",
            str(baseline_dir),
            "--output-root",
            str(output_root),
            *extra_args,
        )
        selection = derive(args)
        return baseline_dir, output_root, selection

    def test_ballpoint_right_patches_empty_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__ballpoint_right/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {"ballpoint": [1, 0, 0]})
            self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])
            self.assertEqual(selection["method"], "icpc_mode_prior")
            self.assertEqual((output_root / "icpc_mode_prior").name, "icpc_mode_prior")
            self.assertNotEqual(selection["method"], "pscr_mode_prior_safe")

    def test_handle_left_blade_right_and_blades_right_priors(self):
        cases = [
            ("handle_left", {"handle": [-1, 0, 0]}),
            ("blade_right", {"blade": [1, 0, 0]}),
            ("blades_right", {"blades": [1, 0, 0]}),
        ]
        for mode, expected in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                task = f"task_refine_6dof/Place_x.__{mode}/20240513"
                baseline_dir = Path(tmp) / "baseline_only"
                _write_result(baseline_dir, task)

                _, output_root, _ = self._run(tmp)
                result = _read_output(output_root, task)

                self.assertEqual(result["target_orientation"], expected)
                self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])

    def test_lying_flat_patches_larger_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__lying_flat/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task, direction_attributes=["larger face"])

            _, output_root, selection = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {"larger face": [0, 0, -1]})
            self.assertEqual(selection["patched_by_rule"], {"lying_flat_larger_face": 1})

    def test_upright_uses_bottom_or_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__upright/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task, direction_attributes=["base"])

            _, output_root, _ = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {"base": [0, 0, -1]})
            self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])

    def test_unknown_mode_does_not_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__glasses/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {})
            self.assertEqual(selection["patched_count"], 0)
            self.assertEqual(selection["skipped_reasons"], {"skipped_unknown_mode": 1})

    def test_existing_orientation_not_replaced_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__handle_right/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task, target_orientation={"handle": [-1, 0, 0]})

            _, output_root, selection = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {"handle": [-1, 0, 0]})
            self.assertEqual(selection["patched_count"], 0)
            self.assertEqual(selection["skipped_reasons"], {"skipped_existing_orientation": 1})

    def test_replace_conflict_replaces_opposite_same_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__handle_right/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task, target_orientation={"handle": [-1, 0, 0]})

            _, output_root, selection = self._run(tmp, "--replace-conflict")
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {"handle": [1, 0, 0]})
            self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])
            self.assertEqual(selection["patched_by_reason"], {"conflict_replaced": 1})
            self.assertEqual(selection["patched_records"][0]["reason"], "conflict_replaced")

    def test_fake_evaluator_output_does_not_affect_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__ballpoint_right/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)
            evaluator_output = baseline_dir / "evaluator_output"
            evaluator_output.mkdir(parents=True)
            (evaluator_output / "eval_6dof.json").write_text(json.dumps({"fake": "fail"}), encoding="utf-8")

            _, _, selection_a = self._run(tmp)
            (evaluator_output / "eval_6dof.json").write_text(json.dumps({"fake": "pass"}), encoding="utf-8")
            _, _, selection_b = self._run(tmp)

            self.assertEqual(selection_a["patched_count"], 1)
            self.assertEqual(selection_b["patched_count"], 1)
            self.assertEqual(selection_a["patched_by_mode"], selection_b["patched_by_mode"])

    def test_part_axis_only_profile_skips_lying_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__lying_flat/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp, "--profile", "part_axis_only")
            result = _read_output(output_root, task)

            self.assertEqual(result["target_orientation"], {})
            self.assertEqual(selection["patched_count"], 0)


if __name__ == "__main__":
    unittest.main()
