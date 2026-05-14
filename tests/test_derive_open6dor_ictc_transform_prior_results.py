import json
import tempfile
import unittest
from pathlib import Path

from analysis.derive_open6dor_ictc_transform_prior_results import derive, parse_args


BASE_MATRIX = [
    [0, -1, 0, 0.11],
    [1, 0, 0, 0.22],
    [0, 0, 1, 0.33],
    [0, 0, 0, 1],
]
IDENTITY = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
YAW_180 = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
]
X_180 = [
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
]


def _make_args(*args):
    import sys

    previous = sys.argv
    try:
        sys.argv = ["derive"] + list(args)
        return parse_args()
    finally:
        sys.argv = previous


def _write_result(baseline_dir, task_path, *, transform_matrix=None):
    result_path = baseline_dir / "eval_dataset_root" / "open6dor_v2" / task_path / "output" / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_position": [0.1, 0.2, 0.3],
        "target_orientation": {},
        "transform_matrix": transform_matrix if transform_matrix is not None else BASE_MATRIX,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _rotation(result):
    return [row[:3] for row in result["transform_matrix"][:3]]


def _translation(result):
    return [row[3] for row in result["transform_matrix"][:3]]


def _read_output(output_root, task_path):
    path = output_root / "ictc_transform_prior" / "eval_dataset_root" / "open6dor_v2" / task_path / "output" / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


class ICTCTransformPriorDerivationTest(unittest.TestCase):
    def _run(self, tmp, *extra_args):
        baseline_dir = Path(tmp) / "baseline_only"
        output_root = Path(tmp) / "ictc_out"
        args = _make_args(
            "--baseline-dir",
            str(baseline_dir),
            "--output-root",
            str(output_root),
            *extra_args,
        )
        selection = derive(args)
        return baseline_dir, output_root, selection

    def test_ballpoint_right_canonical_patches_rotation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__ballpoint_right/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp, "--profile", "part_axis_only", "--matrix-template", "canonical")
            result = _read_output(output_root, task)

            self.assertEqual(_rotation(result), IDENTITY)
            self.assertEqual(_translation(result), [0.11, 0.22, 0.33])
            self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])
            self.assertEqual(selection["method"], "ictc_transform_prior")
            self.assertEqual((output_root / "ictc_transform_prior").name, "ictc_transform_prior")
            self.assertNotIn("pscr", selection["method"])
            self.assertNotEqual(selection["method"], "icpc_mode_prior")

    def test_handle_left_uses_yaw_180(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__handle_left/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp, "--profile", "part_axis_only")
            result = _read_output(output_root, task)

            self.assertEqual(_rotation(result), YAW_180)
            self.assertEqual(_translation(result), [0.11, 0.22, 0.33])
            self.assertEqual(selection["patched_by_rule"], {"part_axis_left_yaw180": 1})

    def test_lying_flat_flat_upright_only_uses_x_180(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__lying_flat/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp, "--profile", "flat_upright_only")
            result = _read_output(output_root, task)

            self.assertEqual(_rotation(result), X_180)
            self.assertEqual(_translation(result), [0.11, 0.22, 0.33])
            self.assertEqual(selection["patched_by_rule"], {"lying_flat_x180": 1})

    def test_upright_flat_upright_only_uses_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__upright/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp, "--profile", "flat_upright_only")
            result = _read_output(output_root, task)

            self.assertEqual(_rotation(result), IDENTITY)
            self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])
            self.assertEqual(selection["patched_by_rule"], {"upright_identity": 1})

    def test_unknown_mode_does_not_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__glasses/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["transform_matrix"], BASE_MATRIX)
            self.assertEqual(selection["patched_count"], 0)
            self.assertEqual(selection["skipped_reasons"], {"skipped_unknown_mode": 1})

    def test_upside_down_does_not_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__upside_down/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp, "--profile", "all_safe")
            result = _read_output(output_root, task)

            self.assertEqual(result["transform_matrix"], BASE_MATRIX)
            self.assertEqual(selection["patched_count"], 0)
            self.assertEqual(selection["skipped_reasons"], {"skipped_upside_down_uncertain": 1})

    def test_target_position_and_translation_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "task_refine_6dof/Place_x.__blade_right/20240513"
            baseline_dir = Path(tmp) / "baseline_only"
            _write_result(baseline_dir, task)

            _, output_root, selection = self._run(tmp)
            result = _read_output(output_root, task)

            self.assertEqual(result["target_position"], [0.1, 0.2, 0.3])
            self.assertEqual(_translation(result), [0.11, 0.22, 0.33])
            self.assertEqual(selection["target_position_modified_count"], 0)
            self.assertEqual(selection["translation_modified_count"], 0)

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

    def test_yaw_only_part_axis_forces_yaw_template_and_skips_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_dir = Path(tmp) / "baseline_only"
            left_task = "task_refine_6dof/Place_x.__handle_left/20240513"
            flat_task = "task_refine_6dof/Place_y.__lying_flat/20240513"
            _write_result(baseline_dir, left_task)
            _write_result(baseline_dir, flat_task)

            _, output_root, selection = self._run(tmp, "--profile", "yaw_only_part_axis", "--matrix-template", "canonical")
            left_result = _read_output(output_root, left_task)
            flat_result = _read_output(output_root, flat_task)

            self.assertEqual(selection["matrix_template"], "yaw_only")
            self.assertEqual(_rotation(left_result), YAW_180)
            self.assertEqual(flat_result["transform_matrix"], BASE_MATRIX)
            self.assertEqual(selection["patched_count"], 1)
            self.assertEqual(selection["skipped_reasons"], {"skipped_profile_not_allowed": 1})


if __name__ == "__main__":
    unittest.main()
