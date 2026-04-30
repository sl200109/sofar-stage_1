from pathlib import Path
import unittest

from sofar.open6dor.eval_subset_sampling import (
    build_eval_subset_from_task_dirs,
    classify_task_family,
    parse_orientation_mode_from_task_dir,
    validate_eval_subset_summary,
)


class Open6DORSubsetSamplingTest(unittest.TestCase):
    def test_parse_orientation_mode(self):
        task_dir = Path("/data/coding/SoFar/datasets/open6dor_v2/open6dor_v2/task_refine_6dof/front/Place_x.__upright")
        self.assertEqual(parse_orientation_mode_from_task_dir(task_dir), "upright")

    def test_parse_orientation_mode_from_run_level_path(self):
        plug_run = Path("/data/coding/SoFar/datasets/open6dor_v2/open6dor_v2/task_refine_6dof/behind/Place_the_USB_behind_the_lighter_on_the_table.__plug_right/20240824-224716_no_interaction")
        flat_run = Path("/data/coding/SoFar/datasets/open6dor_v2/open6dor_v2/task_refine_6dof/behind/Place_the_binder_behind_the_can_on_the_table.__lying_flat/20240824-231004_no_interaction")
        self.assertEqual(parse_orientation_mode_from_task_dir(plug_run), "plug_right")
        self.assertEqual(parse_orientation_mode_from_task_dir(flat_run), "lying_flat")

    def test_classify_task_family(self):
        self.assertEqual(classify_task_family("upright_textual"), "upright_vertical")
        self.assertEqual(classify_task_family("watch_upright"), "upright_vertical")
        self.assertEqual(classify_task_family("lying_flat"), "flat_upside_down_lying_flat")
        self.assertEqual(classify_task_family("lower_rim"), "flat_upside_down_lying_flat")
        self.assertEqual(classify_task_family("plug_right"), "plug_right")
        self.assertEqual(classify_task_family("handle_right"), "plug_right")
        self.assertEqual(classify_task_family("blade_right"), "plug_right")
        self.assertEqual(classify_task_family("cap_right_bottom_left"), "cap_clip_sideways")
        self.assertEqual(classify_task_family("remote_control_forth"), "cap_clip_sideways")
        self.assertEqual(classify_task_family("earpiece_far"), "cap_clip_sideways")

    def test_build_subset_uses_mode_buckets(self):
        dataset_root = Path("/data/coding/SoFar/datasets/open6dor_v2/open6dor_v2/task_refine_6dof")
        task_dirs = [
            dataset_root / "front" / "A.__upright",
            dataset_root / "front" / "B.__upright",
            dataset_root / "front" / "C.__lying_flat",
            dataset_root / "front" / "D.__plug_right",
            dataset_root / "front" / "E.__cap_right_bottom_left",
        ]
        subset = build_eval_subset_from_task_dirs(task_dirs, dataset_root=dataset_root, seed=42, target_total=4)
        self.assertEqual(subset["summary"]["available_total"], 5)
        self.assertEqual(subset["summary"]["selected_total"], 4)
        self.assertLessEqual(subset["summary"]["selected_sample_ids"], 4)
        self.assertIn("upright_vertical", subset["summary"]["selected_family_distribution"])
        self.assertIn("empty_mode_count", subset["summary"])
        self.assertIn("other_family_count", subset["summary"])

    def test_validation_helper_flags_invalid_summary(self):
        validation = validate_eval_subset_summary(
            {
                "empty_mode_count": 1,
                "other_family_count": 0,
                "selected_total": 399,
                "selected_family_distribution": {"upright_vertical": 100},
            }
        )
        self.assertFalse(validation["passed"])
        self.assertIn("empty_mode_count_gt_zero", validation["reasons"])
        self.assertIn("selected_total_not_400", validation["reasons"])


if __name__ == "__main__":
    unittest.main()
