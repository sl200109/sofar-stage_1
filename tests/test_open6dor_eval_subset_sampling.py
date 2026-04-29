from pathlib import Path
import unittest

from sofar.open6dor.eval_subset_sampling import (
    build_eval_subset_from_task_dirs,
    classify_task_family,
    parse_orientation_mode_from_task_dir,
)


class Open6DORSubsetSamplingTest(unittest.TestCase):
    def test_parse_orientation_mode(self):
        task_dir = Path("/data/coding/SoFar/datasets/open6dor_v2/open6dor_v2/task_refine_6dof/front/Place_x.__upright")
        self.assertEqual(parse_orientation_mode_from_task_dir(task_dir), "upright")

    def test_classify_task_family(self):
        self.assertEqual(classify_task_family("upright_textual"), "upright_vertical")
        self.assertEqual(classify_task_family("lying_flat"), "flat_upside_down_lying_flat")
        self.assertEqual(classify_task_family("plug_right"), "plug_right")
        self.assertEqual(classify_task_family("cap_right_bottom_left"), "cap_clip_sideways")

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


if __name__ == "__main__":
    unittest.main()
