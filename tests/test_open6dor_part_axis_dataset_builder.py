import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sofar.analysis.build_open6dor_part_axis_stage5_dataset import build_dataset, parse_split_ratio


def _write_task(dataset_root, name, mode, *, with_part=True):
    task_dir = dataset_root / "task_refine_6dof" / "behind" / f"Place_the_{name}.__{mode}" / "20240101_no_interaction"
    stage4 = task_dir / "output" / "stage4"
    stage4.mkdir(parents=True, exist_ok=True)
    (stage4 / "point_data_cache.json").write_text(json.dumps({"target_object": name}), encoding="utf-8")
    np.savez_compressed(stage4 / "object_points.npz", points=np.array([[0, 0, 0, 0, 0, 0], [0, 0.1, 0, 0, 0, 0]], dtype=np.float32))
    if with_part:
        np.savez_compressed(stage4 / "part_points.npz", points=np.array([[1, 0, 0, 0, 0, 0], [1.1, 0, 0, 0, 0, 0]], dtype=np.float32))
    return task_dir


class Open6DORPartAxisDatasetBuilderTest(unittest.TestCase):
    def test_filters_part_axis_modes_and_counts_missing_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "open6dor_v2"
            handle = _write_task(dataset_root, "mug", "handle_right")
            blade = _write_task(dataset_root, "knife", "blade_right")
            missing = _write_task(dataset_root, "cup", "handle_right", with_part=False)
            plug = _write_task(dataset_root, "usb", "plug_right")
            upright = _write_task(dataset_root, "bottle", "upright")
            entries = [
                str(handle.relative_to(dataset_root)),
                str(blade.relative_to(dataset_root)),
                str(missing.relative_to(dataset_root)),
                str(plug.relative_to(dataset_root)),
                str(upright.relative_to(dataset_root)),
            ]

            payload = build_dataset(dataset_root, entries, seed=42, split_ratio=parse_split_ratio("0.8,0.1,0.1"))
            summary = payload["summary"]

            self.assertEqual(summary["total_records_seen"], 5)
            self.assertEqual(summary["total_samples"], 2)
            self.assertEqual(summary["missing_part_points_count"], 1)
            self.assertEqual(summary["excluded_non_part_axis_count"], 2)
            self.assertEqual(summary["mode_distribution"], {"blade_right": 1, "handle_right": 1})
            self.assertEqual({row["orientation_mode"] for row in payload["samples"]}, {"handle_right", "blade_right"})


if __name__ == "__main__":
    unittest.main()
