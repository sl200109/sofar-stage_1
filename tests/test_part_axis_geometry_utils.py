import unittest

import numpy as np

from sofar.analysis.part_axis_geometry_utils import verify_part_axis_geometry


class PartAxisGeometryUtilsTest(unittest.TestCase):
    def test_right_mode_consistent(self):
        result = verify_part_axis_geometry(
            object_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
            part_points=np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32),
            orientation_mode="handle_right",
        )
        self.assertEqual(result["part_axis_geometry_status"], "consistent")
        self.assertGreater(result["part_axis_geometry_score"], 0.9)

    def test_left_mode_consistent(self):
        result = verify_part_axis_geometry(
            object_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
            part_points=np.array([[-1.0, 0.0, 0.0], [-1.2, 0.0, 0.0]], dtype=np.float32),
            orientation_mode="handle_left",
        )
        self.assertEqual(result["part_axis_geometry_status"], "consistent")
        self.assertGreater(result["part_axis_geometry_score"], 0.9)

    def test_missing_points(self):
        result = verify_part_axis_geometry(
            object_points=np.zeros((0, 3), dtype=np.float32),
            part_points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            orientation_mode="handle_right",
        )
        self.assertEqual(result["part_axis_geometry_status"], "missing_points")
        self.assertIsNone(result["part_axis_geometry_score"])

    def test_uncertain_or_inconsistent_does_not_crash(self):
        result = verify_part_axis_geometry(
            object_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            part_points=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            orientation_mode="handle_right",
        )
        self.assertIn(result["part_axis_geometry_status"], {"uncertain", "inconsistent"})


if __name__ == "__main__":
    unittest.main()
