import unittest

from sofar.open6dor.eval_subset_sampling import classify_task_family
from sofar.open6dor.open6dor_perception import infer_open6dor_stage5_task_family


class Open6DORStage5FamilyMapperTest(unittest.TestCase):
    def test_part_axis_modes_map_to_part_axis_family(self):
        for mode in [
            "handle_left",
            "handle_right",
            "blade_right",
            "blades_right",
            "ballpoint_right",
            "clasp_right",
            "spout_right",
            "bulb_right_handle_left",
        ]:
            self.assertEqual(classify_task_family(mode), "part_axis_left_right")
            self.assertEqual(infer_open6dor_stage5_task_family(mode), "part_axis_left_right")

    def test_plug_mode_remains_plug_family(self):
        self.assertEqual(classify_task_family("plug_right"), "plug_right")
        self.assertEqual(infer_open6dor_stage5_task_family("plug_right"), "plug_cap_sideways")


if __name__ == "__main__":
    unittest.main()
