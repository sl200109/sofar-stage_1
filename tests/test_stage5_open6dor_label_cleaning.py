import unittest

from sofar.serve.stage5_manifest import _resolve_pilot_label


def _payload(mode, vector):
    return {
        "parser_output": {"orientation_mode": mode},
        "geometry_priors": {"part_to_object_vector": vector},
    }


class Stage5Open6DORLabelCleaningTest(unittest.TestCase):
    def test_upright_prefers_semantic_axis_over_geometry(self):
        vector, source, confidence = _resolve_pilot_label("open6dor", _payload("upright", [0.2, -0.1, -0.97]))
        self.assertEqual(source, "orientation_mode.upright_semantic_axis")
        self.assertGreater(vector[2], 0.9)
        self.assertAlmostEqual(confidence, 0.95)

    def test_flat_family_uses_semantic_axes(self):
        cases = [
            ("upside_down", "orientation_mode.upside_down_semantic_axis", -0.9),
            ("lying_flat", "orientation_mode.lying_flat_semantic_axis", -0.9),
            ("lying_sideways", "orientation_mode.lying_sideways_semantic_axis", 0.0),
            ("clip_sideways", "orientation_mode.clip_sideways_semantic_axis", 0.0),
        ]
        for mode, expected_source, expected_z_sign in cases:
            with self.subTest(mode=mode):
                vector, source, _ = _resolve_pilot_label("open6dor", _payload(mode, [0.1, 0.2, 0.3]))
                self.assertEqual(source, expected_source)
                if expected_z_sign < 0:
                    self.assertLess(vector[2], -0.9)
                elif expected_z_sign > 0:
                    self.assertGreater(vector[2], 0.9)
                else:
                    self.assertAlmostEqual(vector[2], 0.0, places=6)

    def test_plug_family_uses_horizontal_semantic_axes(self):
        cases = [
            ("plug_right", "orientation_mode.plug_right_semantic_axis", 1.0),
            ("plug_left", "orientation_mode.plug_left_semantic_axis", -1.0),
            ("handle_right", "orientation_mode.handle_right_semantic_axis", 1.0),
            ("handle_left", "orientation_mode.handle_left_semantic_axis", -1.0),
            ("cap_right_bottom_left", "orientation_mode.cap_right_bottom_left_semantic_axis", 1.0),
            ("cap_left_bottom_right", "orientation_mode.cap_left_bottom_right_semantic_axis", -1.0),
        ]
        for mode, expected_source, expected_x_sign in cases:
            with self.subTest(mode=mode):
                vector, source, _ = _resolve_pilot_label("open6dor", _payload(mode, [0.0, -0.1, -0.9]))
                self.assertEqual(source, expected_source)
                if expected_x_sign > 0:
                    self.assertGreater(vector[0], 0.9)
                else:
                    self.assertLess(vector[0], -0.9)
                self.assertAlmostEqual(vector[2], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
