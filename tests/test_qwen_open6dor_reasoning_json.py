import unittest

try:
    from sofar.serve.qwen_inference import _load_open6dor_reasoning_json
except Exception as exc:  # pragma: no cover - import depends on optional local env
    _load_open6dor_reasoning_json = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_load_open6dor_reasoning_json is None, f"qwen_inference import unavailable: {_IMPORT_ERROR}")
class QwenOpen6DORReasoningJsonTest(unittest.TestCase):
    def test_extracts_labeled_xyz_triplet(self):
        payload = _load_open6dor_reasoning_json(
            "The final position is x=0.53, y=0.11, z=0.30."
        )
        self.assertEqual(payload["target_position"], [0.53, 0.11, 0.3])

    def test_extracts_parenthesized_triplet(self):
        payload = _load_open6dor_reasoning_json(
            "Best-effort answer: target center should be approximately (0.37, 0.21, 0.34)."
        )
        self.assertEqual(payload["target_position"], [0.37, 0.21, 0.34])


if __name__ == "__main__":
    unittest.main()
