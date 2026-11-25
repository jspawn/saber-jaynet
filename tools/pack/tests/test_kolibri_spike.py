import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("kolibri_layer_spike", REPO / "tools/kolibri_layer_spike.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KolibriSpikeTest(unittest.TestCase):
    def test_router_bias_selects_but_does_not_change_weights_or_normalize(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ids, weights = MODULE.topk_router(logits, np.array([10.0, 0.0, 0.0], dtype=np.float32), 2)
        np.testing.assert_array_equal(ids, [[0, 2]])
        np.testing.assert_allclose(weights[0, 0], 1.0 / (1.0 + np.exp(-1.0)))
        np.testing.assert_allclose(weights[0, 1], 1.0 / (1.0 + np.exp(-3.0)))
        self.assertGreater(float(weights.sum()), 1.0)

    def test_artifact_writer_preserves_shapes_and_dtypes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.safetensors"
            tensors = {
                "bf16": MODULE.bf16(np.array([[1.0, 2.0]], dtype=np.float32)),
                "f32": np.array([0.25], dtype=np.float32),
                "ids": np.array([[3, 1]], dtype=np.int32),
            }
            MODULE.save_safetensors(path, tensors)
            data = path.read_bytes()
            size = struct.unpack("<Q", data[:8])[0]
            header = json.loads(data[8:8 + size])
            self.assertEqual(header["bf16"]["dtype"], "BF16")
            self.assertEqual(header["bf16"]["shape"], [1, 2])
            self.assertEqual(header["f32"]["dtype"], "F32")
            self.assertEqual(header["ids"]["dtype"], "I32")
            begin, end = header["ids"]["data_offsets"]
            np.testing.assert_array_equal(np.frombuffer(data[8 + size + begin:8 + size + end], dtype=np.int32), [3, 1])


if __name__ == "__main__":
    unittest.main()
