import builtins
import os
import sys
import tempfile
import types
import unittest
from urllib.error import URLError
from unittest.mock import patch

import cv2
import numpy as np

import engine


class FakeLoFTR:
    def __init__(self, *args, **kwargs):
        raise URLError("SSL certificate verify failed")


class TestLoFTRDownloadFailure(unittest.TestCase):
    def test_get_matcher_raises_clear_runtime_error(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "kornia.feature":
                module = types.ModuleType("kornia.feature")
                module.LoFTR = FakeLoFTR
                sys.modules[name] = module
                return module
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "LoFTR pretrained weights could not be downloaded"):
                engine._get_matcher()

    def test_run_luna_align_uses_sift_fallback_when_loftr_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.png")
            target_path = os.path.join(tmpdir, "target.png")

            rng = np.random.default_rng(42)
            ref = np.zeros((512, 512), dtype=np.uint8)
            target = np.zeros((512, 512), dtype=np.uint8)

            for _ in range(120):
                x = int(rng.integers(30, 480))
                y = int(rng.integers(30, 480))
                radius = int(rng.integers(8, 18))
                color = int(rng.integers(80, 220))
                cv2.circle(ref, (x, y), radius, color, -1)
                cv2.circle(target, (x + 12, y + 10), radius, color, -1)

            cv2.imwrite(ref_path, ref)
            cv2.imwrite(target_path, target)

            with patch.object(engine, "_get_matcher", side_effect=RuntimeError("LoFTR unavailable")):
                aligned, blend, matches_plot, inliers, ratio, rmse = engine.run_luna_align(ref_path, target_path)

            self.assertIsNotNone(aligned)
            self.assertIsNotNone(blend)
            self.assertIsNotNone(matches_plot)
            self.assertGreater(inliers, 0)
            self.assertGreaterEqual(ratio, 0.0)
            self.assertGreaterEqual(rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
