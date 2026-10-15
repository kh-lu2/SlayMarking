"""
Unit and integration tests for Method 1 (DWT-SVD Semi-Binary watermarking).
Based on Morteza Makhloghi et al., IEEE 2011.
"""

import unittest
import numpy as np
from PIL import Image

from watermarking.methods.method_1 import Method1
from watermarking.base import WatermarkResult


class TestMethod1(unittest.TestCase):

    def setUp(self):
        self.method = Method1()

    def test_metadata(self):
        meta = self.method.get_metadata()
        self.assertEqual(meta.method_id, "method_1")
        self.assertIn("Makhloghi", meta.authors)
        self.assertEqual(meta.year, 2011)
        self.assertIn("DWT-SVD", meta.domain)
        self.assertTrue(len(meta.abstract) > 50)

    def test_parameters(self):
        params = self.method.get_parameters()
        param_names = [p.name for p in params]
        self.assertIn("strength", param_names)
        self.assertIn("dwt_level", param_names)
        self.assertIn("wavelet", param_names)

    def test_algo1_and_algo2_roundtrip(self):
        test_values = [0.0, 1.0, -1.0, 13.75, -45.125, 254.99, 0.333, 128.0]
        for val in test_values:
            w = self.method._algo1_to_semi_binary(val, min_len=10)
            reconstructed = self.method._algo2_from_semi_binary(w)
            self.assertAlmostEqual(
                val,
                reconstructed,
                places=6,
                msg=f"Failed round-trip for value {val}"
            )

    def test_haar_dwt_roundtrip(self):
        np.random.seed(42)
        test_img = np.random.rand(128, 128) * 255.0
        ll_k, details = self.method._wavedec2_haar(test_img, level=3)
        reconstructed = self.method._waverec2_haar(ll_k, details)
        max_diff = np.max(np.abs(test_img - reconstructed))
        self.assertLess(max_diff, 1e-10)

    def test_embed_and_extract_grayscale_512(self):
        # Create 512x512 gradient/texture image
        y, x = np.mgrid[0:512, 0:512]
        arr = ((np.sin(x / 16.0) + np.cos(y / 16.0) + 2.0) * 60.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        result: WatermarkResult = self.method.embed(
            image=img,
            watermark_data="UOK",
            params={"strength": 1.0, "dwt_level": 5}
        )

        self.assertIsInstance(result.watermarked_image, Image.Image)
        self.assertEqual(result.watermarked_image.size, img.size)
        self.assertEqual(result.status, "success")

        # Check PSNR is above 50 dB as reported in paper (Table 1: 51.76 to 54.94 dB)
        psnr = result.metrics.get("psnr_db", 0.0)
        self.assertGreater(psnr, 50.0, f"Expected PSNR > 50 dB, got {psnr}")

        # Check watermark correlation rho is high (> 0.95 in unattacked condition)
        rho = result.metrics.get("correlation_rho", 0.0)
        self.assertGreater(rho, 0.95, f"Expected watermark correlation rho > 0.95, got {rho}")

        # Check difference map
        self.assertIsInstance(result.difference_map, Image.Image)
        self.assertEqual(result.difference_map.size, img.size)

    def test_embed_and_extract_rgb(self):
        # Create test RGB image
        img = Image.new("RGB", (256, 256), color=(40, 100, 180))
        result = self.method.embed(
            image=img,
            watermark_data="SLAY",
            params={"strength": 1.0, "dwt_level": 4}
        )

        self.assertEqual(result.watermarked_image.size, img.size)
        self.assertEqual(result.watermarked_image.mode, "RGB")
        self.assertGreater(result.metrics["psnr_db"], 40.0)

        # Test blind extraction
        ext = self.method.extract(result.watermarked_image)
        self.assertTrue(ext["detected"])
        self.assertGreater(ext["correlation"], 0.70)

    def test_arbitrary_image_size(self):
        # Non-power-of-two size
        img = Image.new("RGB", (347, 219), color=(20, 60, 110))
        result = self.method.embed(
            image=img,
            watermark_data="TEST",
            params={"strength": 1.0, "dwt_level": 3}
        )
        self.assertEqual(result.watermarked_image.size, (347, 219))


if __name__ == "__main__":
    unittest.main()
