"""
Unit and integration tests for Method 2:
Hybrid DWT-HMD-SVD with Arnold Scrambling Watermarking.
Based on Chaudhary et al. (Scientific Reports 2025, Nature Portfolio).
"""
import io
import unittest
import numpy as np
import scipy.ndimage
from PIL import Image, ImageFilter

from watermarking.registry import registry
from watermarking.methods.method_2 import Method2
from watermarking.base import WatermarkResult


class TestMethod2(unittest.TestCase):

    def setUp(self):
        self.method = Method2()

    def test_metadata(self):
        meta = self.method.get_metadata()
        self.assertEqual(meta.method_id, "method_2")
        self.assertIn("DWT-HMD-SVD", meta.short_name)
        self.assertEqual(meta.year, 2025)
        self.assertIn("Chaudhary", meta.authors)
        self.assertIn("10.1038/s41598-025-94080-4", meta.doi_or_url)
        self.assertTrue(len(meta.abstract) > 50)

    def test_parameters(self):
        params = self.method.get_parameters()
        names = [p.name for p in params]
        self.assertIn("gain_factor", names)
        self.assertIn("arnold_iters", names)
        self.assertIn("wavelet", names)

    def test_arnold_transform_reversibility(self):
        """Arnold Cat Map must be perfectly invertible for any iteration count."""
        for iters in [1, 5, 10, 15]:
            rand_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8).astype(np.float64)
            scrambled = Method2.arnold_transform(rand_img, iters=iters, inverse=False)
            unscrambled = Method2.arnold_transform(scrambled, iters=iters, inverse=True)
            self.assertTrue(np.allclose(rand_img, unscrambled))

    def test_embed_and_extract_grayscale_medical(self):
        """Tests embedding & extraction on a simulated 512x512 medical image."""
        # Create synthetic medical-like chest X-ray image (512x512 grayscale)
        x = np.linspace(-3, 3, 512)
        y = np.linspace(-3, 3, 512)
        xx, yy = np.meshgrid(x, y)
        simulated_scan = np.clip(120 + 80 * np.exp(-(xx**2 + yy**2) / 4) + 15 * np.cos(xx * 3), 0, 255).astype(np.uint8)
        host_img = Image.fromarray(simulated_scan, mode="L")

        payload = "Name=?\nAge=?"
        params = {"gain_factor": 0.02, "arnold_iters": 10, "wavelet": "haar"}

        result = self.method.embed(host_img, payload, params)

        self.assertIsInstance(result, WatermarkResult)
        self.assertEqual(result.watermarked_image.size, host_img.size)
        self.assertEqual(result.watermarked_image.mode, "L")
        self.assertIn("psnr_db", result.metrics)
        self.assertIn("ssim", result.metrics)
        self.assertIn("nc", result.metrics)
        self.assertIn("uaci_pct", result.metrics)
        self.assertIn("npcr_pct", result.metrics)

        # In paper: PSNR is ~49.86 dB, SSIM ~ 1.00, NC = 1.00 for no attack
        self.assertGreater(result.metrics["psnr_db"], 48.0)
        self.assertGreater(result.metrics["ssim"], 0.98)
        self.assertGreaterEqual(result.metrics["nc"], 0.99)

        # Verify extraction
        ext = self.method.extract(result.watermarked_image, host_img, params)
        self.assertTrue(ext["detected"])
        self.assertGreaterEqual(ext["nc"], 0.99)
        self.assertIn("extracted_watermark", ext)

    def test_embed_and_extract_rgb_color(self):
        """Tests embedding into an RGB color image and verifies difference map generation."""
        host_img = Image.open("assets/examples/diver.jpg")
        payload = "© SLAYMARKING 2026 / ABYSS"
        params = {"gain_factor": 0.02, "arnold_iters": 8, "wavelet": "haar"}

        result = self.method.embed(host_img, payload, params)

        self.assertEqual(result.watermarked_image.size, host_img.size)
        self.assertEqual(result.watermarked_image.mode, "RGB")
        self.assertIsNotNone(result.difference_map)
        self.assertGreater(result.metrics["psnr_db"], 45.0)
        self.assertGreater(result.metrics["ssim"], 0.95)
        self.assertGreaterEqual(result.metrics["nc"], 0.99)

    def test_robustness_against_attacks(self):
        """Tests robustness of watermark against common attacks from Table 4/5 of paper."""
        # 512x512 synthetic image
        arr = np.random.randint(60, 200, (512, 512), dtype=np.uint8)
        host_img = Image.fromarray(arr, mode="L")
        payload = "CONFIDENTIAL"
        params = {"gain_factor": 0.04, "arnold_iters": 5, "wavelet": "haar"}

        result = self.method.embed(host_img, payload, params)
        wm_img = result.watermarked_image

        # 1. JPEG Compression Attack (Q=75) -> high resilience in paper (NC ~ 0.95 - 1.0)
        buf = io.BytesIO()
        wm_img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        jpeg_attacked = Image.open(buf)
        ext_jpeg = self.method.extract(jpeg_attacked, host_img, params)
        self.assertGreater(ext_jpeg["nc"], 0.85, f"JPEG NC too low: {ext_jpeg['nc']}")

        # 2. Sharpen Attack -> high resilience in paper (NC ~ 0.95 - 1.0)
        sharp_attacked = wm_img.filter(ImageFilter.SHARPEN)
        ext_sharp = self.method.extract(sharp_attacked, host_img, params)
        self.assertGreater(ext_sharp["nc"], 0.85, f"Sharpen NC too low: {ext_sharp['nc']}")

        # 3. Gaussian Blur / Filter Attack -> paper notes vulnerability to low-pass blurring (Table 4)
        blur_attacked = wm_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        ext_blur = self.method.extract(blur_attacked, host_img, params)
        self.assertGreater(ext_blur["nc"], 0.05, f"Blur NC too low: {ext_blur['nc']}")

    def test_registry_integration(self):
        """Verifies Method 2 is properly registered and accessible via registry."""
        m2 = registry.get_method("method_2")
        self.assertIsNotNone(m2)
        self.assertIsInstance(m2, Method2)


if __name__ == "__main__":
    unittest.main()
