"""
Unit and integration tests for Method 6:
Generalized Hybrid DWT-HMD-SVD with Arnold Scrambling (Color Luminance & Subband Choice).
"""
import io
import os
import unittest
import numpy as np
from PIL import Image, ImageFilter

from watermarking.registry import registry
from watermarking.methods.method_6 import Method6
from watermarking.base import WatermarkResult


class TestMethod6(unittest.TestCase):

    def setUp(self):
        self.method = Method6()

    def test_metadata(self):
        meta = self.method.get_metadata()
        self.assertEqual(meta.method_id, "method_6")
        self.assertIn("DWT-HMD-SVD", meta.short_name)
        self.assertEqual(meta.year, 2025)
        self.assertIn("Chaudhary", meta.authors)
        self.assertTrue(len(meta.abstract) > 50)

    def test_parameters(self):
        params = self.method.get_parameters()
        names = [p.name for p in params]
        self.assertIn("gain_factor", names)
        self.assertIn("subband", names)
        self.assertIn("arnold_iters", names)
        self.assertIn("wavelet", names)

        # Check subband parameter choices
        subband_param = next(p for p in params if p.name == "subband")
        self.assertEqual(subband_param.param_type, "choice")
        self.assertEqual(subband_param.default, "LL")
        self.assertEqual(set(subband_param.choices), {"LL", "LH", "HL", "HH"})

    def test_arnold_transform_reversibility(self):
        """Arnold Cat Map must be perfectly invertible for any iteration count."""
        for iters in [1, 5, 10, 15]:
            rand_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8).astype(np.float64)
            scrambled = Method6.arnold_transform(rand_img, iters=iters, inverse=False)
            unscrambled = Method6.arnold_transform(scrambled, iters=iters, inverse=True)
            self.assertTrue(np.allclose(rand_img, unscrambled))

    def test_colored_image_embedding_and_extraction(self):
        """Tests that colored RGB images have their color preserved and watermark embedded in brightness."""
        rgb_img = Image.open("assets/examples/bacteria.jpg")  # 512x512 RGB
        self.assertEqual(rgb_img.mode, "RGB")

        payload = "ColorTest"
        params = {"gain_factor": 0.03, "arnold_iters": 5, "wavelet": "haar", "subband": "LL"}

        result = self.method.embed(rgb_img, payload, params)

        self.assertIsInstance(result, WatermarkResult)
        self.assertEqual(result.watermarked_image.mode, "RGB")
        self.assertEqual(result.watermarked_image.size, rgb_img.size)
        self.assertGreater(result.metrics["psnr_db"], 40.0)
        self.assertGreater(result.metrics["ssim"], 0.95)
        self.assertGreaterEqual(result.metrics["nc"], 0.98)

        # Extraction test
        ext = self.method.extract(result.watermarked_image, rgb_img, params)
        self.assertTrue(ext["detected"])
        self.assertGreaterEqual(ext["nc"], 0.98)
        self.assertIn("extracted_watermark", ext)

    def test_all_subbands(self):
        """Tests embedding and extraction across all selectable subbands (LL, LH, HL, HH)."""
        rgb_img = Image.open("assets/examples/bacteria.jpg")  # 512x512 RGB
        payload = "SubbandTest"

        for subband in ["LL", "LH", "HL", "HH"]:
            with self.subTest(subband=subband):
                params = {"gain_factor": 0.03, "arnold_iters": 5, "wavelet": "haar", "subband": subband}
                result = self.method.embed(rgb_img, payload, params)

                self.assertEqual(result.watermarked_image.size, rgb_img.size)
                self.assertGreater(result.metrics["psnr_db"], 35.0)
                self.assertGreater(result.metrics["ssim"], 0.90)

                ext = self.method.extract(result.watermarked_image, rgb_img, params)
                self.assertTrue(ext["detected"], f"Extraction failed for subband {subband}")
                self.assertGreaterEqual(ext["nc"], 0.90, f"NC too low for subband {subband}: {ext['nc']}")

    def test_rgba_image_preservation(self):
        """Tests that RGBA images maintain alpha channel integrity."""
        rgba_img = Image.open("assets/examples/boat_near_shore.png")  # 512x512 RGBA
        self.assertEqual(rgba_img.mode, "RGBA")

        params = {"gain_factor": 0.02, "arnold_iters": 5, "wavelet": "haar", "subband": "LH"}
        result = self.method.embed(rgba_img, "AlphaPayload", params)

        self.assertEqual(result.watermarked_image.mode, "RGBA")
        self.assertEqual(result.watermarked_image.size, rgba_img.size)

        # Verify alpha channel was preserved
        orig_alpha = np.array(rgba_img.split()[-1])
        wm_alpha = np.array(result.watermarked_image.split()[-1])
        self.assertTrue(np.array_equal(orig_alpha, wm_alpha))

    def test_arbitrary_dimensions(self):
        """Tests that non-512x512 rectangular images work seamlessly."""
        rect_img = Image.open("assets/examples/blobfish.jpg")  # 540x360 RGB
        self.assertNotEqual(rect_img.size, (512, 512))

        params = {"gain_factor": 0.02, "arnold_iters": 5, "wavelet": "haar", "subband": "HL"}
        result = self.method.embed(rect_img, "RectPayload", params)

        self.assertEqual(result.watermarked_image.size, rect_img.size)
        self.assertEqual(result.watermarked_image.mode, "RGB")
        self.assertGreater(result.metrics["psnr_db"], 40.0)

        ext = self.method.extract(result.watermarked_image, rect_img, params)
        self.assertTrue(ext["detected"])
        self.assertGreaterEqual(ext["nc"], 0.95)

    def test_grayscale_image(self):
        """Tests that grayscale images continue to work seamlessly."""
        gray_img = Image.open("assets/examples/skull.jpg")  # 512x512 L
        params = {"gain_factor": 0.02, "arnold_iters": 5, "wavelet": "haar", "subband": "HH"}
        result = self.method.embed(gray_img, "GrayPayload", params)

        self.assertEqual(result.watermarked_image.mode, "L")
        self.assertEqual(result.watermarked_image.size, gray_img.size)

        ext = self.method.extract(result.watermarked_image, gray_img, params)
        self.assertTrue(ext["detected"])
        self.assertGreaterEqual(ext["nc"], 0.95)

    def test_all_existing_asset_images_support(self):
        """Verifies that all images in assets/examples work without error."""
        examples_dir = os.path.join("assets", "examples")
        image_files = [
            f for f in os.listdir(examples_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.assertGreaterEqual(len(image_files), 15)

        params = {"gain_factor": 0.02, "arnold_iters": 5, "wavelet": "haar", "subband": "LL"}

        for img_file in image_files:
            img_path = os.path.join(examples_dir, img_file)
            img = Image.open(img_path)
            # Embed watermark
            result = self.method.embed(img, "TestAll", params)
            self.assertEqual(result.watermarked_image.size, img.size)
            self.assertEqual(result.status, "success")

    def test_robustness_against_attacks(self):
        """Tests robustness of watermark against common attacks."""
        host_img = Image.open("assets/examples/bacteria.jpg")
        payload = "AttackTest"
        params = {"gain_factor": 0.04, "arnold_iters": 5, "wavelet": "haar", "subband": "LL"}

        result = self.method.embed(host_img, payload, params)
        wm_img = result.watermarked_image

        # 1. JPEG Compression Attack (Q=75)
        buf = io.BytesIO()
        wm_img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        jpeg_attacked = Image.open(buf)
        ext_jpeg = self.method.extract(jpeg_attacked, host_img, params)
        self.assertGreater(ext_jpeg["nc"], 0.80, f"JPEG NC too low: {ext_jpeg['nc']}")

        # 2. Sharpen Attack
        sharp_attacked = wm_img.filter(ImageFilter.SHARPEN)
        ext_sharp = self.method.extract(sharp_attacked, host_img, params)
        self.assertGreater(ext_sharp["nc"], 0.85, f"Sharpen NC too low: {ext_sharp['nc']}")

    def test_registry_integration(self):
        """Verifies Method 6 is properly registered and accessible via registry."""
        m6 = registry.get_method("method_6")
        self.assertIsNotNone(m6)
        self.assertIsInstance(m6, Method6)


if __name__ == "__main__":
    unittest.main()
