"""
Unit and integration tests for Method 4:
Digital Watermarking Model using Honey Encryption and Reversible Cellular Automata.
Based on Xiong et al. (Scientific Reports 2025, Nature Portfolio).
"""
import io
import unittest
import numpy as np
from PIL import Image

from watermarking.registry import registry
from watermarking.methods.method_4 import Method4
from watermarking.base import WatermarkResult


class TestMethod4(unittest.TestCase):

    def setUp(self):
        self.method = Method4()

    def test_metadata(self):
        """Verifies metadata conform to paper citation and registry standards."""
        meta = self.method.get_metadata()
        self.assertEqual(meta.method_id, "method_4")
        self.assertIn("Honey Encryption", meta.short_name)
        self.assertEqual(meta.year, 2025)
        self.assertIn("Xiong", meta.authors)
        self.assertIn("10.1038/s41598-025-05492-1", meta.doi_or_url)
        self.assertGreater(len(meta.abstract), 50)

    def test_parameters(self):
        """Verifies method parameter definitions and default values."""
        params = self.method.get_parameters()
        names = [p.name for p in params]
        self.assertIn("l_rounds", names)
        self.assertIn("he_key_seed", names)
        self.assertIn("bits_per_coeff", names)

    def test_image_validation(self):
        """Verifies strict 512x512 grayscale image requirements matching Method 2."""
        # Non-512x512 image must raise ValueError
        wrong_size = Image.new("L", (256, 256), color=128)
        with self.assertRaises(ValueError) as ctx:
            self.method.embed(wrong_size, "TOP SECRET", {})
        self.assertIn("512×512", str(ctx.exception))

        # RGB image must raise ValueError
        wrong_mode = Image.new("RGB", (512, 512), color=(100, 100, 100))
        with self.assertRaises(ValueError) as ctx:
            self.method.embed(wrong_mode, "TOP SECRET", {})
        self.assertIn("Grayscale", str(ctx.exception))

        # Valid 512x512 Grayscale image must pass validation
        valid_img = Image.new("L", (512, 512), color=128)
        Method4.validate_image_specifications(valid_img, operation="Test")

    def test_initial_propagation_reversibility(self):
        """Initial propagation using XOR (Eq. 4) must be perfectly invertible."""
        for size in [64, 256, 65536]:
            m = np.random.randint(0, 256, size, dtype=np.uint8)
            m_prime = Method4.initial_propagation_forward(m)
            recovered = Method4.initial_propagation_inverse(m_prime)
            self.assertTrue(np.array_equal(m, recovered))

    def test_honey_encryption_reversibility(self):
        """Honey Encryption XOR transformation must be perfectly invertible."""
        data = np.random.randint(0, 256, 1024, dtype=np.uint8)
        for key in [0, 42, 127, 255]:
            encrypted = Method4.honey_encrypt(data, key)
            decrypted = Method4.honey_decrypt(encrypted, key)
            self.assertTrue(np.array_equal(data, decrypted))

    def test_rca_transform_reversibility(self):
        """Reversible Cellular Automata using Table 2 must invert losslessly for any L rounds."""
        h, w = 64, 64
        for l_rounds in [1, 2, 4]:
            msb_planes = np.random.randint(0, 2, (4, h, w), dtype=np.uint8)
            z_matrix = np.random.randint(0, 2, (4, h, w), dtype=np.uint8)

            enc = Method4.rca_transform(msb_planes, z_matrix, l_rounds=l_rounds, inverse=False)
            dec = Method4.rca_transform(enc, z_matrix, l_rounds=l_rounds, inverse=True)

            self.assertTrue(np.array_equal(msb_planes, dec))
            # Test significant diffusion (around ~50% bit divergence)
            diff_ratio = np.mean(msb_planes != enc)
            self.assertGreater(diff_ratio, 0.35)

    def test_iwt2_lossless_reconstruction(self):
        """Integer Haar Wavelet Transform must reconstruct original data losslessly."""
        img_arr = np.random.randint(10, 240, (512, 512), dtype=np.uint8)
        ll, lh, hl, hh = Method4.iwt2_haar(img_arr)
        rec = Method4.iiwt2_haar(ll, lh, hl, hh)
        self.assertTrue(np.array_equal(img_arr, rec))

    def test_embed_and_extract_paper_spec(self):
        """
        Embeds a 256x256 watermark into a 512x512 grayscale cover image and extracts it.
        Checks PSNR, SSIM, MSE, MAE, and 100% exact watermark restoration (NC = 1.0).
        """
        cover_img = Image.open("assets/examples/diver.jpg").convert("L").resize((512, 512))
        payload = "TOP SECRET"
        params = {"l_rounds": 4, "he_key_seed": 42, "bits_per_coeff": 2}

        result = self.method.embed(cover_img, payload, params)

        self.assertIsInstance(result, WatermarkResult)
        self.assertEqual(result.watermarked_image.size, (512, 512))
        self.assertEqual(result.watermarked_image.mode, "L")
        self.assertEqual(result.status, "success")

        # Verify quality metrics align with paper (Table 3: PSNR ~ 37 dB, MSE ~ 12)
        metrics = result.metrics
        self.assertIn("psnr_db", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("ssim", metrics)
        self.assertGreater(metrics["psnr_db"], 35.0)
        self.assertGreater(metrics["ssim"], 0.90)

        # Extraction verification
        ext_res = self.method.extract(result.watermarked_image, params=params)
        self.assertTrue(ext_res["detected"])
        self.assertAlmostEqual(ext_res["nc"], 1.0, places=4)
        self.assertEqual(ext_res["extracted_image"].size, (256, 256))

    def test_entropy_and_correlation_statistics(self):
        """Verifies information entropy and neighbor correlation metrics (Tables 4 & 5)."""
        cover_img = Image.open("assets/examples/wreck.jpg").convert("L").resize((512, 512))
        result = self.method.embed(cover_img, "CONFIDENTIAL", {"l_rounds": 4})

        metrics = result.metrics
        # Neighbor correlation of encrypted watermark should be close to 0 (Table 4)
        self.assertLess(abs(metrics["corr_h"]), 0.2)
        self.assertLess(abs(metrics["corr_v"]), 0.2)

        # For a full grayscale image, encrypted entropy approaches ~8.0 (Table 5)
        gray_wm = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        enc_gray, _, _ = self.method.disturb_watermark(gray_wm, l_rounds=4)
        gray_entropy = self.method.compute_entropy(enc_gray)
        self.assertGreater(gray_entropy, 7.8)


if __name__ == "__main__":
    unittest.main()
