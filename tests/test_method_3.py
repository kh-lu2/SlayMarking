"""
Unit and integration tests for Method 3:
Fragile Watermarking Scheme for Image Tamper Detection and Recovery Using Hybrid Transform.
Based on Prasanth Vaidya S, N V P S Rajesh Kandala, Chandra Mouli PVSSR, JVCIR 2023.
"""

import unittest
import numpy as np
from PIL import Image

from watermarking.methods.method_3 import Method3
from watermarking.base import WatermarkResult


class TestMethod3(unittest.TestCase):

    def setUp(self):
        self.method = Method3()

    def test_metadata(self):
        meta = self.method.get_metadata()
        self.assertEqual(meta.method_id, "method_3")
        self.assertIn("Prasanth Vaidya", meta.authors)
        self.assertIn("Rajesh Kandala", meta.authors)
        self.assertEqual(meta.year, 2023)
        self.assertIn("DWT", meta.domain)
        self.assertIn("Schur", meta.domain)
        self.assertIn("Fragile", meta.title)
        self.assertTrue(len(meta.abstract) > 100)

    def test_parameters(self):
        params = self.method.get_parameters()
        names = [p.name for p in params]
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertIn("secret_key", names)
        self.assertIn("attack", names)
        self.assertIn("tamper_size", names)

    def test_schur_abb_tamper_sensitivity(self):
        # 4x4 block with even numbers (LSB = 0)
        blk = np.array([
            [120, 140, 160, 180],
            [130, 150, 170, 190],
            [110, 134, 156, 178],
            [104, 126, 148, 168]
        ], dtype=np.uint8) & 0xFE

        # 1. Determinism: identical blocks yield identical 16 ABB bits
        abb1 = self.method._compute_schur_abb(blk, secret_key=42)
        abb2 = self.method._compute_schur_abb(blk, secret_key=42)
        self.assertEqual(abb1, abb2)
        self.assertEqual(len(abb1), 16)

        # 2. Key sensitivity: different keys yield different signatures
        abb_diff_key = self.method._compute_schur_abb(blk, secret_key=999)
        self.assertNotEqual(abb1, abb_diff_key)

        # 3. Tamper sensitivity: modifying any pixel alters the ABB
        tampered = blk.copy()
        tampered[1, 2] = np.uint8((int(tampered[1, 2]) + 16) & 0xFE)
        abb_tampered = self.method._compute_schur_abb(tampered, secret_key=42)
        self.assertNotEqual(abb1, abb_tampered)

    def test_recovery_bits_roundtrip(self):
        # Create 4x4 block with known 2x2 sub-block values
        blk = np.zeros((4, 4), dtype=np.uint8)
        blk[0:2, 0:2] = 50
        blk[0:2, 2:4] = 100
        blk[2:4, 0:2] = 150
        blk[2:4, 2:4] = 200

        bits = self.method._compute_recovery_bits(blk)
        self.assertEqual(len(bits), 32)

        decoded_means = self.method._decode_recovery_bits(bits)
        self.assertEqual(decoded_means, [50, 100, 150, 200])

    def test_embed_and_extract_unattacked_512(self):
        # Create standard 512x512 synthetic test image
        y, x = np.mgrid[0:512, 0:512]
        arr = ((np.sin(x / 16.0) + np.cos(y / 16.0) + 2.0) * 60.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        result: WatermarkResult = self.method.embed(
            image=img,
            watermark_data="EMBLEM",
            params={"alpha": 0.98, "beta": 0.04, "secret_key": 42, "attack": "None"}
        )

        self.assertIsInstance(result.watermarked_image, Image.Image)
        self.assertEqual(result.watermarked_image.size, (512, 512))
        self.assertEqual(result.status, "success")

        # Check PSNR in high fidelity range (paper reports ~39 to 49 dB)
        psnr = result.metrics.get("psnr_db", 0.0)
        self.assertGreater(psnr, 35.0, f"Expected PSNR > 35 dB, got {psnr}")

        # Check SSIM (paper reports ~0.90 to 0.99)
        ssim = result.metrics.get("ssim", 0.0)
        self.assertGreater(ssim, 0.90, f"Expected SSIM > 0.90, got {ssim}")

        # Check watermark NCC (paper reports ~0.94 to 0.99)
        ncc = result.metrics.get("watermark_ncc", 0.0)
        self.assertGreater(ncc, 0.90, f"Expected NCC > 0.90, got {ncc}")

        # Verify tamper detection on pristine watermarked image: 0 false positives!
        tamper_res = self.method.detect_tamper(result.watermarked_image, secret_key=42)
        self.assertFalse(tamper_res["is_tampered"], "Pristine watermarked image should have 0 false tampered blocks")
        self.assertEqual(tamper_res["tampered_blocks_count"], 0)

    def test_tamper_detection_and_localization(self):
        # Generate 512x512 image
        y, x = np.mgrid[0:512, 0:512]
        arr = ((np.sin(x / 20.0) + np.cos(y / 20.0) + 2.0) * 60.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")

        # Embed watermark
        res = self.method.embed(
            image=img,
            watermark_data="SLAY",
            params={"alpha": 0.98, "beta": 0.04, "secret_key": 42, "attack": "None"}
        )
        wm_img = res.watermarked_image

        # Simulate tampering attack: replace a 64x64 square in the center with white box
        tampered_img = self.method.simulate_attack(wm_img, attack_type="Center Crop", tamper_size=64)

        # Detect tampering
        det_res = self.method.detect_tamper(tampered_img, secret_key=42)
        self.assertTrue(det_res["is_tampered"])
        self.assertGreater(det_res["tampered_blocks_count"], 100)

        # Check localized image (tampered region pixels set to 0)
        loc_img = det_res["localized_image"]
        loc_arr = np.asarray(loc_img.convert("L"))
        # Center region should have blacked out pixels (mean close to 0)
        cx, cy = 256, 256
        center_patch = loc_arr[cy - 20:cy + 20, cx - 20:cx + 20]
        self.assertEqual(np.mean(center_patch), 0.0, "Localized image should set tampered region to 0")

    def test_tamper_recovery(self):
        # Generate 512x512 image
        y, x = np.mgrid[0:512, 0:512]
        arr = ((np.sin(x / 16.0) + np.cos(y / 16.0) + 2.0) * 60.0).astype(np.uint8)
        orig_img = Image.fromarray(arr, mode="L")

        res = self.method.embed(
            orig_img,
            "KEY",
            {"alpha": 0.98, "beta": 0.04, "secret_key": 42, "attack": "None"}
        )
        wm_img = res.watermarked_image

        # Tamper 64x64 square
        tampered_img = self.method.simulate_attack(wm_img, attack_type="Center Crop", tamper_size=64)
        det_res = self.method.detect_tamper(tampered_img, secret_key=42)

        # Recover image using recovery bits
        rec_res = self.method.recover(
            tampered_image=tampered_img,
            tamper_mask=det_res["tamper_mask"],
            secret_key=42,
            original_image=orig_img
        )

        self.assertGreater(rec_res["recovery_rate"], 90.0)
        # Recovered PSNR should be > 35 dB (paper Table 3 reports 35-50 dB)
        self.assertGreater(rec_res["recovered_psnr"], 35.0)
        self.assertGreater(rec_res["recovered_ssim"], 0.90)

    def test_attack_simulation_in_embed(self):
        # Verify that setting attack in embed returns a tamper-recovered result
        img = Image.new("RGB", (256, 256), color=(60, 120, 200))
        res = self.method.embed(
            image=img,
            watermark_data="ATTACK_TEST",
            params={"alpha": 0.98, "beta": 0.04, "secret_key": 42, "attack": "Copy-Move", "tamper_size": 48}
        )
        self.assertEqual(res.status, "tamper_recovered")
        self.assertIn("tampered_blocks", res.metrics)
        self.assertIn("recovered_psnr_db", res.metrics)
        self.assertGreater(res.metrics["recovered_psnr_db"], 30.0)

    def test_arbitrary_image_dimension(self):
        # Test non-512 image size (e.g. 300x200)
        img = Image.new("RGB", (300, 200), color=(40, 80, 160))
        res = self.method.embed(
            image=img,
            watermark_data="NON_SQUARE",
            params={"alpha": 0.98, "beta": 0.04, "secret_key": 100, "attack": "None"}
        )
        self.assertEqual(res.watermarked_image.size, (300, 200))
        self.assertEqual(res.status, "success")


if __name__ == "__main__":
    unittest.main()
