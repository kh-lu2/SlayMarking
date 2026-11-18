"""
Unit and integration tests for Method 5:
Reversible Visible Watermarking and Lossless Recovery of Original Images.
Based on Yongjian Hu and Byeungwoo Jeon, IEEE TCSVT, Vol. 16, No. 11, November 2006.
"""

import unittest
import numpy as np
from PIL import Image

from watermarking.methods.method_5 import Method5
from watermarking.base import WatermarkResult
from watermarking.common import (
    pack_user_key_80bit,
    unpack_user_key_80bit,
    AdaptiveArithmeticCoder,
    create_binary_watermark,
    extract_bit_plane,
    set_bit_plane
)


class TestMethod5(unittest.TestCase):

    def setUp(self):
        self.method = Method5()

    def test_metadata(self):
        """Verifies paper metadata, authors, year, and structured step description."""
        meta = self.method.get_metadata()
        self.assertEqual(meta.method_id, "method_5")
        self.assertIn("Hu", meta.authors)
        self.assertIn("Jeon", meta.authors)
        self.assertEqual(meta.year, 2006)
        self.assertIn("Reversible Visible", meta.domain)
        self.assertEqual(meta.paper_title, "Reversible Visible Watermarking and Lossless Recovery of Original Images")
        self.assertIn("TCSVT", meta.venue)

        # Verify description: brief explanation first, then structured steps
        desc = meta.description
        self.assertTrue(len(desc) > 200)
        self.assertIn("primarily designed", desc.lower())
        self.assertIn("lossless", desc.lower())
        self.assertIn("Step 1:", desc)
        self.assertIn("Step 2:", desc)
        self.assertIn("Step 3:", desc)
        self.assertIn("Step 4:", desc)
        self.assertIn("Step 5:", desc)
        self.assertIn("Step 6:", desc)
        self.assertIn("Step 7:", desc)
        self.assertIn("Step 8:", desc)
        self.assertIn("Step 9:", desc)

        # Verify abstract
        self.assertTrue(len(meta.abstract) > 200)
        self.assertIn("losslessly recover", meta.abstract)

    def test_parameters(self):
        """Verifies configurable parameters match the paper's framework."""
        params = self.method.get_parameters()
        names = [p.name for p in params]
        self.assertIn("bit_plane_rd", names)
        self.assertIn("watermark_size", names)
        self.assertIn("region_origin_y", names)
        self.assertIn("region_origin_x", names)
        self.assertIn("selection_mode", names)
        self.assertIn("user_key", names)

    def test_user_key_pack_unpack_80bit(self):
        """Tests bit-exact packing and unpacking of the 80-bit user key (Section II-D.1)."""
        test_cases = [
            (64, 128, 64, 128, 269, 3, 1, 3, 0),
            (128, 128, 128, 128, 459, 4, 1, 2, 1),
            (64, 128, 0, 0, 117, 3, 1, 5, 0),
            (128, 256, 100, 200, 741, 5, 0, 8, 4),
        ]
        for hw, ww, r0, c0, dc_len, nik, hyb, sp, n in test_cases:
            key_hex = pack_user_key_80bit(hw, ww, r0, c0, dc_len, nik, hyb, sp, n)
            self.assertEqual(len(key_hex), 20, "User key must be 80 bits = 20 hex characters")

            unpacked = unpack_user_key_80bit(key_hex)
            self.assertEqual(unpacked["h_w"], hw)
            self.assertEqual(unpacked["w_w"], ww)
            self.assertEqual(unpacked["r0"], r0)
            self.assertEqual(unpacked["c0"], c0)
            self.assertEqual(unpacked["d_c_len"], dc_len)
            self.assertEqual(unpacked["n_ik"], nik)
            self.assertEqual(unpacked["hybrid"], hyb)
            self.assertEqual(unpacked["s_p"], sp)
            self.assertEqual(unpacked["n"], n)

    def test_adaptive_arithmetic_coder_roundtrip(self):
        """Tests pure Python adaptive order-0 arithmetic coding compression and exact decoding."""
        test_streams = [
            b"Antigravity Reversible Watermarking 2026",
            bytes(np.random.randint(0, 10, 500, dtype=np.uint8)),
            bytes([0] * 200 + [255] * 200 + [128] * 100),
            b"A" * 1000,
        ]
        for stream in test_streams:
            enc = AdaptiveArithmeticCoder.encode(stream)
            dec = AdaptiveArithmeticCoder.decode(enc)
            self.assertEqual(stream, dec, "Arithmetic coder round-trip failed to match original data")

    def test_bit_plane_helpers(self):
        """Tests extract_bit_plane and set_bit_plane utilities."""
        arr = np.array([[128, 64], [1, 254]], dtype=np.uint8)
        # Plane 7 (MSB)
        plane7 = extract_bit_plane(arr, 7)
        self.assertTrue(np.array_equal(plane7, np.array([[1, 0], [0, 1]])))

        # Overwrite plane 7 with 0s
        new_plane7 = np.zeros((2, 2), dtype=np.uint8)
        modified = set_bit_plane(arr, 7, new_plane7)
        self.assertTrue(np.array_equal(extract_bit_plane(modified, 7), new_plane7))
        # Verify lower bits were untouched
        self.assertEqual(int(modified[0, 1]), 64)
        self.assertEqual(int(modified[1, 0]), 1)

    def test_binary_watermark_generator(self):
        """Tests generation of binary logo matrix {0, 1}."""
        logo = create_binary_watermark("SKKU LOGO", height=64, width=128)
        self.assertEqual(logo.shape, (64, 128))
        self.assertTrue(set(np.unique(logo)).issubset({0, 1}))
        # Check watermark has content (not all zero or all one)
        self.assertGreater(np.sum(logo), 50)
        self.assertLess(np.sum(logo), 64 * 128)

    def test_embed_and_lossless_recovery_grayscale_512(self):
        """
        Paper verification: Embed visible watermark on 512x512 image,
        verify visible watermarking in region R, then perform 100% lossless
        recovery with the 80-bit user key (MSE == 0.0, PSNR == inf).
        """
        # Create a synthetic 512x512 host image with natural gradient & texture
        y, x = np.mgrid[0:512, 0:512]
        arr = ((np.sin(x / 14.0) + np.cos(y / 14.0) + 2.0) * 55.0).astype(np.uint8)
        host_img = Image.fromarray(arr, mode="L")

        params = {
            "bit_plane_rd": "7 (MSB - Heavy Tag)",
            "watermark_size": "64x128",
            "region_origin_y": 64,
            "region_origin_x": 64,
            "selection_mode": "Hybrid (RPS + Step)",
        }

        # 1. Embed Reversible Visible Watermark
        result: WatermarkResult = self.method.embed(
            image=host_img,
            watermark_data="CONFIDENTIAL\nMEDICAL TAG",
            params=params
        )

        self.assertEqual(result.status, "success")
        self.assertIsInstance(result.watermarked_image, Image.Image)
        self.assertEqual(result.watermarked_image.size, (512, 512))

        # Check metrics
        user_key = result.extracted_payload
        self.assertIsNotNone(user_key)
        self.assertEqual(len(user_key), 20)
        self.assertIn("psnr_without_r_db", result.metrics)
        self.assertIn("e_d_percent", result.metrics)
        self.assertIn("e_n_percent", result.metrics)

        # Watermarked image should visually differ from original in region R
        wm_arr = np.asarray(result.watermarked_image, dtype=np.uint8)
        diff_r = np.max(np.abs(arr[64:128, 64:192].astype(float) - wm_arr[64:128, 64:192].astype(float)))
        self.assertGreater(diff_r, 0, "Watermark region R must be visually modified")

        # 2. Extract / Remove Visible Watermark and Losslessly Recover Original Image
        rec_res = self.method.extract(
            watermarked_image=result.watermarked_image,
            original_image=host_img,
            params={"user_key": user_key}
        )

        self.assertEqual(rec_res["status"], "success")
        self.assertTrue(rec_res["lossless_verified"], "Reconstructed image must be 100% bit-exact to original!")

        # Verify bit-for-bit identity: max error is 0.0
        rec_arr = np.asarray(rec_res["recovered_image"], dtype=np.uint8)
        max_err = np.max(np.abs(arr.astype(float) - rec_arr.astype(float)))
        self.assertEqual(max_err, 0.0, f"Expected 0.0 reconstruction error, got {max_err}")

    def test_second_msb_transparency_mode(self):
        """
        Tests N_RD = 6 (second MSB plane) for lighter watermark transparency
        as evaluated in Section IV and Fig. 2 of the paper.
        """
        y, x = np.mgrid[0:256, 0:256]
        arr = ((np.sin(x / 10.0) + np.cos(y / 10.0) + 2.0) * 55.0).astype(np.uint8)
        host_img = Image.fromarray(arr, mode="L")

        params = {
            "bit_plane_rd": "6 (2nd MSB - Light Tag)",
            "watermark_size": "64x128",
            "region_origin_y": 32,
            "region_origin_x": 32,
            "selection_mode": "Hybrid (RPS + Step)",
        }

        result = self.method.embed(host_img, "SLAY 2026", params)
        self.assertEqual(result.status, "success")

        # Test recovery
        rec_res = self.method.extract(
            watermarked_image=result.watermarked_image,
            original_image=host_img,
            params={"user_key": result.extracted_payload}
        )
        self.assertTrue(rec_res["lossless_verified"])

    def test_key_security_wrong_key(self):
        """
        Verifies that attempting recovery with a wrong user key fails or does not
        losslessly reconstruct the image.
        """
        arr = np.full((256, 256), 120, dtype=np.uint8)
        # Add smooth texture
        for r in range(256):
            arr[r, :] = (arr[r, :] + r // 4).astype(np.uint8)
        host_img = Image.fromarray(arr, mode="L")

        result = self.method.embed(
            host_img,
            "SECRET EMBLEM",
            {"watermark_size": "64x128", "region_origin_y": 32, "region_origin_x": 32}
        )
        self.assertEqual(result.status, "success")

        # Corrupt the 80-bit user key
        corrupted_key = "00000000000000000000"
        rec_res = self.method.extract(
            watermarked_image=result.watermarked_image,
            original_image=host_img,
            params={"user_key": corrupted_key}
        )
        # Should either fail with error status or fail lossless verification
        self.assertFalse(rec_res.get("lossless_verified", False))

    def test_rgb_image_support(self):
        """Tests that Method 5 handles RGB images seamlessly."""
        color_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        y, x = np.mgrid[0:256, 0:256]
        color_arr[:, :, 0] = ((np.sin(x / 12.0) + 1.0) * 110.0).astype(np.uint8)
        color_arr[:, :, 1] = ((np.cos(y / 12.0) + 1.0) * 110.0).astype(np.uint8)
        color_arr[:, :, 2] = 180
        color_img = Image.fromarray(color_arr, mode="RGB")

        result = self.method.embed(
            color_img,
            "RGB WATERMARK",
            {"watermark_size": "64x128", "region_origin_y": 32, "region_origin_x": 32}
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.watermarked_image.mode, "RGB")

        rec_res = self.method.extract(
            result.watermarked_image,
            original_image=color_img,
            params={"user_key": result.extracted_payload}
        )
        self.assertEqual(rec_res["status"], "success")
        self.assertTrue(rec_res["lossless_verified"])


if __name__ == "__main__":
    unittest.main()
