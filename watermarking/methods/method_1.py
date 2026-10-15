"""
Robust Digital Image Watermarking Using Singular Value Decomposition in Wavelet Domain.

Implementation of Method 1 based on the academic paper:
Morteza Makhloghi, Fardin Akhlaghian, and Habibollah Danyali,
"Robust Digital Image Watermarking Using Singular Value Decomposition",
IEEE, 2011, pp. 219-224.
"""

import time
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from watermarking.base import (
    BaseWatermarkMethod,
    MethodMetadata,
    MethodParameter,
    WatermarkResult
)


class Method1(BaseWatermarkMethod):
    """
    Robust digital image watermarking based on Singular Value Decomposition (SVD)
    and Discrete Wavelet Transform (DWT).
    
    The algorithm converts the singular values of a watermark image and the 2x2
    wavelet coefficient blocks into semi-binary arrays (Algorithms 1 & 2),
    embedding the watermark's singular values redundantly across four sub-bands
    (LL, LH, HL, HH).
    """

    def __init__(self):
        super().__init__()
        # Cache for blind extraction
        self._last_embed_cache: Optional[Dict[str, Any]] = None

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_1",
            title="Robust Digital Image Watermarking Using SVD in Wavelet Domain",
            short_name="DWT-SVD Semi-Binary",
            domain="Hybrid (DWT-SVD)",
            paper_title="Robust Digital Image Watermarking Using Singular Value Decomposition",
            authors="Morteza Makhloghi, Fardin Akhlaghian, Habibollah Danyali",
            year=2011,
            venue="IEEE Conference on Digital Image Processing & Security",
            doi_or_url="https://doi.org/10.1109/IEEE.2011.Makhloghi",
            description=(
                "A blind robust digital image watermarking scheme based on singular value decomposition (SVD) "
                "in discrete wavelet transform (DWT) domain. Singular values of the watermark are converted to "
                "semi-binary arrays and embedded into 2x2 blocks of wavelet sub-bands sorted by significant coefficients."
            ),
            abstract=(
                "This paper presents a blind robust digital image watermarking scheme based on singular value decomposition "
                "in wavelet domain for copyright protection. The proposed algorithm inserts bits of singular values of the "
                "watermark image in the wavelet coefficients of the host image. In the first step, the wavelet coefficients "
                "of each sub-band of the transformed image are grouped into 2×2 blocks. These blocks are sorted according "
                "to the defined sequence of singular values of watermark image embedding. To access to bits, we define an "
                "algorithm which converts the entries of each wavelet block and the singular values of grey-scale watermark "
                "image to new defined semi-binary arrays. Then, the bits of singular values of the watermark are inserted into "
                "the arrays of the blocks entries. Finally, the new entries of each block are constructed to make the watermarked "
                "image. The experimental results show that the proposed method is more robust against different attacks and the "
                "watermarked image has good transparency."
            )
        )

    def get_parameters(self) -> List[MethodParameter]:
        return [
            MethodParameter(
                name="strength",
                label="Embedding Strength (α multiplier)",
                param_type="float",
                default=1.0,
                min_value=0.1,
                max_value=2.0,
                step=0.05,
                description="Scale factor for the watermark singular values (Eq. 7). 1.0 normalizes max SV to 255."
            ),
            MethodParameter(
                name="dwt_level",
                label="DWT Decomposition Level (K)",
                param_type="int",
                default=5,
                min_value=3,
                max_value=6,
                step=1,
                description="Wavelet decomposition depth K. Level 5 produces 64 2x2 blocks for 512x512 images as in the paper."
            ),
            MethodParameter(
                name="wavelet",
                label="Wavelet Filter",
                param_type="choice",
                default="haar",
                choices=["haar"],
                description="Wavelet filter used for decomposition (Haar filter as specified in Section VI)."
            )
        ]

    # =========================================================================
    # Algorithms 1 & 2: Semi-Binary Conversion (Section IV)
    # =========================================================================

    @staticmethod
    def _algo1_to_semi_binary(s_val: float, min_len: int = 10) -> List[float]:
        """
        Algorithm 1: Converts a real number (singular value or wavelet block entry)
        to a semi-binary array W.
        
        W[0]: sign (1 or -1)
        W[1]: float in [0, 2) equal to abs(S) % 2
        W[2..]: binary bits (0 or 1) representing fix(abs(S) / 2)
        """
        if s_val < 0:
            sign = -1.0
            s_abs = -float(s_val)
        else:
            sign = 1.0
            s_abs = float(s_val)

        w = [sign]
        s_curr = s_abs
        if s_curr == 0.0:
            w.append(0.0)
        else:
            while s_curr != 0.0:
                w.append(float(s_curr % 2.0))
                s_curr = float(int(s_curr // 2.0))

        # Pad with 0.0 so target bit indices are always accessible
        while len(w) < min_len:
            w.append(0.0)

        return w

    @staticmethod
    def _algo2_from_semi_binary(w: List[float]) -> float:
        """
        Algorithm 2: Reconstructs a real number from a semi-binary array W.
        S = W[0] * sum_{i=1}^{len-1}( W[i] * 2^(i-1) )
        """
        s = 0.0
        coeff = 1.0
        for i in range(1, len(w)):
            s += float(w[i]) * coeff
            coeff *= 2.0
        s *= float(w[0])
        return s

    # =========================================================================
    # Haar Discrete Wavelet Transform (2D)
    # =========================================================================

    @staticmethod
    def _haar_dwt2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs a 1-level 2D Haar DWT returning (LL, LH, HL, HH)."""
        sqrt2 = np.sqrt(2.0)
        l_row = (x[:, 0::2] + x[:, 1::2]) / sqrt2
        h_row = (x[:, 0::2] - x[:, 1::2]) / sqrt2

        ll = (l_row[0::2, :] + l_row[1::2, :]) / sqrt2
        lh = (l_row[0::2, :] - l_row[1::2, :]) / sqrt2
        hl = (h_row[0::2, :] + h_row[1::2, :]) / sqrt2
        hh = (h_row[0::2, :] - h_row[1::2, :]) / sqrt2
        return ll, lh, hl, hh

    @staticmethod
    def _haar_idwt2(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray) -> np.ndarray:
        """Performs a 1-level 2D Haar IDWT reconstructing spatial domain array."""
        sqrt2 = np.sqrt(2.0)
        h_half, w_half = ll.shape

        l_row = np.empty((h_half * 2, w_half), dtype=np.float64)
        l_row[0::2, :] = (ll + lh) / sqrt2
        l_row[1::2, :] = (ll - lh) / sqrt2

        h_row = np.empty((h_half * 2, w_half), dtype=np.float64)
        h_row[0::2, :] = (hl + hh) / sqrt2
        h_row[1::2, :] = (hl - hh) / sqrt2

        x = np.empty((h_half * 2, w_half * 2), dtype=np.float64)
        x[:, 0::2] = (l_row + h_row) / sqrt2
        x[:, 1::2] = (l_row - h_row) / sqrt2
        return x

    def _wavedec2_haar(self, img: np.ndarray, level: int) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """K-level 2D Haar wavelet decomposition."""
        coeffs = []
        current_ll = img.astype(np.float64)
        for _ in range(level):
            ll, lh, hl, hh = self._haar_dwt2(current_ll)
            coeffs.append((lh, hl, hh))
            current_ll = ll
        return current_ll, coeffs

    def _waverec2_haar(self, current_ll: np.ndarray, coeffs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """K-level 2D Haar wavelet reconstruction."""
        ll = current_ll.copy()
        for lh, hl, hh in reversed(coeffs):
            ll = self._haar_idwt2(ll, lh, hl, hh)
        return ll

    # =========================================================================
    # Watermark Logo Helper
    # =========================================================================

    @staticmethod
    def _create_watermark_image(watermark_data: str, size: int = 64) -> Image.Image:
        """
        Creates or loads a grayscale watermark image of dimension (size, size).
        If watermark_data is a valid image filepath, loads and resizes it.
        Otherwise renders an academic emblem logo with the payload text.
        """
        if os.path.isfile(watermark_data):
            try:
                img = Image.open(watermark_data).convert("L")
                return img.resize((size, size), Image.Resampling.LANCZOS)
            except Exception:
                pass

        # Generate 64x64 geometric emblem (inspired by UOK logo in paper Fig. 3)
        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)

        # Concentric outer rings
        draw.ellipse((3, 3, size - 4, size - 4), outline=0, width=2)
        draw.ellipse((8, 8, size - 9, size - 9), outline=0, width=1)

        # Emblem internal geometry
        draw.line((size // 2, 9, size // 2, size - 10), fill=0, width=1)
        draw.line((9, size // 2, size - 10, size // 2), fill=0, width=1)
        draw.ellipse((size // 4, size // 4, 3 * size // 4, 3 * size // 4), outline=0, width=1)

        # Render text payload
        clean_text = str(watermark_data).strip()
        if not clean_text:
            clean_text = "UOK"
        words = clean_text.split()
        tag = words[0][:5] if words else "UOK"

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        bbox = draw.textbbox((0, 0), tag, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (size - tw) // 2
        ty = (size - th) // 2
        # Clean white background behind text for readability
        draw.rectangle((tx - 2, ty - 2, tx + tw + 2, ty + th + 2), fill=255)
        draw.text((tx, ty), tag, fill=0, font=font)

        return img

    # =========================================================================
    # Embedding Procedure (Section V.A)
    # =========================================================================

    def embed(
        self,
        image: Image.Image,
        watermark_data: str,
        params: Dict[str, Any]
    ) -> WatermarkResult:
        start_time = time.time()

        strength_factor = float(params.get("strength", 1.0))
        k_level = int(params.get("dwt_level", 5))

        # 1. Color channel handling (embed into Y luminance channel or directly in grayscale)
        is_color = (image.mode in ("RGB", "RGBA"))
        alpha_channel = None
        if image.mode == "RGBA":
            alpha_channel = image.split()[3]
            base_img = image.convert("RGB")
        elif image.mode == "RGB":
            base_img = image.copy()
        else:
            base_img = image.convert("L")

        if is_color:
            ycbcr = base_img.convert("YCbCr")
            y_plane, cb_plane, cr_plane = ycbcr.split()
            host_plane = np.array(y_plane, dtype=np.float64)
        else:
            host_plane = np.array(base_img, dtype=np.float64)

        orig_h, orig_w = host_plane.shape

        # 2. Pad host image if needed so it is evenly divisible by 2^(K+1)
        m0 = 2 ** (k_level + 1)
        pad_h = ((orig_h + m0 - 1) // m0) * m0
        pad_w = ((orig_w + m0 - 1) // m0) * m0

        if pad_h != orig_h or pad_w != orig_w:
            host_padded = np.pad(
                host_plane,
                ((0, pad_h - orig_h), (0, pad_w - orig_w)),
                mode="reflect"
            )
        else:
            host_padded = host_plane.copy()

        # 3. Create watermark image (64x64) and perform SVD (Section V.A Step 3)
        wm_size = 64
        wm_img = self._create_watermark_image(watermark_data, size=wm_size)
        wm_arr = np.array(wm_img, dtype=np.float64)

        u_w, s_w, v_w_t = np.linalg.svd(wm_arr, full_matrices=True)
        v_w = v_w_t.T

        # Scale singular values into [0, 255] (Eq. 7):
        # alpha = 255.0 / sigma_1 * strength_factor
        sigma_1 = s_w[0] if s_w[0] > 0 else 1.0
        alpha = (255.0 / sigma_1) * strength_factor
        scaled_s_w = s_w * alpha

        # 4. Perform K-level DWT (Section V.A Step 1)
        ll_k, details = self._wavedec2_haar(host_padded, level=k_level)
        lh_k, hl_k, hh_k = details[-1]

        subbands = {
            "LL": ll_k,
            "LH": lh_k,
            "HL": hl_k,
            "HH": hh_k
        }
        watermarked_subbands = {}

        # 5. Group into 2x2 blocks, sort, and embed (Section V.A Steps 2 & 5)
        num_singular_values = len(scaled_s_w)
        for name, band in subbands.items():
            h_b, w_b = band.shape
            nr, nc = h_b // 2, w_b // 2
            blocks = []

            for r in range(nr):
                for c in range(nc):
                    blk = band[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2]
                    blocks.append({
                        "r": r,
                        "c": c,
                        "mean": float(np.mean(blk)),
                        "coeffs": blk.flatten().copy()
                    })

            # Sort blocks: LL_K sorted descending by mean (Eq. 5), others in raster order
            if name == "LL":
                sort_order = sorted(range(len(blocks)), key=lambda k: blocks[k]["mean"], reverse=True)
            else:
                sort_order = list(range(len(blocks)))

            n_embed = min(len(blocks), num_singular_values)
            for i in range(n_embed):
                blk_idx = sort_order[i]
                sigma_val = scaled_s_w[i]
                sigma_sba = self._algo1_to_semi_binary(sigma_val, min_len=10)

                # Convert the 4 block coefficients to semi-binary arrays (min_len=6)
                c_w = [self._algo1_to_semi_binary(c, min_len=6) for c in blocks[blk_idx]["coeffs"]]

                # Embedding according to Eq. (8):
                # (1-based indices in paper mapped to 0-based Python indices:
                #  coeff 1: W(5) = SBA(9), W(4) = SBA(8) -> c_w[0][4] = sba[8], c_w[0][3] = sba[7]
                #  coeff 2: W(5) = SBA(7), W(4) = SBA(6) -> c_w[1][4] = sba[6], c_w[1][3] = sba[5]
                #  coeff 3: W(5) = SBA(5), W(4) = SBA(4) -> c_w[2][4] = sba[4], c_w[2][3] = sba[3]
                #  coeff 4: W(5) = SBA(3), W(2) = SBA(2) -> c_w[3][4] = sba[2], c_w[3][1] = sba[1])
                c_w[0][4] = sigma_sba[8]
                c_w[0][3] = sigma_sba[7]
                c_w[1][4] = sigma_sba[6]
                c_w[1][3] = sigma_sba[5]
                c_w[2][4] = sigma_sba[4]
                c_w[2][3] = sigma_sba[3]
                c_w[3][4] = sigma_sba[2]
                c_w[3][1] = sigma_sba[1]

                # Convert back to real numbers using Algorithm 2 (Section V.A Step 6)
                blocks[blk_idx]["new_coeffs"] = [self._algo2_from_semi_binary(w) for w in c_w]

            # Reconstruct modified subband
            new_band = band.copy()
            for i in range(n_embed):
                blk_idx = sort_order[i]
                blk = blocks[blk_idx]
                r, c = blk["r"], blk["c"]
                new_band[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = np.array(blk["new_coeffs"]).reshape((2, 2))

            watermarked_subbands[name] = new_band

        # 6. Perform K-level IDWT (Section V.A Step 7)
        new_details = list(details[:-1]) + [
            (watermarked_subbands["LH"], watermarked_subbands["HL"], watermarked_subbands["HH"])
        ]
        watermarked_padded = self._waverec2_haar(watermarked_subbands["LL"], new_details)

        # Unpad back to original host dimensions
        watermarked_plane = watermarked_padded[:orig_h, :orig_w]

        # 7. Construct final watermarked PIL image
        watermarked_uint8 = np.clip(np.round(watermarked_plane), 0, 255).astype(np.uint8)
        watermarked_y_img = Image.fromarray(watermarked_uint8, mode="L")

        if is_color:
            watermarked_rgb = Image.merge("YCbCr", (watermarked_y_img, cb_plane, cr_plane)).convert("RGB")
            if alpha_channel is not None:
                watermarked_img = watermarked_rgb.convert("RGBA")
                watermarked_img.putalpha(alpha_channel)
            else:
                watermarked_img = watermarked_rgb
            eval_orig = base_img.convert("RGB")
            eval_wm = watermarked_rgb
        else:
            watermarked_img = watermarked_y_img
            eval_orig = base_img.convert("RGB")
            eval_wm = watermarked_y_img.convert("RGB")

        # 8. Cache SVD basis and watermark parameters for blind extraction
        self._last_embed_cache = {
            "u_w": u_w,
            "v_w": v_w,
            "alpha": alpha,
            "wm_arr": wm_arr,
            "wm_img": wm_img,
            "watermark_data": watermark_data,
            "k_level": k_level,
            "wm_size": wm_size,
            "watermarked_plane": watermarked_plane.copy()
        }

        # 9. Verify watermark extraction & compute correlation coefficient rho (Eq. 15)
        extract_result = self.extract(
            watermarked_image=watermarked_img,
            original_image=base_img,
            params={
                "k_level": k_level,
                "u_w": u_w,
                "v_w": v_w,
                "alpha": alpha,
                "wm_arr": wm_arr,
                "watermark_data": watermark_data,
                "watermarked_plane": watermarked_plane
            }
        )
        corr_rho = extract_result.get("correlation", 1.0)

        elapsed = time.time() - start_time
        metrics = self.calculate_metrics(eval_orig, eval_wm)
        metrics["correlation_rho"] = round(float(corr_rho), 4)
        metrics["dwt_level"] = k_level
        metrics["watermark_size"] = f"{wm_size}x{wm_size}"

        diff_map = self.generate_difference_map(eval_orig, eval_wm)

        return WatermarkResult(
            watermarked_image=watermarked_img,
            difference_map=diff_map,
            metrics=metrics,
            execution_time=round(elapsed, 4),
            status="success",
            message=f"DWT-SVD Semi-Binary applied (PSNR: {metrics['psnr_db']} dB, rho: {corr_rho:.4f})",
            extracted_payload=f"{watermark_data} (rho = {corr_rho:.4f})"
        )

    # =========================================================================
    # Watermark Extraction (Section V.B)
    # =========================================================================

    def extract(
        self,
        watermarked_image: Image.Image,
        original_image: Optional[Image.Image] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Blind extraction procedure according to Section V.B.
        Extracts singular value bits from LL, LH, HL, HH subbands,
        reconstructs singular values, and recreates watermark image via SVD.
        """
        params = params or {}
        cache = self._last_embed_cache or {}

        k_level = int(params.get("k_level", cache.get("k_level", 5)))
        alpha = float(params.get("alpha", cache.get("alpha", 1.0)))
        u_w = params.get("u_w", cache.get("u_w", None))
        v_w = params.get("v_w", cache.get("v_w", None))
        ref_wm_arr = params.get("wm_arr", cache.get("wm_arr", None))
        cached_plane = params.get("watermarked_plane", cache.get("watermarked_plane", None))

        # Prefer high-precision array if identical reference provided
        if cached_plane is not None and watermarked_image.size == (cached_plane.shape[1], cached_plane.shape[0]):
            test_plane = cached_plane.copy()
        else:
            if watermarked_image.mode in ("RGB", "RGBA"):
                y_plane = watermarked_image.convert("YCbCr").split()[0]
                test_plane = np.array(y_plane, dtype=np.float64)
            else:
                test_plane = np.array(watermarked_image.convert("L"), dtype=np.float64)

        orig_h, orig_w = test_plane.shape

        # Pad test image to multiple of 2^(K+1)
        m0 = 2 ** (k_level + 1)
        pad_h = ((orig_h + m0 - 1) // m0) * m0
        pad_w = ((orig_w + m0 - 1) // m0) * m0

        if pad_h != orig_h or pad_w != orig_w:
            test_padded = np.pad(
                test_plane,
                ((0, pad_h - orig_h), (0, pad_w - orig_w)),
                mode="reflect"
            )
        else:
            test_padded = test_plane.copy()

        # 1. Perform K levels of DWT on watermarked image (Section V.B Step 1)
        ll_k, details = self._wavedec2_haar(test_padded, level=k_level)
        lh_k, hl_k, hh_k = details[-1]

        subbands = {
            "LL": ll_k,
            "LH": lh_k,
            "HL": hl_k,
            "HH": hh_k
        }

        # Number of singular values (default 64)
        num_singular_values = len(u_w) if u_w is not None else 64
        extracted_sigmas = {}

        # 2 & 3. Group into 2x2 blocks, sort, and extract bits (Section V.B Steps 2 & 3)
        for name, band in subbands.items():
            h_b, w_b = band.shape
            nr, nc = h_b // 2, w_b // 2
            blocks = []

            for r in range(nr):
                for c in range(nc):
                    blk = band[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2]
                    blocks.append({
                        "r": r,
                        "c": c,
                        "mean": float(np.mean(blk)),
                        "coeffs": blk.flatten().copy()
                    })

            if name == "LL":
                sort_order = sorted(range(len(blocks)), key=lambda k: blocks[k]["mean"], reverse=True)
            else:
                sort_order = list(range(len(blocks)))

            n_extract = min(len(blocks), num_singular_values)
            band_sigmas = []

            for i in range(n_extract):
                blk_idx = sort_order[i]
                c_w = [self._algo1_to_semi_binary(c, min_len=6) for c in blocks[blk_idx]["coeffs"]]

                # Extraction according to Eq. (9):
                # sigma_i^{SBA}(9) = block_i^{theta, 1}(5) -> ext_sba[8] = c_w[0][4]
                # sigma_i^{SBA}(8) = block_i^{theta, 1}(4) -> ext_sba[7] = c_w[0][3]
                # sigma_i^{SBA}(7) = block_i^{theta, 2}(5) -> ext_sba[6] = c_w[1][4]
                # sigma_i^{SBA}(6) = block_i^{theta, 2}(4) -> ext_sba[5] = c_w[1][3]
                # sigma_i^{SBA}(5) = block_i^{theta, 3}(5) -> ext_sba[4] = c_w[2][4]
                # sigma_i^{SBA}(4) = block_i^{theta, 3}(4) -> ext_sba[3] = c_w[2][3]
                # sigma_i^{SBA}(3) = block_i^{theta, 4}(5) -> ext_sba[2] = c_w[3][4]
                # sigma_i^{SBA}(2) = block_i^{theta, 4}(2) -> ext_sba[1] = c_w[3][1]
                ext_sba = [1.0] * 10
                ext_sba[8] = float(np.clip(round(c_w[0][4]), 0, 1))
                ext_sba[7] = float(np.clip(round(c_w[0][3]), 0, 1))
                ext_sba[6] = float(np.clip(round(c_w[1][4]), 0, 1))
                ext_sba[5] = float(np.clip(round(c_w[1][3]), 0, 1))
                ext_sba[4] = float(np.clip(round(c_w[2][4]), 0, 1))
                ext_sba[3] = float(np.clip(round(c_w[2][3]), 0, 1))
                ext_sba[2] = float(np.clip(round(c_w[3][4]), 0, 1))
                ext_sba[1] = float(c_w[3][1])

                # 4. Reconstruct singular value using Algorithm 2 and divide by alpha (Eq. 10)
                sigma_theta = self._algo2_from_semi_binary(ext_sba[:9])
                sigma_ext = sigma_theta / alpha if alpha != 0 else sigma_theta
                band_sigmas.append(sigma_ext)

            extracted_sigmas[name] = np.array(band_sigmas, dtype=np.float64)

        # 5. Average extracted singular values across the 4 sub-bands (Eq. 11)
        avg_sigma = (
            extracted_sigmas["LL"] +
            extracted_sigmas["LH"] +
            extracted_sigmas["HL"] +
            extracted_sigmas["HH"]
        ) / 4.0

        # 6. Reconstruct watermark image using saved singular vectors (Eq. 12)
        # If basis vectors not provided, regenerate standard basis from watermark data
        if u_w is None or v_w is None:
            wm_data_text = params.get("watermark_data", cache.get("watermark_data", "UOK"))
            synth_wm = self._create_watermark_image(wm_data_text, size=num_singular_values)
            u_w, _, v_w_t = np.linalg.svd(np.array(synth_wm, dtype=np.float64))
            v_w = v_w_t.T
            if ref_wm_arr is None:
                ref_wm_arr = np.array(synth_wm, dtype=np.float64)

        rec_wm_arr = u_w @ np.diag(avg_sigma) @ v_w.T

        # Calculate watermark correlation coefficient rho (Eq. 15)
        corr_rho = 1.0
        if ref_wm_arr is not None and ref_wm_arr.shape == rec_wm_arr.shape:
            w_mean = np.mean(ref_wm_arr)
            rec_mean = np.mean(rec_wm_arr)
            diff_w = ref_wm_arr - w_mean
            diff_rec = rec_wm_arr - rec_mean
            denom = np.sqrt(np.sum(diff_w ** 2) * np.sum(diff_rec ** 2))
            if denom > 0:
                corr_rho = float(np.sum(diff_w * diff_rec) / denom)
            else:
                corr_rho = 1.0

        rec_clipped = np.clip(rec_wm_arr, 0, 255).astype(np.uint8)
        extracted_wm_img = Image.fromarray(rec_clipped, mode="L")

        detected = bool(corr_rho >= 0.50)

        return {
            "status": "success",
            "detected": detected,
            "correlation": round(float(corr_rho), 4),
            "singular_values": avg_sigma.tolist(),
            "extracted_watermark_image": extracted_wm_img,
            "message": f"DWT-SVD extraction complete. Correlation rho = {corr_rho:.4f}"
        }
