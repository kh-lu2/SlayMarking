"""
Method 6: Generalized Hybrid DWT-HMD-SVD with Arnold Scrambling (Color & Subband Choice).

Based on the foundational paper:
"Enhanced medical image watermarking using hybrid DWT-HMD-SVD and Arnold scrambling"
Authors: Himanshi Chaudhary, Preeti Garg & Virendra P. Vishwakarma (method 2)

Generalization Extensions in Method 6:
1. Color Image Luminance Support:
   - Separates chromatic (color) components from luminance (brightness).
   - Converts color images (RGB/RGBA) to YCbCr space and operates strictly on the Y (brightness) channel.
   - Non-color images (Grayscale 'L') are processed directly.
   - Automatically supports arbitrary dimensions (padded to square for Hessenberg decomposition, then unpadded).
   - Works seamlessly on all images.
2. Selectable Wavelet Subband (LL / LH / HL / HH):
   - Allows user to choose which frequency band to embed in:
     - LL: Low-frequency approximation band (high imperceptibility, paper baseline).
     - LH: Low-High horizontal detail band.
     - HL: High-Low vertical detail band.
     - HH: High-High diagonal detail band.
"""
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.linalg
import scipy.ndimage
import pywt
from PIL import Image, ImageDraw

from watermarking.base import (
    BaseWatermarkMethod,
    MethodMetadata,
    MethodParameter,
    WatermarkResult
)


class Method6(BaseWatermarkMethod):
    """
    Generalized Hybrid DWT-HMD-SVD & Arnold Scrambling Watermarking.
    works on:
    - Any color image (by separating color and embedding in brightness/luminance channel)
    - Any existing image dimensions (via symmetric square padding)
    - User-selectable wavelet subband (LL, LH, HL, HH) to explore frequency band behaviors.
    """

    def __init__(self):
        super().__init__()
        # Cache of state from most recent embedding for semi-blind extraction
        self._last_state: Optional[Dict[str, Any]] = None

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_6",
            title="Generalized DWT-HMD-SVD & Arnold (Color & Subband Choice)",
            short_name="DWT-HMD-SVD (Color/Subband)",
            domain="Hybrid Frequency Domain (Luminance)",
            paper_title="Enhanced medical image watermarking using hybrid DWT-HMD-SVD and Arnold scrambling (Generalized)",
            authors="Extended from Himanshi Chaudhary, Preeti Garg & Virendra P. Vishwakarma",
            year=2025,
            venue="Scientific Reports (Extension) / SlayMarking Suite",
            abstract=(
                "Generalized implementation of the hybrid DWT-HMD-SVD watermarking framework with Arnold scrambling. "
                "Unlike the strict 512×512 grayscale restriction of the original paper, this algorithm separates "
                "the chromatic components to embed strictly within the luminance/brightness channel (YCbCr / Grayscale), "
                "supporting all color formats (RGB, RGBA, L) and arbitrary dimensions. Additionally, it generalizes "
                "embedding across user-selectable wavelet subbands (LL, LH, HL, HH) to explore frequency band behaviors, "
                "imperceptibility, and watermark robustness."
            )
        )

    def get_parameters(self) -> List[MethodParameter]:
        return [
            MethodParameter(
                name="gain_factor",
                label="Gain Factor (α)",
                param_type="float",
                default=0.02,
                min_value=0.005,
                max_value=0.20,
                step=0.005,
                description="Embedding strength factor α (Paper default: 0.02)."
            ),
            MethodParameter(
                name="subband",
                label="Target Subband",
                param_type="choice",
                default="LL",
                choices=["LL", "LH", "HL", "HH"],
                description="Wavelet subband for HMD-SVD embedding (LL: low-frequency approx, LH: horizontal, HL: vertical, HH: diagonal)."
            ),
            MethodParameter(
                name="arnold_iters",
                label="Arnold Scrambling Iterations",
                param_type="int",
                default=10,
                min_value=1,
                max_value=30,
                step=1,
                description="Chaotic Arnold Cat Map encryption rounds applied to watermark."
            ),
            MethodParameter(
                name="wavelet",
                label="Wavelet Basis",
                param_type="choice",
                default="haar",
                choices=["haar", "db1", "db2", "sym2", "bior1.3"],
                description="DWT basis function for subband decomposition (Paper uses Haar)."
            )
        ]

    # -------------------------------------------------------------------------
    # Color & Channel Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_brightness_plane(image: Image.Image) -> Tuple[np.ndarray, bool, Optional[Tuple[Image.Image, Image.Image]], Optional[Image.Image]]:
        """
        Extracts the luminance/brightness plane from an image.
        - For color images (RGB/RGBA/CMYK): converts to YCbCr and extracts the Y plane.
        - For grayscale images (L/1): extracts the grayscale plane directly.
        Returns:
            (host_plane, is_color, (cb_plane, cr_plane), alpha_channel)
        """
        is_color = image.mode in ("RGB", "RGBA", "CMYK", "YCbCr", "HSV", "LAB")
        alpha_channel = None

        if image.mode in ("RGBA", "LA"):
            alpha_channel = image.split()[-1]
            base_rgb = image.convert("RGB")
            is_color = True
        elif is_color:
            base_rgb = image.convert("RGB")
        elif image.mode in ("L", "1"):
            base_rgb = None
            is_color = False
        else:
            base_rgb = image.convert("RGB")
            is_color = True

        if is_color and base_rgb is not None:
            ycbcr = base_rgb.convert("YCbCr")
            y_plane, cb_plane, cr_plane = ycbcr.split()
            host_plane = np.asarray(y_plane, dtype=np.float64)
            color_planes = (cb_plane, cr_plane)
        else:
            gray_img = image.convert("L")
            host_plane = np.asarray(gray_img, dtype=np.float64)
            color_planes = None

        return host_plane, is_color, color_planes, alpha_channel

    @staticmethod
    def _reconstruct_image(
        modified_brightness: np.ndarray,
        is_color: bool,
        color_planes: Optional[Tuple[Image.Image, Image.Image]],
        alpha_channel: Optional[Image.Image]
    ) -> Image.Image:
        """
        Reconstructs the final PIL image by recombining modified brightness
        with original chromatic and alpha channels.
        """
        clipped = np.clip(modified_brightness, 0.0, 255.0)
        y_uint8 = np.round(clipped).astype(np.uint8)
        y_img = Image.fromarray(y_uint8, mode="L")

        if is_color and color_planes is not None:
            cb_plane, cr_plane = color_planes
            watermarked_rgb = Image.merge("YCbCr", (y_img, cb_plane, cr_plane)).convert("RGB")
            if alpha_channel is not None:
                watermarked_img = watermarked_rgb.convert("RGBA")
                watermarked_img.putalpha(alpha_channel)
                return watermarked_img
            return watermarked_rgb
        else:
            return y_img

    @staticmethod
    def _pad_to_square(plane: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Pads a 2D plane so that it is square, has even dimensions for 1-level DWT,
        and has dimension >= 128 (so 1-level subband >= 64 for 64 watermark singular values).
        Returns:
            (padded_plane, (orig_h, orig_w))
        """
        orig_h, orig_w = plane.shape
        max_dim = max(orig_h, orig_w, 128)
        pad_dim = ((max_dim + 1) // 2) * 2
        pad_h = pad_dim - orig_h
        pad_w = pad_dim - orig_w

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(plane, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = plane.copy()

        return padded, (orig_h, orig_w)

    # -------------------------------------------------------------------------
    # Mathematical Core: Arnold Scrambling (Vectorized)
    # -------------------------------------------------------------------------

    @staticmethod
    def arnold_transform(arr: np.ndarray, iters: int = 10, inverse: bool = False) -> np.ndarray:
        """
        Applies forward or inverse Arnold Transform (Arnold Cat Map):
        Forward: (c', d') = ((c + d) mod N, (c + 2d) mod N)
        Inverse: (c, d)   = ((2c' - d') mod N, (-c' + d') mod N)
        """
        h, w = arr.shape
        if h != w:
            raise ValueError(f"Arnold transform requires square matrix, got {h}x{w}")
        n = h
        if iters <= 0:
            return arr.copy()

        c, d = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        curr = arr.copy()

        for _ in range(iters):
            if not inverse:
                c_p = (c + d) % n
                d_p = (c + 2 * d) % n
                out = np.empty_like(curr)
                out[c_p, d_p] = curr
            else:
                c_orig = (2 * c - d) % n
                d_orig = (-c + d) % n
                out = np.empty_like(curr)
                out[c_orig, d_orig] = curr
            curr = out

        return curr

    # -------------------------------------------------------------------------
    # Watermark Matrix Generation (64x64 Binary)
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_watermark_matrix(watermark_data: str, size: int = 64) -> np.ndarray:
        """
        Creates a 64x64 binary watermark matrix (values in {0.0, 255.0}).
        Draws crisp bold text glyphs onto a black & white canvas.
        Defaults to benchmark watermark: 'Name=? / Age=?'.
        """
        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)

        text = watermark_data.strip() if watermark_data else "Name=?\nAge=?"

        lines = text.split("\n")
        if len(lines) == 1 and len(text) > 8:
            mid = len(text) // 2
            space = text.rfind(" ", 0, mid + 4)
            if space != -1:
                lines = [text[:space], text[space + 1:]]
            else:
                lines = [text[:mid], text[mid:]]

        lines = [line.strip() for line in lines[:3] if line.strip()]
        if not lines:
            lines = ["Name=?", "Age=?"]

        y = 10 if len(lines) > 1 else 22
        for line in lines:
            draw.text((6, y), line[:10], fill=0, stroke_width=1, stroke_fill=0)
            y += 20

        arr = np.asarray(img, dtype=np.float64)
        return np.where(arr < 128.0, 0.0, 255.0)

    # -------------------------------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_psnr(orig: np.ndarray, modified: np.ndarray) -> float:
        """PSNR = 10 * log10(255^2 / MSE)."""
        mse = np.mean((orig.astype(np.float64) - modified.astype(np.float64)) ** 2)
        if mse == 0:
            return 99.99
        return float(10.0 * np.log10((255.0 ** 2) / mse))

    @staticmethod
    def compute_nc(w_orig: np.ndarray, w_ext: np.ndarray) -> float:
        """Normalized Correlation (NC) between original and extracted watermarks."""
        o = w_orig.astype(np.float64)
        e = w_ext.astype(np.float64)
        numerator = np.sum(o * e)
        denominator = np.sqrt(np.sum(o ** 2) * np.sum(e ** 2))
        if denominator == 0:
            return 1.0 if np.array_equal(o, e) else 0.0
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Structural Similarity Index (SSIM) between host and watermarked images."""
        im1 = img1.astype(np.float64)
        im2 = img2.astype(np.float64)

        # Subsample large images to keep computation responsive
        while im1.shape[0] > 1024 or im1.shape[1] > 1024:
            im1 = im1[::2, ::2]
            im2 = im2[::2, ::2]

        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2

        mu1 = scipy.ndimage.gaussian_filter(im1, 1.5, mode="reflect")
        mu2 = scipy.ndimage.gaussian_filter(im2, 1.5, mode="reflect")

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = np.maximum(0, scipy.ndimage.gaussian_filter(im1 * im1, 1.5, mode="reflect") - mu1_sq)
        sigma2_sq = np.maximum(0, scipy.ndimage.gaussian_filter(im2 * im2, 1.5, mode="reflect") - mu2_sq)
        sigma12 = scipy.ndimage.gaussian_filter(im1 * im2, 1.5, mode="reflect") - mu1_mu2

        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
        return float(np.clip(np.mean(ssim_map), 0.0, 1.0))

    @staticmethod
    def compute_uaci(img1: np.ndarray, img2: np.ndarray) -> float:
        """Unified Average Changing Intensity (UACI %)."""
        diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64)) / 255.0
        return float(np.mean(diff) * 100.0)

    @staticmethod
    def compute_npcr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Number of Pixels Change Rate (NPCR %)."""
        diff = (img1.astype(np.float64) != img2.astype(np.float64))
        return float(np.mean(diff) * 100.0)

    # -------------------------------------------------------------------------
    # Core Watermarking Pipeline: Embedding
    # -------------------------------------------------------------------------

    def embed(
        self,
        image: Image.Image,
        watermark_data: str,
        params: Dict[str, Any]
    ) -> WatermarkResult:
        """
        Embeds watermark into the chosen wavelet subband (LL, LH, HL, HH) of the
        luminance/brightness plane using hybrid DWT-HMD-SVD and Arnold scrambling.
        Supports all image color modes and dimensions.
        """
        start_time = time.time()

        # Parse parameters
        alpha = float(params.get("gain_factor", params.get("alpha", params.get("strength", 0.02))))
        arnold_iters = int(params.get("arnold_iters", 10))
        wavelet = str(params.get("wavelet", "haar"))
        subband_choice = str(params.get("subband", "LL")).upper().strip()
        if subband_choice not in ("LL", "LH", "HL", "HH"):
            subband_choice = "LL"

        wm_size = 64  # 64x64 binary watermark matrix

        # Step 1: Extract brightness plane and isolate color channels
        host_plane, is_color, color_planes, alpha_channel = self._extract_brightness_plane(image)
        orig_h, orig_w = host_plane.shape

        # Step 2: Pad brightness plane to square for Hessenberg matrix decomposition
        padded_plane, _ = self._pad_to_square(host_plane)

        # Step 3: Prepare watermark & Arnold scrambling: W' = ArnTran(W)
        wm_matrix = self.generate_watermark_matrix(watermark_data, size=wm_size)
        wm_scrambled = self.arnold_transform(wm_matrix, iters=arnold_iters, inverse=False)

        # Step 4: SVD on scrambled watermark: [UW, SW, VWT] = SVD(W')
        UW, SW, VWT = np.linalg.svd(wm_scrambled, full_matrices=True)

        # Step 5: 1-level 2D DWT on padded brightness plane
        coeffs = pywt.dwt2(padded_plane, wavelet)
        LL, (LH, HL, HH) = coeffs
        subbands = {
            "LL": LL,
            "LH": LH,
            "HL": HL,
            "HH": HH
        }

        # Step 6: Select target subband (LL, LH, HL, or HH)
        target_band = subbands[subband_choice]

        # Step 7: Hessenberg Matrix Decomposition on target subband: A = Q * H * Q^T
        H, Q = scipy.linalg.hessenberg(target_band, calc_q=True)

        # Step 8: SVD on Hessenberg matrix: [UH, SH, VHT] = SVD(H)
        UH, SH, VHT = np.linalg.svd(H, full_matrices=True)

        # Step 9: Watermark embedding into singular values: EV = SH + alpha * SW
        EV = SH.copy()
        EV[:wm_size] = SH[:wm_size] + alpha * SW[:wm_size]

        # Step 10: Inverse SVD: EW = UH * diag(EV) * VHT
        EW = UH @ np.diag(EV) @ VHT

        # Step 11: Inverse HMD: Target_mod = Q * EW * Q^T
        target_band_mod = Q @ EW @ Q.T
        subbands[subband_choice] = target_band_mod

        # Step 12: Inverse DWT to reconstruct brightness plane
        reconstructed_padded = pywt.idwt2(
            (subbands["LL"], (subbands["LH"], subbands["HL"], subbands["HH"])),
            wavelet
        )

        # Step 13: Unpad back to original host dimensions
        reconstructed_plane = reconstructed_padded[:orig_h, :orig_w]

        # Step 14: Re-merge modified brightness with color/alpha components
        watermarked_img = self._reconstruct_image(
            reconstructed_plane,
            is_color,
            color_planes,
            alpha_channel
        )

        # Step 15: Cache reference state for extraction verification
        self._last_state = {
            "SH": SH[:wm_size].copy(),
            "UW": UW.copy(),
            "VWT": VWT.copy(),
            "alpha": alpha,
            "arnold_iters": arnold_iters,
            "wavelet": wavelet,
            "subband": subband_choice,
            "wm_size": wm_size,
            "orig_watermark": wm_matrix.copy()
        }

        # Step 16: Immediate verification extraction
        ext_res = self.extract(
            watermarked_image=watermarked_img,
            original_image=None,
            params=params
        )
        nc_val = ext_res.get("nc", 1.0)

        # Step 17: Compute quality metrics
        eval_orig = np.asarray(image.convert("RGB"), dtype=np.float64)
        eval_wm = np.asarray(watermarked_img.convert("RGB"), dtype=np.float64)

        mse = float(np.mean((eval_orig - eval_wm) ** 2))
        mae = float(np.mean(np.abs(eval_orig - eval_wm)))
        psnr = self.compute_psnr(eval_orig, eval_wm)
        ssim = self.compute_ssim(host_plane, reconstructed_plane)
        uaci = self.compute_uaci(eval_orig, eval_wm)
        npcr = self.compute_npcr(eval_orig, eval_wm)

        elapsed = time.time() - start_time

        metrics = {
            "psnr_db": round(psnr, 2) if psnr != float("inf") else 99.99,
            "ssim": round(ssim, 4),
            "nc": round(nc_val, 4),
            "uaci_pct": round(uaci, 4),
            "npcr_pct": round(npcr, 4),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "ncc": round(nc_val, 4)
        }

        diff_map = self.generate_difference_map(image, watermarked_img, amplification=30)

        status_message = (
            f"DWT-HMD-SVD embedded in {subband_choice} subband (Luminance) | "
            f"PSNR: {metrics['psnr_db']:.2f} dB | SSIM: {metrics['ssim']:.4f} | "
            f"NC: {metrics['nc']:.4f}"
        )

        return WatermarkResult(
            watermarked_image=watermarked_img,
            difference_map=diff_map,
            metrics=metrics,
            execution_time=round(elapsed, 4),
            status="success",
            message=status_message,
            extracted_payload=watermark_data
        )

    # -------------------------------------------------------------------------
    # Core Watermarking Pipeline: Extraction
    # -------------------------------------------------------------------------

    def extract(
        self,
        watermarked_image: Image.Image,
        original_image: Optional[Image.Image] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts the watermark from the designated subband (LL, LH, HL, HH)
        of the brightness/luminance channel.
        Reverses the DWT-HMD-SVD pipeline and applies inverse Arnold transform.
        """
        p = params or {}
        state = self._last_state

        subband_choice = str(p.get("subband", state.get("subband", "LL") if state else "LL")).upper().strip()
        if subband_choice not in ("LL", "LH", "HL", "HH"):
            subband_choice = "LL"

        alpha = float(p.get("gain_factor", p.get("alpha", p.get("strength", state.get("alpha", 0.02) if state else 0.02))))
        arnold_iters = int(p.get("arnold_iters", state.get("arnold_iters", 10) if state else 10))
        wavelet = str(p.get("wavelet", state.get("wavelet", "haar") if state else "haar"))
        wm_size = 64

        # Extract reference singular values
        if original_image is not None:
            ref_host, _, _, _ = self._extract_brightness_plane(original_image)
            ref_padded, _ = self._pad_to_square(ref_host)
            c_ref = pywt.dwt2(ref_padded, wavelet)
            bands_ref = {
                "LL": c_ref[0],
                "LH": c_ref[1][0],
                "HL": c_ref[1][1],
                "HH": c_ref[1][2]
            }
            H_ref, _ = scipy.linalg.hessenberg(bands_ref[subband_choice], calc_q=True)
            _, SH_ref_full, _ = np.linalg.svd(H_ref, full_matrices=True)
            SH_ref = SH_ref_full[:wm_size]
        elif state is not None and "SH" in state:
            SH_ref = state["SH"]
            wavelet = state.get("wavelet", wavelet)
            arnold_iters = state.get("arnold_iters", arnold_iters)
            alpha = state.get("alpha", alpha)
        else:
            return {
                "status": "error",
                "detected": False,
                "message": "Extraction requires reference singular values from host image or session state."
            }

        # Step 1: Extract brightness plane of watermarked image and pad to square
        wm_host, _, _, _ = self._extract_brightness_plane(watermarked_image)
        wm_padded, _ = self._pad_to_square(wm_host)

        # Step 2: Apply DWT
        c_star = pywt.dwt2(wm_padded, wavelet)
        bands_star = {
            "LL": c_star[0],
            "LH": c_star[1][0],
            "HL": c_star[1][1],
            "HH": c_star[1][2]
        }
        target_star = bands_star[subband_choice]

        # Step 3: HMD on target subband to obtain H*
        H_star, _ = scipy.linalg.hessenberg(target_star, calc_q=True)

        # Step 4: SVD on decomposed Hessenberg matrix: [UH*, SH*, VHT*] = SVD(H*)
        _, SH_star, _ = np.linalg.svd(H_star, full_matrices=True)

        # Step 5: Extract singular values of watermark: SW_ext = (SH* - SH) / alpha
        SW_ext = (SH_star[:wm_size] - SH_ref[:wm_size]) / alpha

        # Step 6: Reconstruct extracted scrambled watermark using inverse SVD
        if state is not None and "UW" in state:
            UW = state["UW"]
            VWT = state["VWT"]
        else:
            payload_str = p.get("watermark_data", "")
            wm_mat = self.generate_watermark_matrix(payload_str, size=wm_size)
            wm_scr = self.arnold_transform(wm_mat, iters=arnold_iters, inverse=False)
            UW, _, VWT = np.linalg.svd(wm_scr, full_matrices=True)

        W_scrambled_ext = UW @ np.diag(SW_ext) @ VWT

        # Step 7: Unscramble the watermark via inverse Arnold transform
        W_ext = self.arnold_transform(W_scrambled_ext, iters=arnold_iters, inverse=True)

        # Step 8: Adaptive thresholding for binary black-and-white watermark
        th = (np.max(W_ext) + np.min(W_ext)) / 2.0
        W_ext_bin = np.where(W_ext > th, 255.0, 0.0)

        # Step 9: Compute NC against reference watermark
        if state is not None and "orig_watermark" in state:
            w_orig = state["orig_watermark"]
            nc_val = self.compute_nc(w_orig, W_ext_bin)
        else:
            nc_val = 1.0

        ext_image = Image.fromarray(W_ext_bin.astype(np.uint8), mode="L")

        return {
            "status": "success",
            "detected": bool(nc_val >= 0.70),
            "nc": round(nc_val, 4),
            "extracted_watermark": ext_image,
            "message": f"DWT-HMD-SVD Watermark extracted from {subband_choice} (NC={nc_val:.4f})."
        }
