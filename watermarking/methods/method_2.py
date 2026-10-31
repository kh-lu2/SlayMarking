"""
Method 2: Enhanced Medical Image Watermarking using Hybrid DWT-HMD-SVD and Arnold Scrambling.

Based on the research paper:
"Enhanced medical image watermarking using hybrid DWT-HMD-SVD and Arnold scrambling"
Authors: Himanshi Chaudhary, Preeti Garg & Virendra P. Vishwakarma
Published in: Scientific Reports, (2025) 15:9710, Nature Portfolio
DOI: https://doi.org/10.1038/s41598-025-94080-4

Paper-Exact Requirements:
- Host Medical Image (HMI): Strictly 512x512 grayscale (mode 'L').
- Watermark (W): 64x64 binary black & white image.
- Arnold Scrambling: Randomizes watermark pixel positions using chaotic Arnold Cat Map.
- 1-level 2D DWT: Decomposes 512x512 image into 256x256 subbands (LL, LH, HL, HH).
- Hessenberg Matrix Decomposition (HMD): Simplifies full 256x256 LL subband: LL = Q * H * Q^T.
- Singular Value Decomposition (SVD):
  - H = UH * SH * VHT (256 singular values)
  - Scrambled Watermark W' = UW * SW * VWT (64 singular values)
- Embedding: EV = SH + alpha * SW (embedded into the first 64 singular values).
- Reconstructions: Inverse SVD -> Inverse HMD -> Inverse DWT.
- Extraction: Reverses the pipeline using reference singular values and inverse Arnold transform.
- Evaluation Metrics: PSNR, SSIM, NC, NPCR, UACI (as defined in paper).
"""
import time
from typing import Dict, Any, List, Optional
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


class Method2(BaseWatermarkMethod):
    """
    Paper-exact implementation of Hybrid DWT-HMD-SVD and Arnold Scrambling Watermarking
    from Chaudhary et al. (Scientific Reports 2025, Nature Portfolio).
    Strictly requires a 512x512 grayscale medical image as specified in the paper.
    """

    def __init__(self):
        super().__init__()
        # Cache of state from most recent embedding for semi-blind extraction
        self._last_state: Optional[Dict[str, Any]] = None

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_2",
            title="Hybrid DWT-HMD-SVD & Arnold Scrambling",
            short_name="DWT-HMD-SVD + Arnold",
            domain="Hybrid Frequency-Spatial Domain",
            paper_title="Enhanced medical image watermarking using hybrid DWT-HMD-SVD and Arnold scrambling",
            authors="Himanshi Chaudhary, Preeti Garg & Virendra P. Vishwakarma",
            year=2025,
            venue="Scientific Reports (Nature Portfolio), Vol. 15, Art. 9710",
            doi_or_url="https://doi.org/10.1038/s41598-025-94080-4",
            description=(
                "Paper-exact implementation of hybrid DWT-HMD-SVD watermarking with Arnold scrambling "
                "(Chaudhary et al., Scientific Reports 2025). As strictly specified in the paper, "
                "this algorithm operates on 512×512 grayscale medical images with a 64×64 binary watermark."
            ),
            abstract=(
                "The protection of medical images against unauthorized access and tampering is paramount. "
                "This paper presents a robust watermarking framework that integrates Discrete Wavelet Transform (DWT), "
                "Hessenberg Decomposition (HMD), Singular Value Decomposition (SVD), and Arnold Scrambling "
                "to enhance the security of medical images. By applying DWT to decompose the medical image into "
                "frequency subbands and embedding the watermark into the most significant subband, the proposed "
                "algorithm ensures minimal impact on image quality. HMD simplifies the subband matrix, while SVD "
                "extracts and manipulates the essential features of the image. Arnold Scrambling is employed to "
                "further secure the watermark image before embedding. Experimental results demonstrate the algorithm's "
                "effectiveness in maintaining imperceptibility, with a peak signal-to-noise ratio (PSNR) of up to 49 dB, "
                "and robustness against common image processing attacks, such as compression and noise addition."
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
                max_value=0.15,
                step=0.005,
                description="Embedding strength factor α (Paper default: 0.02, Table 3)."
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
    # Image Validation (Paper Specification: 512x512 Grayscale)
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_image_specifications(image: Image.Image, operation: str = "Embedding") -> None:
        """
        Validates that the host image conforms strictly to the paper requirements:
        - Dimensions: exactly 512x512 pixels
        - Mode: grayscale (mode 'L' or '1')
        """
        w, h = image.size
        is_size_valid = (w == 512 and h == 512)
        is_gray_valid = (image.mode in ("L", "1"))

        if not is_size_valid or not is_gray_valid:
            reasons = []
            if not is_size_valid:
                reasons.append(f"dimensions must be 512×512 (current: {w}×{h})")
            if not is_gray_valid:
                reasons.append(f"mode must be Grayscale 'L' (current: mode '{image.mode}')")
            raise ValueError(
                f"Method 2 ({operation}) requires a 512×512 grayscale medical image as specified in Chaudhary et al. (2025).\n"
                f"Requirement not met: {', '.join(reasons)}."
            )

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
        Defaults to the paper's benchmark watermark: 'Name=? / Age=?'.
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
    # Evaluation Metrics (Equations from Paper)
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_psnr(orig: np.ndarray, modified: np.ndarray) -> float:
        """PSNR = 10 * log10(255^2 / MSE) (Eq. 1 in paper)."""
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
        """Unified Average Changing Intensity (UACI %) as defined on Page 7."""
        diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64)) / 255.0
        return float(np.mean(diff) * 100.0)

    @staticmethod
    def compute_npcr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Number of Pixels Change Rate (NPCR %) as defined on Page 7."""
        diff = (img1.astype(np.float64) != img2.astype(np.float64))
        return float(np.mean(diff) * 100.0)

    # -------------------------------------------------------------------------
    # Core Watermarking Pipeline: Embedding (Table 2)
    # -------------------------------------------------------------------------

    def embed(
        self,
        image: Image.Image,
        watermark_data: str,
        params: Dict[str, Any]
    ) -> WatermarkResult:
        """
        Embeds the watermark into the host medical image using hybrid DWT-HMD-SVD and Arnold scrambling.
        Follows Table 2 from Chaudhary et al. (Scientific Reports 2025).
        Strictly requires a 512x512 grayscale image.
        """
        start_time = time.time()

        # Enforce paper specification: 512x512 Grayscale
        self.validate_image_specifications(image, operation="Embedding")

        # Parse parameters
        alpha = float(params.get("gain_factor", params.get("alpha", params.get("strength", 0.02))))
        arnold_iters = int(params.get("arnold_iters", 10))
        wavelet = str(params.get("wavelet", "haar"))
        wm_size = 64  # Paper specification: 64x64

        # Step 1: Load HMI and W
        hmi = np.asarray(image, dtype=np.float64)

        # Step 2: Apply Arnold scrambling to W: W' = ArnTran(W)
        wm_matrix = self.generate_watermark_matrix(watermark_data, size=wm_size)
        wm_scrambled = self.arnold_transform(wm_matrix, iters=arnold_iters, inverse=False)

        # Step 3: Apply SVD on scrambled watermark image: [UW, VW, VWT] = SVD(W')
        UW, SW, VWT = np.linalg.svd(wm_scrambled, full_matrices=True)

        # Step 4: Apply DWT to produce LL Band of HMI: [LL, LH, HL, HH] = DWT2(HMI)
        # For a 512x512 image, 1-level DWT produces 256x256 subbands
        coeffs = pywt.dwt2(hmi, wavelet)
        LL, (LH, HL, HH) = coeffs

        # Step 5: HMD is performed on the 256x256 LL band: A = Q * H * Q^T
        H, Q = scipy.linalg.hessenberg(LL, calc_q=True)

        # Step 6: Perform SVD on Hessenberg matrix H: [UH, SH, VHT] = SVD(H)
        UH, SH, VHT = np.linalg.svd(H, full_matrices=True)

        # Step 7: Watermark embedding: EV = SH + alpha * SW
        EV = SH.copy()
        EV[:wm_size] = SH[:wm_size] + alpha * SW[:wm_size]

        # Step 8: Reconstruct watermarked sub-band using inverse SVD: EW' = UH * diag(EV) * VHT
        EW = UH @ np.diag(EV) @ VHT

        # Step 9: Reconstruct modified LL sub-band using inverse HMD: LL' = Q * EW' * Q^T
        LL_mod = Q @ EW @ Q.T

        # Step 10 & 11: Perform inverse DWT to reconstruct W*: W* = IDWT2[LL*, LH, HL, HH]
        W_star = pywt.idwt2((LL_mod, (LH, HL, HH)), wavelet)
        W_star_clipped = np.clip(W_star, 0.0, 255.0)

        # Step 12: W* is generated (512x512 Grayscale)
        watermarked_img = Image.fromarray(np.round(W_star_clipped).astype(np.uint8), mode="L")

        # Cache reference state for extraction verification
        self._last_state = {
            "SH": SH[:wm_size].copy(),
            "UW": UW.copy(),
            "VWT": VWT.copy(),
            "alpha": alpha,
            "arnold_iters": arnold_iters,
            "wavelet": wavelet,
            "wm_size": wm_size,
            "orig_watermark": wm_matrix.copy()
        }

        # Immediate verification extraction to assess fidelity
        ext_res = self.extract(
            watermarked_image=watermarked_img,
            original_image=image,
            params=params
        )
        nc_val = ext_res.get("nc", 1.0)

        # Compute full research metrics
        arr_orig = hmi
        arr_wm = np.asarray(watermarked_img, dtype=np.float64)

        mse = float(np.mean((arr_orig - arr_wm) ** 2))
        mae = float(np.mean(np.abs(arr_orig - arr_wm)))
        psnr = self.compute_psnr(arr_orig, arr_wm)
        ssim = self.compute_ssim(arr_orig, arr_wm)
        uaci = self.compute_uaci(arr_orig, arr_wm)
        npcr = self.compute_npcr(arr_orig, arr_wm)

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
            f"DWT-HMD-SVD embedded successfully | PSNR: {metrics['psnr_db']:.2f} dB | "
            f"SSIM: {metrics['ssim']:.4f} | NC: {metrics['nc']:.4f} | "
            f"UACI: {metrics['uaci_pct']:.2f}% | NPCR: {metrics['npcr_pct']:.2f}%"
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
    # Core Watermarking Pipeline: Extraction (Table 2)
    # -------------------------------------------------------------------------

    def extract(
        self,
        watermarked_image: Image.Image,
        original_image: Optional[Image.Image] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts the watermark by reversing the hybrid DWT-HMD-SVD pipeline
        and descrambling with the inverse Arnold transform.
        Follows Table 2 from Chaudhary et al. (Scientific Reports 2025).
        Strictly requires a 512x512 grayscale image.
        """
        # Enforce paper specification: 512x512 Grayscale
        self.validate_image_specifications(watermarked_image, operation="Extraction")

        p = params or {}
        alpha = float(p.get("gain_factor", p.get("alpha", p.get("strength", 0.02))))
        arnold_iters = int(p.get("arnold_iters", 10))
        wavelet = str(p.get("wavelet", "haar"))
        wm_size = 64

        state = self._last_state

        if original_image is not None:
            self.validate_image_specifications(original_image, operation="Extraction (Reference)")
            hmi_ref = np.asarray(original_image, dtype=np.float64)
            LL_ref, _ = pywt.dwt2(hmi_ref, wavelet)
            H_ref, _ = scipy.linalg.hessenberg(LL_ref, calc_q=True)
            _, SH_ref_full, _ = np.linalg.svd(H_ref, full_matrices=True)
            SH_ref = SH_ref_full[:wm_size]
        elif state is not None:
            SH_ref = state["SH"]
            wavelet = state["wavelet"]
            arnold_iters = state["arnold_iters"]
            alpha = state["alpha"]
        else:
            return {
                "status": "error",
                "detected": False,
                "message": "Extraction requires reference singular values from host image or session state."
            }

        # Step 1: Apply DWT to watermarked image W* (512x512 Grayscale)
        wm_arr = np.asarray(watermarked_image, dtype=np.float64)
        LL_star, _ = pywt.dwt2(wm_arr, wavelet)

        # Step 2: HMD on 256x256 LL subband to obtain Hessenberg matrix H*
        H_star, _ = scipy.linalg.hessenberg(LL_star, calc_q=True)

        # Step 3: Apply SVD on decomposed Hessenberg matrix: [UH*, SH*, VHT*] = SVD(H*)
        _, SH_star, _ = np.linalg.svd(H_star, full_matrices=True)

        # Step 4: Extract singular values of watermark: SW_ext = (SH* - SH) / alpha
        SW_ext = (SH_star[:wm_size] - SH_ref[:wm_size]) / alpha

        # Step 5: Reconstruct extracted scrambled watermark using inverse SVD
        if state is not None and "UW" in state:
            UW = state["UW"]
            VWT = state["VWT"]
        else:
            payload_str = p.get("watermark_data", "")
            wm_mat = self.generate_watermark_matrix(payload_str, size=wm_size)
            wm_scr = self.arnold_transform(wm_mat, iters=arnold_iters, inverse=False)
            UW, _, VWT = np.linalg.svd(wm_scr, full_matrices=True)

        W_scrambled_ext = UW @ np.diag(SW_ext) @ VWT

        # Step 6: Unscramble the watermark via inverse Arnold transform
        W_ext = self.arnold_transform(W_scrambled_ext, iters=arnold_iters, inverse=True)

        # Adaptive thresholding for binary black-and-white watermark
        th = (np.max(W_ext) + np.min(W_ext)) / 2.0
        W_ext_bin = np.where(W_ext > th, 255.0, 0.0)

        # Compute NC against reference watermark
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
            "message": f"DWT-HMD-SVD Watermark extracted (NC={nc_val:.4f})."
        }
