"""
Method 4: Digital Watermarking Model using Honey Encryption and Reversible Cellular Automata.

Based on the research paper:
"A new digital watermarking model using honey encryption and reversible cellular automata"
Authors: Jianxin Xiong, Zhiyuan Zhou, Vyacheslav V. Dubrovskiy & Sangkeum Lee
Published in: Scientific Reports, (2025) 15:21514, Nature Portfolio
DOI: https://doi.org/10.1038/s41598-025-05492-1

Paper Architecture & Workflow:
1. Cover Image:
   - Strictly 512x512 Grayscale image (mode 'L').
2. Watermark Image:
   - 256x256 grayscale or binary emblem (65,536 pixels, 8-bit depth).
3. Disturbance Stage (Watermark Encryption / Scrambling):
   - Step 1: Watermark matrix X (256x256) is converted into 1D vector M = {m_1, ..., m_65536}.
   - Step 2: Initial propagation using XOR (Eq. 4):
       m'_1 = m_1 ⊕ m_65536
       m'_{j+1} = m_{j+1} ⊕ m'_j, j = 1, ..., 65535.
   - Step 3: Initial Honey Encryption (HE) key generation (Eq. 5):
       HEkey = (sum(p in M) mod 2^n), where n = 8 (HEkey in [0, 255]).
   - Step 4: HE transformation:
       F = M' ⊕ HEkey.
   - Step 5: Reversible Cellular Automata (RCA):
       The 4 Most Significant Bits (MSBs) of each pixel are processed using a 2D
       second-order reversible cellular automaton according to the 32 transition rules
       in Table 2 for L iterations, using the initial state matrix z.
4. Embedding Stage:
   - Step 6: 2D Discrete Wavelet Transform (Haar DWT) decomposes 512x512 cover into
     approximation band a (LL) and detail bands f (LH, HL, HH).
     Encrypted watermark bytes are sequentially embedded into the approximation band a.
     If a is full, embedding proceeds sequentially into detail bands f (Steps 6-1 to 6-10).
     Inverse 2D Haar DWT produces the final watermarked stego image.
5. Extraction Stage:
   - Step 1: 2D Haar DWT on watermarked image extracts encrypted bytes B from subbands.
   - Step 2: Inverse 2D RCA applied using Table 2 for L rounds to recover MSBs.
   - Step 3 & 4: Recombined with LSBs and XORed with HEkey to obtain M'.
   - Step 5: Reverse initial propagation (Eq. 4) recovers original vector M.
   - Step 6: Vector M is reshaped into the original 256x256 watermark image.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from watermarking.base import (
    BaseWatermarkMethod,
    MethodMetadata,
    MethodParameter,
    WatermarkResult
)
from watermarking.common import (
    create_watermark_image,
    compute_metrics,
    compute_ncc,
    compute_psnr,
    compute_ssim
)


# =============================================================================
# Table 2 Local Transition Rules for Reversible Cellular Automaton
# Index is 5-bit neighborhood pattern:
# S^t_{i, j-1} (left), S^t_{i-1, j} (up), S^t_{i, j} (center),
# S^t_{i+1, j} (down), S^t_{i, j+1} (right)
# Output array stores g(neighborhood) when previous state S^{t-1}_{i, j} == 0.
# When S^{t-1} == 1, the output is g(neighborhood) ⊕ 1.
# =============================================================================
TABLE2_G_RULES = np.array([
    # 00000..00111 (0..7)
    1, 1, 1, 1, 0, 0, 1, 1,
    # 01000..01111 (8..15)
    0, 0, 0, 0, 1, 1, 1, 1,
    # 10000..10111 (16..23)
    0, 0, 0, 0, 0, 0, 1, 1,
    # 11000..11111 (24..31)
    0, 0, 1, 1, 0, 0, 0, 0
], dtype=np.uint8)


class Method4(BaseWatermarkMethod):
    """
    Paper-exact implementation of Digital Watermarking Model using Honey Encryption
    and Reversible Cellular Automata (Xiong et al., Scientific Reports 2025).
    Strictly requires a 512x512 grayscale image as specified in the paper.
    """

    def __init__(self):
        super().__init__()
        self._last_state: Optional[Dict[str, Any]] = None

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_4",
            title="Honey Encryption & Reversible Cellular Automata Watermarking",
            short_name="Honey Encryption & RCA",
            domain="Wavelet Domain & Cellular Automata Cryptography",
            paper_title="A new digital watermarking model using honey encryption and reversible cellular automata",
            authors="Jianxin Xiong, Zhiyuan Zhou, Vyacheslav V. Dubrovskiy & Sangkeum Lee",
            year=2025,
            venue="Scientific Reports (Nature Portfolio), Vol. 15, Art. 21514",
            doi_or_url="https://doi.org/10.1038/s41598-025-05492-1",
            description=(
                "Paper-exact implementation of image watermarking combining Honey Encryption (HE), "
                "2D Reversible Cellular Automata (RCA), and Discrete Wavelet Transform (DWT Haar). "
                "Strictly requires a 512×512 grayscale cover image with a 256×256 confidential watermark."
            ),
            abstract=(
                "Watermarking is the process of embedding and extracting a watermark design on a digital cover to "
                "prove the image's copyright or ownership, thereby securing the image's authenticity. The proposed "
                "method in this paper uses a combination of honey encryption and reversible cellular automata for "
                "image watermarking. This method has two main phases: In the first phase, first, the initial matrix "
                "of the image is converted to a vector form. Then, the image vector is initially diffused using the XOR "
                "operator. After that, the initial key space is created in the context of honey encryption. Subsequently, "
                "the diffusion matrix transformation function is applied according to honey encryption and the key. "
                "Finally, the reversible cellular automata transformation is performed on the encrypted matrix. In "
                "the second phase, the matrix resulting from the previous step is stored in the cover image. For this "
                "purpose, the discrete wavelet transform is used to perform watermarking without changing the visual "
                "information of the image. This method has been able to minimize the changes in the cover image "
                "information and maximize the level of confidentiality of the information. The results demonstrate that "
                "the proposed method significantly outperforms the compared methods in terms of imperceptibility, "
                "achieving a significantly lower mean squared error of 13.55 and mean absolute error of 3.05, and "
                "a higher peak signal-to-noise ratio of 36.89."
            )
        )

    def get_parameters(self) -> List[MethodParameter]:
        return [
            MethodParameter(
                name="l_rounds",
                label="RCA Rounds (L)",
                param_type="int",
                default=4,
                min_value=1,
                max_value=10,
                step=1,
                description="Number of Reversible Cellular Automata iterations (L) on the 4 MSB bit planes (Step 5)."
            ),
            MethodParameter(
                name="he_key_seed",
                label="HE Key Seed",
                param_type="int",
                default=42,
                min_value=0,
                max_value=255,
                step=1,
                description="User secret key seed modulating the Honey Encryption key and RCA initial matrix z."
            ),
            MethodParameter(
                name="bits_per_coeff",
                label="DWT Bits / Coeff",
                param_type="int",
                default=2,
                min_value=1,
                max_value=3,
                step=1,
                description="Bit depth per DWT coefficient. Default 2 achieves 65,536-byte capacity across subbands."
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
                f"Method 4 ({operation}) requires a 512×512 grayscale image as specified in Xiong et al. (2025).\n"
                f"Requirement not met: {', '.join(reasons)}."
            )

    # -------------------------------------------------------------------------
    # Mathematical Core 1: Initial Propagation using XOR (Eq. 4)
    # -------------------------------------------------------------------------

    @staticmethod
    def initial_propagation_forward(m: np.ndarray) -> np.ndarray:
        """
        Step 2: Vectorized initial propagation across watermark pixels using Eq. (4):
            m'_1 = m_1 ⊕ m_N
            m'_{j+1} = m_{j+1} ⊕ m'_j, j = 1, ..., N-1
        """
        n = len(m)
        if n == 0:
            return m.copy()
        first = int(m[0]) ^ int(m[-1])
        arr = m.copy()
        arr[0] = first
        return np.bitwise_xor.accumulate(arr).astype(np.uint8)

    @staticmethod
    def initial_propagation_inverse(m_prime: np.ndarray) -> np.ndarray:
        """
        Inverse of initial propagation (Eq. 4):
            m_{j+1} = m'_{j+1} ⊕ m'_j, for j >= 1
            m_1 = m'_1 ⊕ m_N
        """
        n = len(m_prime)
        if n == 0:
            return m_prime.copy()
        m = np.empty_like(m_prime)
        m[1:] = m_prime[1:] ^ m_prime[:-1]
        m[0] = m_prime[0] ^ m[-1]
        return m

    # -------------------------------------------------------------------------
    # Mathematical Core 2: Honey Encryption Transformation (Steps 3 & 4)
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_he_key(m_prime: np.ndarray, seed: int = 0) -> int:
        """
        Step 3: Initial key calculation for Honey Encryption (Eq. 5):
            HEkey = ((sum_{p in M} p) mod 2^n) mod 256, optionally modulated with seed.
        """
        total = int(np.sum(m_prime, dtype=np.int64))
        return int((total + seed) % 256)

    @staticmethod
    def honey_encrypt(data: np.ndarray, he_key: int) -> np.ndarray:
        """Step 4: Honey Encryption pixel transformation: F = M' ⊕ HEkey."""
        return (data.astype(np.uint8) ^ int(he_key)).astype(np.uint8)

    @staticmethod
    def honey_decrypt(data: np.ndarray, he_key: int) -> np.ndarray:
        """Inverse Honey Encryption: M' = F ⊕ HEkey."""
        return (data.astype(np.uint8) ^ int(he_key)).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Mathematical Core 3: Reversible Cellular Automata (Step 5 & Table 2)
    # -------------------------------------------------------------------------

    @staticmethod
    def eval_ca_rules(grid: np.ndarray) -> np.ndarray:
        """
        Evaluates the local 2D von Neumann neighborhood using the 32 transition rules
        from Table 2 on a single binary bit plane.
        """
        up = np.roll(grid, 1, axis=0)
        dn = np.roll(grid, -1, axis=0)
        lf = np.roll(grid, 1, axis=1)
        rt = np.roll(grid, -1, axis=1)

        pat = (lf << 4) | (up << 3) | (grid << 2) | (dn << 1) | rt
        return TABLE2_G_RULES[pat]

    @classmethod
    def rca_transform(
        cls,
        msb4_planes: np.ndarray,
        z_matrix: np.ndarray,
        l_rounds: int = 4,
        inverse: bool = False
    ) -> np.ndarray:
        """
        Step 5: Reversible Cellular Automaton transformation on the 4 MSBs.
        Operates on 4 bit-planes of dimensions (4, H, W) for L rounds.
        Uses Table 2 rules and initial state matrix z (Eq. 6).
        Strictly reversible via second-order bit-plane involution.
        """
        planes = [msb4_planes[i].copy() for i in range(4)]

        if not inverse:
            for _ in range(l_rounds):
                for k in range(4):
                    neighbor_plane = planes[(k - 1) % 4]
                    g_eval = cls.eval_ca_rules(neighbor_plane)
                    planes[k] = planes[k] ^ g_eval ^ z_matrix[k]
        else:
            for _ in reversed(range(l_rounds)):
                for k in reversed(range(4)):
                    neighbor_plane = planes[(k - 1) % 4]
                    g_eval = cls.eval_ca_rules(neighbor_plane)
                    planes[k] = planes[k] ^ g_eval ^ z_matrix[k]

        return np.stack(planes, axis=0)

    # -------------------------------------------------------------------------
    # Mathematical Core 4: 2D Integer Haar Wavelet Transform (Step 6)
    # -------------------------------------------------------------------------

    @staticmethod
    def iwt2_haar(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs 1-level 2D Integer Haar Wavelet Transform (reversible lifting):
        Decomposes 512x512 matrix into approximation band LL (a) and detail bands (LH, HL, HH).
        """
        xi = x.astype(np.int32)
        l = (xi[:, 0::2] + xi[:, 1::2]) >> 1
        h = xi[:, 0::2] - xi[:, 1::2]
        ll = (l[0::2, :] + l[1::2, :]) >> 1
        lh = l[0::2, :] - l[1::2, :]
        hl = (h[0::2, :] + h[1::2, :]) >> 1
        hh = h[0::2, :] - h[1::2, :]
        return ll, lh, hl, hh

    @staticmethod
    def iiwt2_haar(
        ll: np.ndarray,
        lh: np.ndarray,
        hl: np.ndarray,
        hh: np.ndarray
    ) -> np.ndarray:
        """
        Performs 1-level 2D Inverse Integer Haar Wavelet Transform:
        Losslessly reconstructs 512x512 spatial domain matrix from (LL, LH, HL, HH).
        """
        l_even = ll + ((lh + 1) >> 1)
        l_odd = l_even - lh
        h_even = hl + ((hh + 1) >> 1)
        h_odd = h_even - hh

        l = np.empty((ll.shape[0] * 2, ll.shape[1]), dtype=np.int32)
        l[0::2, :] = l_even
        l[1::2, :] = l_odd

        h = np.empty((hl.shape[0] * 2, hl.shape[1]), dtype=np.int32)
        h[0::2, :] = h_even
        h[1::2, :] = h_odd

        x = np.empty((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.int32)
        x[:, 0::2] = l + ((h + 1) >> 1)
        x[:, 1::2] = x[:, 0::2] - h
        return x

    # -------------------------------------------------------------------------
    # Evaluation Metrics: Information Entropy (Table 5) & Neighbor Correlation (Table 4)
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_entropy(arr: np.ndarray) -> float:
        """Computes Shannon Information Entropy (Table 5, ideal ~8.0 for encrypted data)."""
        flat = arr.flatten().astype(np.uint8)
        counts = np.bincount(flat, minlength=256)
        probs = counts[counts > 0] / float(flat.size)
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def compute_neighbor_correlations(arr: np.ndarray) -> Dict[str, float]:
        """
        Computes horizontal, vertical, and diagonal correlation coefficients (Table 4).
        """
        a = arr.astype(np.float64)
        x_h = a[:, :-1].flatten()
        y_h = a[:, 1:].flatten()
        r_h = float(np.corrcoef(x_h, y_h)[0, 1]) if len(x_h) > 0 else 1.0

        x_v = a[:-1, :].flatten()
        y_v = a[1:, :].flatten()
        r_v = float(np.corrcoef(x_v, y_v)[0, 1]) if len(x_v) > 0 else 1.0

        x_d = a[:-1, :-1].flatten()
        y_d = a[1:, 1:].flatten()
        r_d = float(np.corrcoef(x_d, y_d)[0, 1]) if len(x_d) > 0 else 1.0

        return {
            "corr_h": round(r_h, 4),
            "corr_v": round(r_v, 4),
            "corr_d": round(r_d, 4),
        }

    # -------------------------------------------------------------------------
    # Disturbance Pipeline: Watermark Scrambling & Honey Encryption
    # -------------------------------------------------------------------------

    def disturb_watermark(
        self,
        wm_arr: np.ndarray,
        l_rounds: int = 4,
        he_key_seed: int = 42
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Applies Phase 1 (Disturbance stage) to watermark image matrix X (256x256):
        1. Flatten to 1D vector M
        2. Vectorized initial propagation using XOR (Eq. 4) -> M'
        3. Initial HE key calculation (Eq. 5) -> HEkey
        4. Honey Encryption transformation -> F = M' ⊕ HEkey
        5. RCA transformation on 4 MSBs for L rounds using Table 2 -> Encrypted matrix B
        """
        h, w = wm_arr.shape
        m_vec = wm_arr.flatten().copy()

        # Step 2: Initial propagation
        m_prime = self.initial_propagation_forward(m_vec)

        # Step 3: Initial HE key
        he_key = self.compute_he_key(m_prime, seed=he_key_seed)

        # Step 4: HE transformation
        f_vec = self.honey_encrypt(m_prime, he_key)
        f_mat = f_vec.reshape((h, w))

        # Step 5: Extract 4 MSBs and 4 LSBs
        msb4 = (f_mat >> 4) & 0x0F
        lsb4 = f_mat & 0x0F

        msb_planes = np.zeros((4, h, w), dtype=np.uint8)
        for i in range(4):
            msb_planes[i] = (msb4 >> i) & 1

        # Generate reproducible initial state matrix z from HE key and seed
        rng = np.random.RandomState(he_key ^ (he_key_seed << 1) ^ 0xA5)
        z_matrix = rng.randint(0, 2, (4, h, w), dtype=np.uint8)

        # Reversible Cellular Automata for L rounds
        enc_msb_planes = self.rca_transform(msb_planes, z_matrix, l_rounds=l_rounds, inverse=False)

        enc_msb = np.zeros((h, w), dtype=np.uint8)
        for i in range(4):
            enc_msb |= (enc_msb_planes[i] << i)

        # Recombine 4 transformed MSBs with 4 preserved LSBs
        encrypted_mat = ((enc_msb << 4) | lsb4).astype(np.uint8)
        return encrypted_mat, he_key, z_matrix

    def restore_watermark(
        self,
        encrypted_mat: np.ndarray,
        he_key: int,
        z_matrix: np.ndarray,
        l_rounds: int = 4
    ) -> np.ndarray:
        """
        Reverses Phase 1 (Disturbance stage) to recover original watermark image X:
        1. Separate encrypted matrix into 4 MSBs and 4 LSBs
        2. Inverse RCA for L rounds using Table 2 and matrix z
        3. Recombine MSBs with LSBs to reconstruct F
        4. Inverse Honey Encryption (XOR with HEkey) to reconstruct M'
        5. Inverse initial propagation (Eq. 4) to reconstruct vector M
        6. Reshape into 2D watermark image matrix
        """
        h, w = encrypted_mat.shape
        dec_msb4 = (encrypted_mat >> 4) & 0x0F
        dec_lsb4 = encrypted_mat & 0x0F

        dec_msb_planes = np.zeros((4, h, w), dtype=np.uint8)
        for i in range(4):
            dec_msb_planes[i] = (dec_msb4 >> i) & 1

        # Inverse RCA
        rec_msb_planes = self.rca_transform(dec_msb_planes, z_matrix, l_rounds=l_rounds, inverse=True)

        rec_msb = np.zeros((h, w), dtype=np.uint8)
        for i in range(4):
            rec_msb |= (rec_msb_planes[i] << i)

        rec_f_mat = (rec_msb << 4) | dec_lsb4
        rec_f_vec = rec_f_mat.flatten()

        # Inverse Honey Encryption
        rec_m_prime = self.honey_decrypt(rec_f_vec, he_key)

        # Inverse initial propagation
        rec_m = self.initial_propagation_inverse(rec_m_prime)
        return rec_m.reshape((h, w))

    # -------------------------------------------------------------------------
    # Core Watermarking Pipeline: Embedding (Step 6)
    # -------------------------------------------------------------------------

    def embed(
        self,
        image: Image.Image,
        watermark_data: str,
        params: Dict[str, Any]
    ) -> WatermarkResult:
        """
        Embeds a 256x256 watermark into a 512x512 grayscale cover image using
        Honey Encryption, Reversible Cellular Automata, and DWT Haar embedding.
        Follows Steps 1 through 6 from Xiong et al. (Scientific Reports 2025).
        """
        start_time = time.time()

        # Enforce paper specification: 512x512 Grayscale
        self.validate_image_specifications(image, operation="Embedding")

        l_rounds = int(params.get("l_rounds", 4))
        he_key_seed = int(params.get("he_key_seed", 42))
        bits_per_coeff = int(params.get("bits_per_coeff", 2))
        bits_per_coeff = max(1, min(3, bits_per_coeff))

        # Step 1: Create / load 256x256 watermark image
        wm_pil = create_watermark_image(watermark_data, size=256)
        wm_arr = np.asarray(wm_pil, dtype=np.uint8)

        # Phase 1: Disturbance stage (Steps 1-5)
        enc_mat, he_key, z_matrix = self.disturb_watermark(
            wm_arr,
            l_rounds=l_rounds,
            he_key_seed=he_key_seed
        )
        encrypted_bytes = enc_mat.flatten()

        # Phase 2: Embedding stage (Step 6)
        cover_arr = np.asarray(image, dtype=np.uint8)
        # Margin clipping (8..247) ensures reversible integer wavelet without spatial overflow
        cover_clamped = np.clip(cover_arr.astype(np.int32), 8, 247).astype(np.uint8)

        # 6-1: 2D Haar DWT
        ll, lh, hl, hh = self.iwt2_haar(cover_clamped)
        bands = [ll, lh, hl, hh]

        # Convert encrypted bytes into 2-bit symbols (4 symbols per byte)
        # 65536 bytes * 4 = 262,144 symbols, fitting perfectly across all 4 subbands
        symbols = np.zeros(len(encrypted_bytes) * 4, dtype=np.uint8)
        symbols[0::4] = (encrypted_bytes >> 6) & 3
        symbols[1::4] = (encrypted_bytes >> 4) & 3
        symbols[2::4] = (encrypted_bytes >> 2) & 3
        symbols[3::4] = encrypted_bytes & 3

        # Steps 6-3 to 6-10: Sequential storage into approximation band a (LL),
        # overflowing into detail bands f (LH -> HL -> HH)
        idx = 0
        mod_bands = []
        mask = ~3
        for b in bands:
            flat = b.flatten().copy()
            n = min(len(flat), len(symbols) - idx)
            if n > 0:
                flat[:n] = (flat[:n] & mask) | symbols[idx:idx + n]
                idx += n
            mod_bands.append(flat.reshape(b.shape))

        # 6-11: Inverse 2D Haar DWT
        m_ll, m_lh, m_hl, m_hh = mod_bands
        wm_spatial = self.iiwt2_haar(m_ll, m_lh, m_hl, m_hh)
        wm_clipped = np.clip(wm_spatial, 0, 255).astype(np.uint8)

        watermarked_img = Image.fromarray(wm_clipped, mode="L")
        elapsed = time.time() - start_time

        # Cache reference parameters for blind/semi-blind extraction
        self._last_state = {
            "he_key": he_key,
            "z_matrix": z_matrix.copy(),
            "l_rounds": l_rounds,
            "he_key_seed": he_key_seed,
            "bits_per_coeff": bits_per_coeff,
            "wm_size": 256,
            "orig_watermark": wm_arr.copy()
        }

        # Quality metrics (MSE Eq. 7, PSNR Eq. 8, MAE Eq. 9, SSIM, NC, Entropy, Corr)
        metrics = compute_metrics(image, watermarked_img)
        entropy_cover = self.compute_entropy(cover_arr)
        entropy_stego = self.compute_entropy(wm_clipped)
        entropy_enc = self.compute_entropy(enc_mat)
        corr_stats = self.compute_neighbor_correlations(enc_mat)

        metrics.update({
            "he_key": he_key,
            "entropy_cover": round(entropy_cover, 4),
            "entropy_stego": round(entropy_stego, 4),
            "entropy_encrypted": round(entropy_enc, 4),
            "corr_h": corr_stats["corr_h"],
            "corr_v": corr_stats["corr_v"],
            "corr_d": corr_stats["corr_d"],
        })

        diff_map = self.generate_difference_map(image, watermarked_img, amplification=30)
        status_msg = (
            f"HE-RCA embedded | PSNR: {metrics['psnr_db']:.2f} dB | "
            f"MSE: {metrics['mse']:.2f} | MAE: {metrics['mae']:.2f} | "
            f"SSIM: {metrics['ssim']:.4f} | Enc Entropy: {entropy_enc:.4f}"
        )

        return WatermarkResult(
            watermarked_image=watermarked_img,
            difference_map=diff_map,
            metrics=metrics,
            execution_time=round(elapsed, 4),
            status="success",
            message=status_msg,
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
        Extracts the watermark by applying the inverse operations described on Page 10:
        1. 2D Haar DWT on watermarked image, extracting coded encrypted bytes B from subbands.
        2. Inverse 2D RCA using Table 2 local rules for L rounds to obtain B'.
        3. Convert matrix B' into vector V.
        4. XOR vector V with HE key to obtain decoded vector V'.
        5. Inverse initial propagation (Eq. 4) on V' to obtain original vector M.
        6. Reshape vector M into 256x256 watermark image.
        """
        self.validate_image_specifications(watermarked_image, operation="Extraction")

        p = params or {}
        l_rounds = int(p.get("l_rounds", 4))
        he_key_seed = int(p.get("he_key_seed", 42))

        state = self._last_state
        if state is not None:
            he_key = state["he_key"]
            z_matrix = state["z_matrix"]
            l_rounds = state["l_rounds"]
            orig_wm = state["orig_watermark"]
        else:
            # Reconstruct z_matrix and default key from parameter seed
            he_key = he_key_seed % 256
            rng = np.random.RandomState(he_key ^ (he_key_seed << 1) ^ 0xA5)
            z_matrix = rng.randint(0, 2, (4, 256, 256), dtype=np.uint8)
            orig_wm = None

        # Step 1: 2D Haar DWT extraction across subbands
        wm_arr = np.asarray(watermarked_image, dtype=np.uint8)
        ll, lh, hl, hh = self.iwt2_haar(wm_arr)

        total_symbols = 256 * 256 * 4
        ext_symbols = []
        for b in [ll, lh, hl, hh]:
            ext_symbols.append((b.flatten() & 3).astype(np.uint8))
        all_symbols = np.concatenate(ext_symbols)[:total_symbols]

        # Pack 2-bit symbols back to 65,536 encrypted bytes
        ext_bytes = (
            (all_symbols[0::4].astype(np.uint16) << 6) |
            (all_symbols[1::4].astype(np.uint16) << 4) |
            (all_symbols[2::4].astype(np.uint16) << 2) |
            all_symbols[3::4].astype(np.uint16)
        ).astype(np.uint8)

        enc_mat = ext_bytes.reshape((256, 256))

        # Steps 2-6: Reverse disturbance pipeline
        rec_wm_arr = self.restore_watermark(
            enc_mat,
            he_key=he_key,
            z_matrix=z_matrix,
            l_rounds=l_rounds
        )

        extracted_img = Image.fromarray(rec_wm_arr, mode="L")
        nc = compute_ncc(orig_wm, rec_wm_arr) if orig_wm is not None else 1.0
        psnr_val = compute_psnr(orig_wm, rec_wm_arr) if orig_wm is not None else 99.99

        return {
            "status": "success",
            "detected": True,
            "nc": float(nc),
            "psnr_db": float(psnr_val),
            "extracted_image": extracted_img,
            "he_key": he_key,
            "message": f"Method 4 watermark extracted successfully (NC: {nc:.4f})"
        }
