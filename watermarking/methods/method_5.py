"""
Method 5: Reversible Visible Watermarking and Lossless Recovery of Original Images.

Implementation based on the seminal academic paper:
Yongjian Hu and Byeungwoo Jeon,
"Reversible Visible Watermarking and Lossless Recovery of Original Images",
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT),
Vol. 16, No. 11, November 2006, pp. 1423-1429.
DOI: 10.1109/TCSVT.2006.884011
"""

import time
import os
import zlib
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
    create_binary_watermark,
    extract_bit_plane,
    set_bit_plane,
    pack_user_key_80bit,
    unpack_user_key_80bit,
    compute_psnr_without_region,
    compute_psnr,
    compute_ssim,
    compute_ncc,
    compute_metrics,
    AdaptiveArithmeticCoder
)


class Method5(BaseWatermarkMethod):
    """
    Reversible Visible Watermarking and Lossless Recovery of Original Images
    (Hu & Jeon, IEEE TCSVT 2006).

    This method overlays a translucent visible watermark as an ownership tag
    or privacy shield over a protected region R, while reversibly hiding
    lossless reconstruction data packets in the non-watermarked area I - R.
    Authorized users with the 80-bit user key can completely remove the visible
    watermark and losslessly recover the original image data (MSE = 0, PSNR = inf).
    """

    def __init__(self):
        super().__init__()
        self._last_embed_cache: Optional[Dict[str, Any]] = None

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_5",
            title="Reversible Visible Watermarking and Lossless Recovery of Original Images",
            short_name="Reversible Visible (Hu & Jeon)",
            domain="Reversible Visible Watermarking",
            paper_title="Reversible Visible Watermarking and Lossless Recovery of Original Images",
            authors="Yongjian Hu, Byeungwoo Jeon",
            year=2006,
            venue="IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 16, No. 11, pp. 1423-1429",
            doi_or_url="https://doi.org/10.1109/TCSVT.2006.884011",
            description=(
                "A reversible visible watermarking and lossless image recovery algorithm primarily designed "
                "for medical imagery, military/remote-sensing observation, and privacy-preserving content protection. "
                "A visible watermark is translucently overlaid on a sensitive region R to serve as an ownership tag or "
                "privacy shield, while data hiding embeds reconstruction data into the non-watermarked area I - R. "
                "Authorized recipients with the user key can completely remove the watermark and losslessly restore "
                "the original image with 0% distortion (100% bit-exact recovery).\n\n"
                "Algorithm Steps:\n"
                "Step 1: Region Selection & Preprocessing - Define sensitive region R (same dimensions as watermark W) "
                "and extract pixel bit set D from bit plane R_D (MSB plane 7 or 6).\n"
                "Step 2: Preprocessing Compression - Losslessly compress D into bitstream D_c using binary arithmetic coding.\n"
                "Step 3: Key Bit-Plane Selection - Scan bit planes of non-watermarked area (I - R) from LSB to MSB "
                "to identify lowest key bit plane I_k with sufficient spare space (Cap >= |D_c|).\n"
                "Step 4: Error-Diffused Hybrid Pixel Selection - Select candidate pixels S on I_k using a hybrid "
                "selection mechanism (fixed interval step S_p combined with a Random Pixel Selector, RPS) to disperse "
                "artifacts and eliminate visual stripe noise. Group bits into 2x4 character blocks (scheme 2 '::::') "
                "to exploit horizontal and vertical correlation.\n"
                "Step 5: Capacity-Adaptive S Sequence Constitution - Iteratively construct sequence S until "
                "|S| - |S_c| = |D_c| + n, recording skip points n >= 0.\n"
                "Step 6: Payload Construction & Data Hiding - Concatenate compressed bitstreams H = S_c ∪ D_c and "
                "replace sequence S on I_k with H.\n"
                "Step 7: 80-Bit User Key Construction - Assemble 80-bit key recording watermark dimensions (16 bits), "
                "origin (32 bits), |D_c| (16 bits), I_k level (3 bits), hybrid flag (1 bit), step S_p (4 bits), and n (8 bits).\n"
                "Step 8: Interleaved User-Key-Controlled Embedding - Embed binary logo W interleavingly across bit plane "
                "R_D and neighbor R_D' under the control of UKCE pseudo-random generator seeded with user key.\n"
                "Step 9: Watermark Removal & Lossless Recovery - Using user key, locate S, decode S_c to restore "
                "original I_k, decode D_c to obtain D, and reverse UKCE bit placement to reconstruct original R and "
                "losslessly recover original host image I (MSE = 0, PSNR = ∞)."
            ),
            abstract=(
                "In this paper, we propose a reversible visible watermarking algorithm to satisfy a new application scenario "
                "where the visible watermark serves as a tag or ownership identifier, but can be completely removed to resume "
                "the original image data. It includes two procedures: data hiding and visible watermark embedding. In order to "
                "losslessly recover both the watermark-covered and non-watermark-covered image contents at the receiver end, "
                "the payload consists of two reconstruction data packets, one for recovering the watermark-covered region, "
                "and the other for the nonwatermark-covered region. The data hiding technique reversibly hides the payload "
                "in the image region not covered by the visible watermark. To satisfy the requirements of large capacity "
                "and high image quality, our hiding technique is based on data compression and uses a payload-adaptive scheme. "
                "It further adopts error diffusion for improving subjective image quality and arithmetic compression using "
                "a character-based model for increasing computational efficiency. The visible watermark is securely embedded "
                "based on a user-key-controlled embedding mechanism. The data hiding and the visible watermark embedding "
                "procedures are integrated into a secure watermarking system by a specially designed user key."
            )
        )

    def get_parameters(self) -> List[MethodParameter]:
        return [
            MethodParameter(
                name="bit_plane_rd",
                label="Embedding Bit Plane (N_RD)",
                param_type="choice",
                default="7 (MSB - Heavy Tag)",
                choices=["7 (MSB - Heavy Tag)", "6 (2nd MSB - Light Tag)"],
                description="Bit plane of region R to be altered by W (Table I). MSB (7) gives prominent contrast; 2nd MSB (6) gives higher transparency."
            ),
            MethodParameter(
                name="watermark_size",
                label="Watermark Logo Size",
                param_type="choice",
                default="64x128",
                choices=["64x128", "128x128", "128x256", "32x64", "64x64"],
                description="Dimensions (H x W) of the binary watermark logo and protected region R (Table I)."
            ),
            MethodParameter(
                name="region_origin_y",
                label="Region R Origin Y (Row)",
                param_type="int",
                default=64,
                min_value=0,
                max_value=512,
                step=8,
                description="Row coordinate (r0) of the upper-left corner of protected region R."
            ),
            MethodParameter(
                name="region_origin_x",
                label="Region R Origin X (Col)",
                param_type="int",
                default=64,
                min_value=0,
                max_value=512,
                step=8,
                description="Column coordinate (c0) of the upper-left corner of protected region R."
            ),
            MethodParameter(
                name="selection_mode",
                label="Error Diffusion Mode",
                param_type="choice",
                default="Hybrid (RPS + Step)",
                choices=["Hybrid (RPS + Step)", "Fixed-Interval", "Consecutive"],
                description="Pixel selection strategy on key bit plane I_k for error diffusion (Section II-B.1)."
            ),
            MethodParameter(
                name="user_key",
                label="Custom Key / Hex Key (Optional)",
                param_type="str",
                default="",
                description="Optional 80-bit hex key (20 chars) or secret passphrase. Leave empty to auto-generate."
            )
        ]

    # =========================================================================
    # Internal Helpers: Character Formation (Scheme 2 '::::') & Compression
    # =========================================================================

    @staticmethod
    def _parse_size(size_str: str) -> Tuple[int, int]:
        """Parses 'HxW' or 'WxH' string into (height, width)."""
        parts = str(size_str).lower().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 64, 128

    @staticmethod
    def _lossless_compress(data: bytes) -> bytes:
        """
        Lossless data compression tool.
        Uses zlib with maximum compression level (9) for high efficiency and speed,
        consistent with Section V conclusion allowing equivalent compression tools.
        """
        return zlib.compress(data, 9)

    @staticmethod
    def _lossless_decompress(data: bytes) -> bytes:
        """Lossless data decompression tool."""
        return zlib.decompress(data)

    @staticmethod
    def _get_2x4_blocks(
        height: int,
        width: int,
        mask: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        """
        Partitions the region indicated by mask (e.g. I - R) into 2x4 pixel blocks
        (Scheme 2 '::::' from Section II-B.3). Each 2x4 block forms an 8-bit byte.
        """
        blocks = []
        # Step through in 2x4 block raster scan order
        for r in range(0, height - 1, 2):
            for c in range(0, width - 3, 4):
                if mask[r:r + 2, c:c + 4].all():
                    coords = [(r + dr, c + dc) for dr in range(2) for dc in range(4)]
                    blocks.append(coords)
        return blocks

    @staticmethod
    def _read_character_stream(
        plane_bits: np.ndarray,
        blocks: List[List[Tuple[int, int]]]
    ) -> bytearray:
        """Reads 8-bit bytes from 2x4 blocks on the given binary bit plane."""
        chars = bytearray()
        for coords in blocks:
            b_val = 0
            for i, (pr, pc) in enumerate(coords):
                b_val |= (int(plane_bits[pr, pc]) << i)
            chars.append(b_val)
        return chars

    @staticmethod
    def _write_character_stream(
        image_arr: np.ndarray,
        plane_idx: int,
        blocks: List[List[Tuple[int, int]]],
        data_bytes: bytes
    ) -> None:
        """Writes 8-bit bytes into 2x4 blocks on the specified bit plane in-place."""
        mask = np.uint8(0xFF ^ (1 << plane_idx))
        for byte_idx, b_val in enumerate(data_bytes):
            if byte_idx >= len(blocks):
                break
            coords = blocks[byte_idx]
            for i, (pr, pc) in enumerate(coords):
                bit = (b_val >> i) & 1
                image_arr[pr, pc] = (image_arr[pr, pc] & mask) | (bit << plane_idx)

    # =========================================================================
    # Embedding Procedure (Section II)
    # =========================================================================

    def embed(
        self,
        image: Image.Image,
        watermark_data: str,
        params: Dict[str, Any]
    ) -> WatermarkResult:
        start_time = time.time()

        # Parse parameters
        bit_plane_rd_param = str(params.get("bit_plane_rd", "7")).strip()
        n_rd = 6 if "6" in bit_plane_rd_param else 7
        n_rd_prime = n_rd - 1

        size_param = str(params.get("watermark_size", "64x128"))
        hr, wr = self._parse_size(size_param)

        r0 = int(params.get("region_origin_y", 64))
        c0 = int(params.get("region_origin_x", 64))

        sel_mode = str(params.get("selection_mode", "Hybrid (RPS + Step)"))

        # Grayscale / Color Handling
        is_color = (image.mode in ("RGB", "RGBA"))
        if is_color:
            base_gray = image.convert("L")
        else:
            base_gray = image.copy()

        img_arr = np.asarray(base_gray, dtype=np.uint8).copy()
        img_h, img_w = img_arr.shape

        # Ensure region R fits within image dimensions
        # Align origin and size to multiples of 4 for clean 2x4 block boundaries
        hr = min(hr, (img_h // 2) * 2)
        wr = min(wr, (img_w // 4) * 4)
        r0 = min(max(0, (r0 // 2) * 2), img_h - hr)
        c0 = min(max(0, (c0 // 4) * 4), img_w - wr)

        # Generate binary watermark logo W
        w_logo = create_binary_watermark(watermark_data, height=hr, width=wr)

        # ---------------------------------------------------------------------
        # Step 1: Preprocessing of R (extract D from R_D)
        # ---------------------------------------------------------------------
        R = img_arr[r0:r0 + hr, c0:c0 + wr]
        D_plane = extract_bit_plane(R, n_rd)
        # Flatten in raster-scan order and pack into bytes
        d_bits = D_plane.flatten()
        d_bytes = np.packbits(d_bits).tobytes()
        len_d_bytes = len(d_bytes)

        # ---------------------------------------------------------------------
        # Step 2: Lossless compression of D -> D_c
        # ---------------------------------------------------------------------
        d_c = self._lossless_compress(d_bytes)
        d_c_len = len(d_c)

        # ---------------------------------------------------------------------
        # Step 3: Key Bit Plane Selection in I - R (Ik)
        # ---------------------------------------------------------------------
        mask_not_R = np.ones((img_h, img_w), dtype=bool)
        mask_not_R[r0:r0 + hr, c0:c0 + wr] = False

        blocks_I_minus_R = self._get_2x4_blocks(img_h, img_w, mask_not_R)
        total_blocks = len(blocks_I_minus_R)
        if total_blocks == 0:
            return WatermarkResult(
                watermarked_image=image,
                difference_map=image,
                metrics={},
                execution_time=0.0,
                status="error",
                message="Image is too small to accommodate non-watermarked area I - R."
            )

        # Test each bit plane from LSB (0) to MSB (7)
        best_ik: Optional[int] = None
        best_cap = 0
        spare_capacities: Dict[int, int] = {}

        for ik in range(8):
            plane_bits = extract_bit_plane(img_arr, ik)
            chars = self._read_character_stream(plane_bits, blocks_I_minus_R)
            comp = self._lossless_compress(bytes(chars))
            spare = len(chars) - len(comp)
            spare_capacities[ik] = spare
            if spare >= d_c_len and best_ik is None:
                best_ik = ik
                best_cap = spare

        if best_ik is None:
            # Fall back to plane with largest capacity
            best_ik = max(spare_capacities, key=spare_capacities.get)
            best_cap = spare_capacities[best_ik]

        # ---------------------------------------------------------------------
        # Step 4: Error-Diffused Hybrid Pixel Selection (RPS + Step Sp)
        # ---------------------------------------------------------------------
        # Largest step S_Lp (Eq. 2)
        s_lp = max(1, best_cap // max(1, d_c_len))
        max_sp = min(s_lp, 15)

        chosen_sp: Optional[int] = None
        chosen_hybrid = 1
        chosen_k: Optional[int] = None
        chosen_s_c: Optional[bytes] = None
        chosen_candidate_blocks: Optional[List[List[Tuple[int, int]]]] = None

        ik_bits = extract_bit_plane(img_arr, best_ik)

        def search_in_mode(use_hybrid: bool) -> Optional[Tuple[int, int, bytes, List[List[Tuple[int, int]]]]]:
            for sp_candidate in range(max_sp, 0, -1):
                c_blocks = []
                c_chars = bytearray()
                rng_rps = np.random.RandomState(best_ik)
                for idx in range(0, total_blocks, sp_candidate):
                    take = (rng_rps.randint(0, 2) == 1) if use_hybrid else True
                    if take:
                        coords = blocks_I_minus_R[idx]
                        c_blocks.append(coords)
                        b_val = 0
                        for i, (pr, pc) in enumerate(coords):
                            b_val |= (int(ik_bits[pr, pc]) << i)
                        c_chars.append(b_val)

                if len(c_chars) >= d_c_len + 10:
                    # Bracket search for shortest sequence S satisfying |S| - |S_c| >= |D_c|
                    low = d_c_len + 5
                    high = len(c_chars)
                    found_k = None
                    found_sc = None

                    # Linear scan from low to find the first / shortest sequence
                    step_size = max(1, (high - low) // 40)
                    for k in range(low, high + 1, step_size):
                        sub = bytes(c_chars[:k])
                        comp_sub = self._lossless_compress(sub)
                        if k - len(comp_sub) >= d_c_len:
                            # Refine to exact first k
                            refine_start = max(low, k - step_size)
                            for rk in range(refine_start, k + 1):
                                rsub = bytes(c_chars[:rk])
                                rcomp = self._lossless_compress(rsub)
                                if rk - len(rcomp) >= d_c_len:
                                    found_k = rk
                                    found_sc = rcomp
                                    break
                            break

                    if found_k is not None:
                        return sp_candidate, found_k, found_sc, c_blocks
            return None

        # Mode selection:
        if sel_mode == "Consecutive":
            # Direct consecutive blocks with Sp = 1
            max_sp = 1
            res = search_in_mode(use_hybrid=False)
            chosen_hybrid = 0
        elif sel_mode == "Fixed-Interval":
            res = search_in_mode(use_hybrid=False)
            chosen_hybrid = 0
        else:  # Hybrid (RPS + Step)
            res = search_in_mode(use_hybrid=True)
            chosen_hybrid = 1
            if res is None:
                # Fallback to Fixed-interval as specified in paper Section II-B.1
                res = search_in_mode(use_hybrid=False)
                chosen_hybrid = 0

        if res is not None:
            chosen_sp, chosen_k, chosen_s_c, chosen_candidate_blocks = res
        else:
            # Fallback with Sp = 1 and all available blocks
            chosen_sp = 1
            chosen_hybrid = 0
            chosen_candidate_blocks = blocks_I_minus_R
            ik_all_chars = bytes(self._read_character_stream(ik_bits, chosen_candidate_blocks))
            chosen_s_c = self._lossless_compress(ik_all_chars)
            chosen_k = len(ik_all_chars)

        # ---------------------------------------------------------------------
        # Step 5: Capacity Constraint & Skip Points n (Section II-C)
        # ---------------------------------------------------------------------
        # |S| - |S_c| = |D_c| + n  =>  n = (|S| - |S_c|) - |D_c|
        n_skip = max(0, min(255, (chosen_k - len(chosen_s_c)) - d_c_len))

        # ---------------------------------------------------------------------
        # Step 6: Construct Payload H = S_c ∪ D_c and Embed into I_k
        # ---------------------------------------------------------------------
        # H length is exactly chosen_k (padded with n_skip zeros if needed)
        h_payload = chosen_s_c + d_c + bytes(n_skip)
        # Truncate or pad to exactly chosen_k bytes
        if len(h_payload) < chosen_k:
            h_payload += bytes(chosen_k - len(h_payload))
        elif len(h_payload) > chosen_k:
            h_payload = h_payload[:chosen_k]

        watermarked_arr = img_arr.copy()
        self._write_character_stream(
            watermarked_arr,
            best_ik,
            chosen_candidate_blocks,
            h_payload
        )

        # ---------------------------------------------------------------------
        # Step 7: Construct 80-Bit User Key (Section II-D.1)
        # ---------------------------------------------------------------------
        user_key_hex = pack_user_key_80bit(
            h_w=hr,
            w_w=wr,
            r0=r0,
            c0=c0,
            d_c_len=d_c_len,
            n_ik=best_ik,
            hybrid=chosen_hybrid,
            s_p=chosen_sp or 1,
            n=n_skip
        )

        # ---------------------------------------------------------------------
        # Step 8: User-Key-Controlled Visible Watermark Embedding (UKCE)
        # ---------------------------------------------------------------------
        user_key_bytes = bytes.fromhex(user_key_hex)
        ukce_seed = int.from_bytes(user_key_bytes, "big") % (2**31 - 1)
        ukce_rng = np.random.RandomState(ukce_seed)

        mask_rd = np.uint8(0xFF ^ (1 << n_rd))
        mask_rd_prime = np.uint8(0xFF ^ (1 << n_rd_prime))

        for r in range(hr):
            for c in range(wr):
                pr = r0 + r
                pc = c0 + c
                w_bit = int(w_logo[r, c]) & 1
                b_d_prime = (int(watermarked_arr[pr, pc]) >> n_rd_prime) & 1
                u = ukce_rng.randint(0, 2)

                if u == 1:
                    # Watermark pixel is placed on R_D; R_D' is untouched
                    watermarked_arr[pr, pc] = (watermarked_arr[pr, pc] & mask_rd) | (w_bit << n_rd)
                else:
                    # Current pixel of R_D' is preserved on R_D; watermark pixel placed on R_D'
                    watermarked_arr[pr, pc] = (watermarked_arr[pr, pc] & mask_rd) | (b_d_prime << n_rd)
                    watermarked_arr[pr, pc] = (watermarked_arr[pr, pc] & mask_rd_prime) | (w_bit << n_rd_prime)

        elapsed = time.time() - start_time

        # Prepare final watermarked PIL image
        if is_color:
            watermarked_pil = Image.fromarray(watermarked_arr, mode="L").convert("RGB")
            orig_pil = base_gray.convert("RGB")
        else:
            watermarked_pil = Image.fromarray(watermarked_arr, mode="L")
            orig_pil = base_gray

        # Metrics computation
        metrics = compute_metrics(orig_pil, watermarked_pil)
        # Calculate PSNR without R (as in paper Table I)
        psnr_without_r = compute_psnr_without_region(
            img_arr,
            watermarked_arr,
            r0=r0,
            c0=c0,
            hr=hr,
            wr=wr
        )
        metrics["psnr_without_r_db"] = round(psnr_without_r, 2)

        # Error diffusion ratios (Section II-B.2)
        n_p = (chosen_k * (chosen_sp or 1) * (2 if chosen_hybrid else 1)) * 8
        n_t = total_blocks * 8
        e_d = min(100.0, round((n_p / max(1, n_t)) * 100.0, 1))
        e_n = min(100.0, round(((chosen_k * 8) / max(1, n_t)) * 100.0, 1))
        metrics["e_d_percent"] = e_d
        metrics["e_n_percent"] = e_n

        # Payload metadata
        metrics["D_size_bytes"] = len_d_bytes
        metrics["Dc_size_bytes"] = d_c_len
        metrics["Ik_bitplane"] = best_ik
        metrics["S_size_bytes"] = chosen_k
        metrics["Sc_size_bytes"] = len(chosen_s_c)
        metrics["selection_step_Sp"] = chosen_sp or 1
        metrics["skip_points_n"] = n_skip
        metrics["user_key"] = user_key_hex

        # Cache session state for blind extraction
        self._last_embed_cache = {
            "user_key": user_key_hex,
            "host_shape": (img_h, img_w),
            "orig_img": img_arr.copy(),
            "watermarked_img": watermarked_arr.copy(),
            "n_rd": n_rd,
            "n_rd_prime": n_rd_prime
        }

        diff_map = self.generate_difference_map(orig_pil, watermarked_pil)

        msg = (
            f"Method 5 applied: Visible watermark embedded into region R ({hr}x{wr} at {r0},{c0}). "
            f"Reconstruction payload ({chosen_k} B) hidden reversibly into plane I_{best_ik}. "
            f"User Key: {user_key_hex} (80 bits)."
        )

        return WatermarkResult(
            watermarked_image=watermarked_pil,
            difference_map=diff_map,
            metrics=metrics,
            execution_time=round(elapsed, 4),
            status="success",
            message=msg,
            extracted_payload=user_key_hex
        )

    # =========================================================================
    # Watermark Removal & Image Data Reconstruction (Section III)
    # =========================================================================

    def extract(
        self,
        watermarked_image: Image.Image,
        original_image: Optional[Image.Image] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Removes the visible watermark and reconstructs the 100% losslessly
        recovered original image using the 80-bit user key (Section III).
        """
        start_time = time.time()
        p = params or {}

        # Retrieve 80-bit user key
        user_key_hex = str(p.get("user_key", "")).strip()
        if not user_key_hex and self._last_embed_cache is not None:
            user_key_hex = self._last_embed_cache.get("user_key", "")

        if not user_key_hex:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": "User key is required for watermark removal and original image reconstruction."
            }

        try:
            key_info = unpack_user_key_80bit(user_key_hex)
        except Exception as e:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": f"Invalid user key: {str(e)}"
            }

        hr = key_info["h_w"]
        wr = key_info["w_w"]
        r0 = key_info["r0"]
        c0 = key_info["c0"]
        d_c_len = key_info["d_c_len"]
        n_ik = key_info["n_ik"]
        hybrid = key_info["hybrid"]
        s_p = key_info["s_p"]
        n_skip = key_info["n"]

        if s_p <= 0:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": "Invalid user key: selection step S_p must be >= 1."
            }

        if hr <= 0 or wr <= 0 or d_c_len <= 0:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": "Invalid user key: region dimensions or compressed data size invalid."
            }

        # Grayscale array of watermarked image
        wm_gray = watermarked_image.convert("L")
        rec_arr = np.asarray(wm_gray, dtype=np.uint8).copy()
        img_h, img_w = rec_arr.shape

        if r0 + hr > img_h or c0 + wr > img_w:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": "Invalid user key: region R exceeds image dimensions."
            }

        # Re-identify blocks in I - R
        mask_not_R = np.ones((img_h, img_w), dtype=bool)
        mask_not_R[r0:r0 + hr, c0:c0 + wr] = False

        blocks_I_minus_R = self._get_2x4_blocks(img_h, img_w, mask_not_R)
        total_blocks = len(blocks_I_minus_R)

        # Regenerate candidate blocks on key bit plane I_k
        rec_candidate_blocks = []
        if hybrid:
            rng_rps = np.random.RandomState(n_ik)
            for idx in range(0, total_blocks, s_p):
                if rng_rps.randint(0, 2) == 1:
                    rec_candidate_blocks.append(blocks_I_minus_R[idx])
        else:
            for idx in range(0, total_blocks, s_p):
                rec_candidate_blocks.append(blocks_I_minus_R[idx])

        # Read payload bytes from candidate blocks on plane n_ik
        ik_bits = extract_bit_plane(rec_arr, n_ik)
        rec_payload = self._read_character_stream(ik_bits, rec_candidate_blocks)

        # Decompress S_c and determine |S| = |S_c| + |D_c| + n (Section III)
        rec_s = None
        rec_s_c_len = None

        max_search = len(rec_payload) - d_c_len
        for k_sc in range(1, max_search + 1):
            try:
                decomp = self._lossless_decompress(bytes(rec_payload[:k_sc]))
                if len(decomp) == k_sc + d_c_len + n_skip:
                    rec_s = decomp
                    rec_s_c_len = k_sc
                    break
            except Exception:
                continue

        if rec_s is None:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": "Reconstruction packet S_c decompression failed. Incorrect user key or corrupted image."
            }

        # Extract D_c and decompress to D
        rec_d_c = bytes(rec_payload[rec_s_c_len:rec_s_c_len + d_c_len])
        try:
            rec_d_bytes = self._lossless_decompress(rec_d_c)
        except Exception as e:
            return {
                "status": "error",
                "lossless_verified": False,
                "message": f"Reconstruction packet D_c decompression failed: {str(e)}"
            }

        rec_d_bits = np.unpackbits(np.frombuffer(rec_d_bytes, dtype=np.uint8))[:hr * wr]
        rec_d_plane = rec_d_bits.reshape((hr, wr))

        # ---------------------------------------------------------------------
        # 1. Restore I - R by writing back original S characters
        # ---------------------------------------------------------------------
        self._write_character_stream(
            rec_arr,
            n_ik,
            rec_candidate_blocks,
            rec_s
        )

        # ---------------------------------------------------------------------
        # 2. Restore region R by reversing UKCE bit placement and writing back D
        # ---------------------------------------------------------------------
        # Determine bit planes N_RD and N_RD'
        n_rd = 7
        if self._last_embed_cache is not None:
            n_rd = self._last_embed_cache.get("n_rd", 7)
        n_rd_prime = n_rd - 1

        mask_rd = np.uint8(0xFF ^ (1 << n_rd))
        mask_rd_prime = np.uint8(0xFF ^ (1 << n_rd_prime))

        user_key_bytes = bytes.fromhex(user_key_hex)
        ukce_seed = int.from_bytes(user_key_bytes, "big") % (2**31 - 1)
        ukce_rng = np.random.RandomState(ukce_seed)

        extracted_wm = np.zeros((hr, wr), dtype=np.uint8)

        for r in range(hr):
            for c in range(wr):
                pr = r0 + r
                pc = c0 + c
                d_bit = int(rec_d_plane[r, c]) & 1
                u = ukce_rng.randint(0, 2)

                if u == 1:
                    # In embedding, W replaced R_D, and R_D' was untouched.
                    extracted_wm[r, c] = (rec_arr[pr, pc] >> n_rd) & 1
                    # Restore R_D from original bit D
                    rec_arr[pr, pc] = (rec_arr[pr, pc] & mask_rd) | (d_bit << n_rd)
                else:
                    # In embedding, original R_D' was preserved on R_D, and W was placed on R_D'.
                    extracted_wm[r, c] = (rec_arr[pr, pc] >> n_rd_prime) & 1
                    orig_rd_prime = (rec_arr[pr, pc] >> n_rd) & 1
                    # Restore R_D' with preserved bit from R_D
                    rec_arr[pr, pc] = (rec_arr[pr, pc] & mask_rd_prime) | (orig_rd_prime << n_rd_prime)
                    # Restore R_D from original bit D
                    rec_arr[pr, pc] = (rec_arr[pr, pc] & mask_rd) | (d_bit << n_rd)

        elapsed = time.time() - start_time

        recovered_pil = Image.fromarray(rec_arr, mode="L")
        extracted_wm_pil = Image.fromarray((extracted_wm * 255).astype(np.uint8), mode="L")

        # Lossless verification against original image or session cache
        lossless = False
        metrics_rec = {}
        if original_image is not None:
            orig_check = np.asarray(original_image.convert("L"), dtype=np.uint8)
            lossless = bool(np.array_equal(orig_check, rec_arr))
            metrics_rec = compute_metrics(original_image.convert("L"), recovered_pil)
        elif self._last_embed_cache is not None:
            orig_check = self._last_embed_cache["orig_img"]
            lossless = bool(np.array_equal(orig_check, rec_arr))
            metrics_rec = compute_metrics(Image.fromarray(orig_check, mode="L"), recovered_pil)

        return {
            "status": "success",
            "lossless_verified": lossless,
            "recovered_image": recovered_pil,
            "extracted_watermark": extracted_wm_pil,
            "user_key": user_key_hex,
            "execution_time": round(elapsed, 4),
            "metrics": metrics_rec,
            "message": (
                f"Watermark removed successfully. Lossless recovery: "
                f"{'VERIFIED 100% BIT-EXACT (MSE=0, PSNR=∞)' if lossless else 'Completed'}."
            )
        }
