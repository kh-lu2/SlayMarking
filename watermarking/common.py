"""
Common mathematical transforms, wavelet functions, and utilities for watermarking methods.
Shared across Method 1, Method 2, Method 3, and future algorithms.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont


# =========================================================================
# Haar Discrete Wavelet Transform (2D)
# =========================================================================

def haar_dwt2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a 1-level 2D Haar DWT returning (LL, LH, HL, HH).
    Orthonormal 2D Haar decomposition:
      LL: approximation sub-band (low-low)
      LH: horizontal detail (low-high)
      HL: vertical detail (high-low)
      HH: diagonal detail (high-high)
    """
    sqrt2 = np.sqrt(2.0)
    l_row = (x[:, 0::2] + x[:, 1::2]) / sqrt2
    h_row = (x[:, 0::2] - x[:, 1::2]) / sqrt2

    ll = (l_row[0::2, :] + l_row[1::2, :]) / sqrt2
    lh = (l_row[0::2, :] - l_row[1::2, :]) / sqrt2
    hl = (h_row[0::2, :] + h_row[1::2, :]) / sqrt2
    hh = (h_row[0::2, :] - h_row[1::2, :]) / sqrt2
    return ll, lh, hl, hh


def haar_idwt2(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """
    Performs a 1-level 2D Haar IDWT reconstructing spatial domain array.
    """
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


def wavedec2_haar(img: np.ndarray, level: int) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    K-level 2D Haar wavelet decomposition.
    Returns (LL_K, [(LH_1, HL_1, HH_1), ..., (LH_K, HL_K, HH_K)]).
    """
    coeffs = []
    current_ll = img.astype(np.float64)
    for _ in range(level):
        ll, lh, hl, hh = haar_dwt2(current_ll)
        coeffs.append((lh, hl, hh))
        current_ll = ll
    return current_ll, coeffs


def waverec2_haar(current_ll: np.ndarray, coeffs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    K-level 2D Haar wavelet reconstruction.
    """
    ll = current_ll.copy()
    for lh, hl, hh in reversed(coeffs):
        ll = haar_idwt2(ll, lh, hl, hh)
    return ll


# =========================================================================
# Semi-Binary Conversion (Algorithms 1 & 2 - Makhloghi et al.)
# =========================================================================

def algo1_to_semi_binary(s_val: float, min_len: int = 10) -> List[float]:
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

    while len(w) < min_len:
        w.append(0.0)

    return w


def algo2_from_semi_binary(w: List[float]) -> float:
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
# Watermark Emblem / Image Generator
# =========================================================================

def create_watermark_image(watermark_data: str, size: int = 64) -> Image.Image:
    """
    Creates or loads a grayscale watermark image of dimension (size, size).
    If watermark_data is an existing image filepath, loads and resizes it.
    Otherwise renders an academic emblem logo embedded with the payload text.
    """
    if os.path.isfile(watermark_data):
        try:
            img = Image.open(watermark_data).convert("L")
            return img.resize((size, size), Image.Resampling.LANCZOS)
        except Exception:
            pass

    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    # Concentric outer circles
    draw.ellipse((2, 2, size - 3, size - 3), outline=0, width=2)
    draw.ellipse((6, 6, size - 7, size - 7), outline=0, width=1)

    # Emblem internal grid and geometric dividers
    draw.line((size // 2, 7, size // 2, size - 8), fill=0, width=1)
    draw.line((7, size // 2, size - 8, size // 2), fill=0, width=1)
    draw.ellipse((size // 4, size // 4, 3 * size // 4, 3 * size // 4), outline=0, width=1)

    # Text payload
    clean_text = str(watermark_data).strip()
    if not clean_text:
        clean_text = "SLAY"
    words = clean_text.split()
    tag = words[0][:5] if words else "SLAY"

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    bbox = draw.textbbox((0, 0), tag, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (size - tw) // 2
    ty = (size - th) // 2
    draw.rectangle((tx - 2, ty - 2, tx + tw + 2, ty + th + 2), fill=255)
    draw.text((tx, ty), tag, fill=0, font=font)

    return img


# =========================================================================
# Image Quality Assessment Metrics
# =========================================================================

def compute_psnr(original: np.ndarray, watermarked: np.ndarray, max_val: float = 255.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR in dB)."""
    orig_f = original.astype(np.float64)
    wm_f = watermarked.astype(np.float64)
    mse = np.mean((orig_f - wm_f) ** 2)
    if mse <= 1e-12:
        return 99.99
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes Structural Similarity Index (SSIM) according to Wang et al. (2004)
    using Gaussian filtering as defined in Eq. 10 of the paper.
    """
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    sigma = 1.5
    truncate = 3.5

    mu_x = scipy.ndimage.gaussian_filter(x, sigma=sigma, truncate=truncate)
    mu_y = scipy.ndimage.gaussian_filter(y, sigma=sigma, truncate=truncate)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = scipy.ndimage.gaussian_filter(x * x, sigma=sigma, truncate=truncate) - mu_x_sq
    sigma_y_sq = scipy.ndimage.gaussian_filter(y * y, sigma=sigma, truncate=truncate) - mu_y_sq
    sigma_xy = scipy.ndimage.gaussian_filter(x * y, sigma=sigma, truncate=truncate) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


def compute_ncc(w_orig: np.ndarray, w_ext: np.ndarray) -> float:
    """
    Computes Normalized Cross Correlation (NCC) as defined in Eq. 11 of the paper:
    NCC = sum(w(a,b) * ew(a,b)) / (sqrt(sum(w(a,b)^2)) * sqrt(sum(ew(a,b)^2)))
    """
    w1 = w_orig.astype(np.float64)
    w2 = w_ext.astype(np.float64)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 == 0 or norm2 == 0:
        return 1.0 if np.array_equal(w1, w2) else 0.0
    return float(np.sum(w1 * w2) / (norm1 * norm2))


def compute_metrics(original: Image.Image, watermarked: Image.Image) -> Dict[str, float]:
    """Computes comprehensive quality metrics: MSE, PSNR, SSIM, MAE, NCC."""
    orig_rgb = original.convert("RGB")
    wm_rgb = watermarked.convert("RGB")
    if orig_rgb.size != wm_rgb.size:
        wm_rgb = wm_rgb.resize(orig_rgb.size)

    arr1 = np.asarray(orig_rgb, dtype=np.float64)
    arr2 = np.asarray(wm_rgb, dtype=np.float64)

    mse = float(np.mean((arr1 - arr2) ** 2))
    mae = float(np.mean(np.abs(arr1 - arr2)))
    psnr = compute_psnr(arr1, arr2)

    # Convert to grayscale for SSIM
    gray1 = np.asarray(orig_rgb.convert("L"), dtype=np.float64)
    gray2 = np.asarray(wm_rgb.convert("L"), dtype=np.float64)
    ssim = compute_ssim(gray1, gray2)
    ncc = compute_ncc(arr1, arr2)

    return {
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2) if psnr != float("inf") else 99.99,
        "ssim": round(ssim, 4),
        "mae": round(mae, 4),
        "ncc": round(ncc, 4),
    }


# =========================================================================
# Bit-Plane Manipulation Helpers
# =========================================================================

def extract_bit_plane(image_arr: np.ndarray, plane: int) -> np.ndarray:
    """Extracts a binary bit plane (0 or 1) from an integer image array."""
    return ((image_arr.astype(np.uint8) >> plane) & 1).astype(np.uint8)


def set_bit_plane(image_arr: np.ndarray, plane: int, bits: np.ndarray) -> np.ndarray:
    """
    Overwrites the specified bit plane of image_arr with bits (0 or 1).
    Safely avoids negative overflow in uint8 arithmetic.
    """
    arr = image_arr.astype(np.uint8).copy()
    mask = np.uint8(0xFF ^ (1 << plane))
    arr = (arr & mask) | ((bits.astype(np.uint8) & 1) << plane)
    return arr


# =========================================================================
# Binary Watermark Generator (Crisp Logos / Emblems)
# =========================================================================

def create_binary_watermark(watermark_data: str, height: int = 64, width: int = 128) -> np.ndarray:
    """
    Generates a 2D binary numpy array of shape (height, width) with values {0, 1}.
    If watermark_data is an existing image path, loads, resizes, and binarizes it.
    Otherwise renders a crisp binary text logo / emblem with high contrast.
    """
    if os.path.isfile(watermark_data):
        try:
            img = Image.open(watermark_data).convert("L")
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.uint8)
            th = np.mean(arr)
            return (arr > th).astype(np.uint8)
        except Exception:
            pass

    # Render text logo
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # Outer decorative frame
    draw.rectangle((1, 1, width - 2, height - 2), outline=255, width=1)
    if width >= 32 and height >= 32:
        draw.rectangle((3, 3, width - 4, height - 4), outline=255, width=1)

    # Format text payload
    clean_text = str(watermark_data).strip()
    if not clean_text:
        clean_text = "SKKU"

    lines = clean_text.split("\n")
    if len(lines) == 1 and len(clean_text) > 8:
        mid = len(clean_text) // 2
        lines = [clean_text[:mid].strip(), clean_text[mid:].strip()]

    font = ImageFont.load_default()

    total_text_h = len(lines) * 12
    start_y = max(4, (height - total_text_h) // 2)

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        tx = max(4, (width - tw) // 2)
        ty = start_y + i * 14
        draw.text((tx, ty), line, fill=255, font=font)

    binary_arr = (np.asarray(img, dtype=np.uint8) > 128).astype(np.uint8)
    return binary_arr


# =========================================================================
# Adaptive Order-0 Arithmetic Coder (CACM / Moffat et al. [15, 16])
# =========================================================================

class _FenwickTree:
    """Binary Indexed Tree (Fenwick Tree) for O(log N) cumulative frequencies."""

    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)
        self.counts = [0] * size

    def add(self, idx: int, val: int):
        self.counts[idx] += val
        i = idx + 1
        while i <= self.size:
            self.tree[i] += val
            i += i & (-i)

    def prefix_sum(self, idx: int) -> int:
        if idx < 0:
            return 0
        if idx >= self.size:
            idx = self.size - 1
        s = 0
        i = idx + 1
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def find_symbol(self, target_cum: int) -> int:
        """Finds lowest symbol index with prefix_sum(sym) > target_cum using binary lifting."""
        idx = 0
        current_sum = 0
        power = 1 << (self.size.bit_length() - 1)
        while power > 0:
            next_idx = idx + power
            if next_idx <= self.size and current_sum + self.tree[next_idx] <= target_cum:
                idx = next_idx
                current_sum += self.tree[idx]
            power >>= 1
        return idx


class AdaptiveArithmeticCoder:
    """
    Adaptive order-0 arithmetic coder using 32-bit integer registers and
    a Fenwick tree for O(log 256) cumulative frequency maintenance.
    Appends EOF symbol (256) for deterministic decoding termination without
    external length signaling.
    """

    PRECISION = 32
    MAX_VAL = (1 << PRECISION) - 1
    HALF = 1 << (PRECISION - 1)
    Q1 = 1 << (PRECISION - 2)
    Q3 = 3 * Q1
    EOF_SYM = 256

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        """Encodes arbitrary bytes into an arithmetically compressed byte string."""
        if not data:
            return b""

        ft = _FenwickTree(257)
        for i in range(257):
            ft.add(i, 1)

        low = 0
        high = cls.MAX_VAL
        underflow = 0
        out_bits = []

        def output_bit(b: int):
            nonlocal underflow
            out_bits.append(b)
            while underflow > 0:
                out_bits.append(1 - b)
                underflow -= 1

        for sym in list(data) + [cls.EOF_SYM]:
            total = ft.prefix_sum(cls.EOF_SYM)
            low_cum = ft.prefix_sum(sym - 1)
            high_cum = ft.prefix_sum(sym)
            rng = high - low + 1
            high = low + (rng * high_cum) // total - 1
            low = low + (rng * low_cum) // total
            ft.add(sym, 1)

            while True:
                if high < cls.HALF:
                    output_bit(0)
                    low = (low << 1) & cls.MAX_VAL
                    high = ((high << 1) | 1) & cls.MAX_VAL
                elif low >= cls.HALF:
                    output_bit(1)
                    low = ((low - cls.HALF) << 1) & cls.MAX_VAL
                    high = (((high - cls.HALF) << 1) | 1) & cls.MAX_VAL
                elif low >= cls.Q1 and high < cls.Q3:
                    underflow += 1
                    low = ((low - cls.Q1) << 1) & cls.MAX_VAL
                    high = (((high - cls.Q1) << 1) | 1) & cls.MAX_VAL
                else:
                    break

        underflow += 1
        if low < cls.Q1:
            output_bit(0)
        else:
            output_bit(1)

        out_bytes = bytearray()
        for i in range(0, len(out_bits), 8):
            chunk = out_bits[i:i + 8]
            b_val = 0
            for bit in chunk:
                b_val = (b_val << 1) | bit
            b_val <<= (8 - len(chunk))
            out_bytes.append(b_val)
        return bytes(out_bytes)

    @classmethod
    def decode(cls, enc_bytes: bytes) -> bytes:
        """Decodes arithmetically compressed bytes back to original bytes."""
        if not enc_bytes:
            return b""

        ft = _FenwickTree(257)
        for i in range(257):
            ft.add(i, 1)

        low = 0
        high = cls.MAX_VAL

        in_bits = []
        for b in enc_bytes:
            for shift in range(7, -1, -1):
                in_bits.append((b >> shift) & 1)

        bit_idx = 0
        value = 0
        for _ in range(cls.PRECISION):
            b = in_bits[bit_idx] if bit_idx < len(in_bits) else 0
            bit_idx += 1
            value = (value << 1) | b

        decoded = bytearray()
        while True:
            total = ft.prefix_sum(cls.EOF_SYM)
            rng = high - low + 1
            scaled = ((value - low + 1) * total - 1) // rng
            sym = ft.find_symbol(scaled)

            if sym == cls.EOF_SYM:
                break
            decoded.append(sym)

            low_cum = ft.prefix_sum(sym - 1)
            high_cum = ft.prefix_sum(sym)
            high = low + (rng * high_cum) // total - 1
            low = low + (rng * low_cum) // total
            ft.add(sym, 1)

            while True:
                if high < cls.HALF:
                    pass
                elif low >= cls.HALF:
                    value -= cls.HALF
                    low -= cls.HALF
                    high -= cls.HALF
                elif low >= cls.Q1 and high < cls.Q3:
                    value -= cls.Q1
                    low -= cls.Q1
                    high -= cls.Q1
                else:
                    break
                b = in_bits[bit_idx] if bit_idx < len(in_bits) else 0
                bit_idx += 1
                value = ((value << 1) | b) & cls.MAX_VAL
                low = (low << 1) & cls.MAX_VAL
                high = ((high << 1) | 1) & cls.MAX_VAL

        return bytes(decoded)


# =========================================================================
# 80-Bit User Key Packing / Unpacking (Hu & Jeon 2006, Section II-D.1)
# =========================================================================

def pack_user_key_80bit(
    h_w: int,
    w_w: int,
    r0: int,
    c0: int,
    d_c_len: int,
    n_ik: int,
    hybrid: int,
    s_p: int,
    n: int
) -> str:
    """
    Packs the 80-bit user key from Section II-D.1 Step 7 into a 20-character hex string:
    - Watermark size: 16 bits (h_w: 8 bits, w_w: 8 bits)
    - Origin of region R: 32 bits (r0: 16 bits, c0: 16 bits)
    - Compressed D size |D_c|: 16 bits
    - Key bit plane level N_{I_k}: 3 bits
    - Hybrid/fixed flag: 1 bit
    - Selection step S_p: 4 bits
    - Skip points n: 8 bits
    Total = 80 bits (10 bytes = 20 hex characters).
    """
    key_bytes = bytearray(10)
    key_bytes[0] = h_w & 0xFF
    key_bytes[1] = w_w & 0xFF
    key_bytes[2] = (r0 >> 8) & 0xFF
    key_bytes[3] = r0 & 0xFF
    key_bytes[4] = (c0 >> 8) & 0xFF
    key_bytes[5] = c0 & 0xFF
    key_bytes[6] = (d_c_len >> 8) & 0xFF
    key_bytes[7] = d_c_len & 0xFF
    key_bytes[8] = ((n_ik & 0x7) << 5) | ((hybrid & 0x1) << 4) | (s_p & 0xF)
    key_bytes[9] = n & 0xFF
    return key_bytes.hex().upper()


def unpack_user_key_80bit(key_hex: str) -> Dict[str, int]:
    """
    Unpacks an 80-bit user key (20-character hex string) into its component parameters.
    """
    clean_hex = key_hex.strip().replace(" ", "").replace("0x", "")
    if len(clean_hex) != 20:
        raise ValueError(f"User key must be exactly 80 bits (20 hex characters), got {len(clean_hex)} characters.")

    key_bytes = bytes.fromhex(clean_hex)
    h_w = key_bytes[0]
    w_w = key_bytes[1]
    # Handle 256 dimension wrap-around if 0 is stored for 256
    if h_w == 0:
        h_w = 256
    if w_w == 0:
        w_w = 256

    r0 = (key_bytes[2] << 8) | key_bytes[3]
    c0 = (key_bytes[4] << 8) | key_bytes[5]
    d_c_len = (key_bytes[6] << 8) | key_bytes[7]
    b8 = key_bytes[8]
    n_ik = (b8 >> 5) & 0x7
    hybrid = (b8 >> 4) & 0x1
    s_p = b8 & 0xF
    n = key_bytes[9]

    return {
        "h_w": h_w,
        "w_w": w_w,
        "r0": r0,
        "c0": c0,
        "d_c_len": d_c_len,
        "n_ik": n_ik,
        "hybrid": hybrid,
        "s_p": s_p,
        "n": n
    }


def compute_psnr_without_region(
    original: np.ndarray,
    watermarked: np.ndarray,
    r0: int,
    c0: int,
    hr: int,
    wr: int,
    max_val: float = 255.0
) -> float:
    """
    Computes PSNR excluding the protected watermark region R,
    as reported in Table I of Hu & Jeon 2006.
    """
    orig_f = original.astype(np.float64)
    wm_f = watermarked.astype(np.float64)

    mask = np.ones(orig_f.shape[:2], dtype=bool)
    mask[r0:r0 + hr, c0:c0 + wr] = False

    if not np.any(mask):
        return compute_psnr(orig_f, wm_f, max_val=max_val)

    diff_sq = (orig_f[mask] - wm_f[mask]) ** 2
    mse = float(np.mean(diff_sq))
    if mse <= 1e-12:
        return 99.99
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))

