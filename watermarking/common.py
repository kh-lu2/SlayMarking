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
