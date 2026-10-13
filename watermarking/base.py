"""
Base architecture for image watermarking algorithms.
Provides standard interfaces, parameter definitions, and evaluation metrics.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import numpy as np
from PIL import Image, ImageOps, ImageChops

@dataclass
class MethodParameter:
    """Represents a tunable parameter for a watermarking algorithm."""
    name: str
    label: str
    param_type: str  # 'float', 'int', 'str', 'bool', 'choice'
    default: Any
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None

@dataclass
class MethodMetadata:
    """Academic paper and method metadata."""
    method_id: str
    title: str
    short_name: str
    domain: str            # e.g., 'Spatial Domain', 'Transform (DCT)', 'Wavelet (DWT)', 'Hybrid (DWT-SVD)', 'Quantization (QIM)', 'Deep Learning'
    paper_title: str
    authors: str
    year: int
    venue: str             # e.g., 'IEEE Transactions on Image Processing'
    doi_or_url: str = ""
    description: str = ""
    abstract: str = ""

@dataclass
class WatermarkResult:
    """Result returned after applying a watermarking method."""
    watermarked_image: Image.Image
    difference_map: Image.Image
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    status: str = "success"
    message: str = "Watermarking completed successfully (placeholder simulation)."
    extracted_payload: Optional[str] = None

class BaseWatermarkMethod(ABC):
    """Abstract base class that all 6 watermarking methods implement."""

    @abstractmethod
    def get_metadata(self) -> MethodMetadata:
        """Returns academic paper metadata and domain information."""
        pass

    @abstractmethod
    def get_parameters(self) -> List[MethodParameter]:
        """Returns the list of configurable parameters for the method."""
        pass

    @abstractmethod
    def embed(
        self,
        image: Image.Image,
        watermark_data: str,
        params: Dict[str, Any]
    ) -> WatermarkResult:
        """
        Embeds the watermark into the host image.
        Placeholder implementation provides simulated watermarking and metrics.
        """
        pass

    @abstractmethod
    def extract(
        self,
        watermarked_image: Image.Image,
        original_image: Optional[Image.Image] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts/verifies the watermark from an image.
        """
        pass

    @staticmethod
    def calculate_metrics(original: Image.Image, watermarked: Image.Image) -> Dict[str, float]:
        """
        Computes standard image quality metrics:
        - Mean Squared Error (MSE)
        - Peak Signal-to-Noise Ratio (PSNR in dB)
        - Mean Absolute Error (MAE)
        """
        orig_rgb = original.convert("RGB")
        wm_rgb = watermarked.convert("RGB")
        
        # Ensure sizes match for metric computation
        if orig_rgb.size != wm_rgb.size:
            wm_rgb = wm_rgb.resize(orig_rgb.size)

        arr1 = np.asarray(orig_rgb, dtype=np.float64)
        arr2 = np.asarray(wm_rgb, dtype=np.float64)

        mse = np.mean((arr1 - arr2) ** 2)
        mae = np.mean(np.abs(arr1 - arr2))

        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))

        # Normalized cross correlation (NCC)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        ncc = float(np.sum(arr1 * arr2) / (norm1 * norm2)) if (norm1 * norm2) > 0 else 1.0

        return {
            "mse": round(float(mse), 4),
            "psnr_db": round(float(psnr), 2) if psnr != float('inf') else 99.99,
            "mae": round(float(mae), 4),
            "ncc": round(ncc, 4),
        }

    @staticmethod
    def generate_difference_map(original: Image.Image, watermarked: Image.Image, amplification: int = 25) -> Image.Image:
        """
        Generates an amplified difference heatmap highlighting modified pixels.
        """
        orig_rgb = original.convert("RGB")
        wm_rgb = watermarked.convert("RGB")
        if orig_rgb.size != wm_rgb.size:
            wm_rgb = wm_rgb.resize(orig_rgb.size)

        diff = ImageChops.difference(orig_rgb, wm_rgb)
        # Amplify subtle differences so users can visually inspect the watermark footprint
        amplified = ImageOps.autocontrast(diff, cutoff=1)
        
        # Colorize difference in electric ocean cyan
        diff_arr = np.asarray(diff, dtype=np.float32)
        magnitude = np.clip(np.mean(diff_arr, axis=2) * amplification, 0, 255).astype(np.uint8)
        
        # Heatmap mapping: black -> deep ocean blue -> electric cyan -> white
        heat_r = np.clip(magnitude * 0.4, 0, 255).astype(np.uint8)
        heat_g = np.clip(magnitude * 1.0, 0, 255).astype(np.uint8)
        heat_b = np.clip(magnitude * 1.3, 0, 255).astype(np.uint8)
        
        heatmap_arr = np.stack([heat_r, heat_g, heat_b], axis=2)
        return Image.fromarray(heatmap_arr, mode="RGB")
