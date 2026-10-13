"""
Method 1 Placeholder.
Reserved for academic paper #1 implementation.
"""
import time
from PIL import Image, ImageDraw
from typing import Dict, Any, List
from watermarking.base import BaseWatermarkMethod, MethodMetadata, MethodParameter, WatermarkResult

class Method1(BaseWatermarkMethod):
    """Placeholder for Watermarking Method 1 from academic literature."""

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_1",
            title="Method 1",
            short_name="Method 1",
            domain="Research Paper 1",
            paper_title="[Paper 1 Title Placeholder]",
            authors="[Authors TBD]",
            year=2024,
            venue="[Conference / Journal TBD]",
            doi_or_url="",
            description="Placeholder for academic watermarking method 1.",
            abstract="Placeholder implementation for Method 1 from academic literature. Replace this class with the algorithm, parameter set, and citations of your chosen paper."
        )

    def get_parameters(self) -> List[MethodParameter]:
        return [
            MethodParameter(
                name="strength",
                label="Embedding Strength (α)",
                param_type="float",
                default=0.15,
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                description="Placeholder parameter: watermark embedding intensity."
            ),
            MethodParameter(
                name="param_2",
                label="Parameter 2 (Placeholder)",
                param_type="int",
                default=4,
                min_value=1,
                max_value=16,
                step=1,
                description="Placeholder parameter: algorithm tuning parameter."
            )
        ]

    def embed(self, image: Image.Image, watermark_data: str, params: Dict[str, Any]) -> WatermarkResult:
        start_time = time.time()
        strength = float(params.get("strength", 0.15))

        # PLACEHOLDER EMBEDDING:
        # Places a subtle watermark mark on the image
        base_rgb = image.convert("RGB")
        watermarked_img = base_rgb.copy()

        overlay = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = base_rgb.size
        alpha_val = int(min(255, max(25, strength * 220)))
        text = f"[Method 1 Placeholder] {watermark_data}"
        draw.text((w // 2 - 130, h // 2 - 10), text, fill=(0, 245, 212, alpha_val))

        watermarked_img.paste(overlay, (0, 0), overlay)

        elapsed = time.time() - start_time
        metrics = self.calculate_metrics(base_rgb, watermarked_img)
        diff_map = self.generate_difference_map(base_rgb, watermarked_img)

        return WatermarkResult(
            watermarked_image=watermarked_img,
            difference_map=diff_map,
            metrics=metrics,
            execution_time=round(elapsed, 4),
            status="placeholder",
            message="Method 1 placeholder applied. Ready for paper implementation.",
            extracted_payload=watermark_data
        )

    def extract(self, watermarked_image: Image.Image, original_image: Image.Image = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "status": "placeholder",
            "detected": True,
            "message": "Method 1 extraction placeholder."
        }
