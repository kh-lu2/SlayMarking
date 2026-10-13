"""Watermarking package."""
from .base import (
    BaseWatermarkMethod,
    MethodMetadata,
    MethodParameter,
    WatermarkResult
)
from .registry import registry, WatermarkMethodRegistry

__all__ = [
    "BaseWatermarkMethod",
    "MethodMetadata",
    "MethodParameter",
    "WatermarkResult",
    "registry",
    "WatermarkMethodRegistry"
]
