"""
Registry for watermarking methods.
Provides centralized discovery, enumeration, and retrieval of all 6 algorithms.
"""
from typing import Dict, List, Optional
from watermarking.base import BaseWatermarkMethod, MethodMetadata
from watermarking.methods import (
    Method1,
    Method2,
    Method3,
    Method4,
    Method5,
    Method6
)

class WatermarkMethodRegistry:
    """Singleton-style registry managing all active watermarking algorithms."""

    def __init__(self):
        self._methods: Dict[str, BaseWatermarkMethod] = {}
        self._register_default_methods()

    def _register_default_methods(self):
        """Registers the 6 paper placeholder methods."""
        default_instances = [
            Method1(),
            Method2(),
            Method3(),
            Method4(),
            Method5(),
            Method6()
        ]
        for method in default_instances:
            self.register(method)

    def register(self, method: BaseWatermarkMethod):
        """Registers a new method instance."""
        meta = method.get_metadata()
        self._methods[meta.method_id] = method

    def get_method(self, method_id: str) -> Optional[BaseWatermarkMethod]:
        """Retrieves a method by its identifier."""
        return self._methods.get(method_id)

    def list_methods(self) -> List[BaseWatermarkMethod]:
        """Returns all registered methods in order."""
        return list(self._methods.values())

    def list_metadata(self) -> List[MethodMetadata]:
        """Returns metadata for all registered methods."""
        return [m.get_metadata() for m in self._methods.values()]

# Global registry instance
registry = WatermarkMethodRegistry()
